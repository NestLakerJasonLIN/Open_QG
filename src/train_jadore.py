__author__ = 'yanwenl'

import sys
sys.path.append("../evaluate/")

from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

from logger import logger
from params import params
from dataset import Dataset, collate_fn
from torch.utils.tensorboard import SummaryWriter


def prepare_dataloaders(params, data):
    '''
    作用:
    将模型训练集/验证集的输入输出数据构造为batch

    输入参数:
    params: 参数集合
    data: 输入的数据

    输出参数:
    train_loader: 训练集的dataloader
    dev_loader: 验证集的dataloader
    '''

    logger.info('正在从{}中读取数据'.format(params.dataset_dir))

    # 构造train_loader
    train_dataset = Dataset(params, data, mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = True
    )
    logger.info('正在构造train_loader,共有{}个batch'.format(len(train_dataset)))

    # 构造dev_loader
    dev_dataset = Dataset(params, data, mode='dev')
    dev_loader = torch.utils.data.DataLoader(
        dataset = dev_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = False
    )
    logger.info('正在构造dev_loader,共有{}个batch'.format(len(dev_dataset)))

    return train_loader, dev_loader

def train_model(params, vocab, train_loader, dev_loader, model_statistics, writer):
    logger.info('Loading model...')

    model = Transformer(
        len(vocab),
        len(vocab),
        src_pad_idx = vocab.convert_word2index("<pad>"),
        trg_pad_idx = vocab.convert_word2index("<pad>"),
        trg_emb_prj_weight_sharing = params.share_embeddings,
        emb_src_trg_weight_sharing = params.share_embeddings,
        d_k = params.d_k,
        d_v = params.d_v,
        d_model = params.d_model,
        d_word_vec = params.d_model,
        d_inner = params.d_ff,
        n_layers = params.num_layers,
        n_head = params.num_heads,
        dropout = params.dropout).to(params.device)

    # 如果参数中设置了打印模型结构,则打印模型结构
    if params.print_model:
        logger.info(model)

    # 如果参数中设置了加载训练好的模型参数且模型参数文件存在,则加载模型参数
    if params.load_model and os.path.exists(params.checkpoint_file):
        model_params = torch.load(params.checkpoint_file, map_location=params.device)
        model.load_state_dict(model_params)
        logger.info('正在从{}中读取已经训练好的模型参数'.format(params.checkpoint_file))
    else:
        logger.info('没有训练好的模型参数,从头开始训练')
    
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        params.learning_rate, params.d_model, params.warmup_steps)
    
    # curr_epoch = model_statistics["epochs"]
    curr_epoch = 0

    # 每一轮的训练和验证
    for epoch in range(1+curr_epoch, params.num_epochs + 1 + curr_epoch):
        # 一轮模型训练
        model, _, training_total_loss, train_output_list = train_epoch(params, vocab, train_loader, model, optimizer, epoch, model_statistics)
                
        # 一轮模型验证
        model, sentences_pred, dev_total_loss, dev_output_list = eval_epoch(params, vocab, dev_loader, model, optimizer, epoch, model_statistics, mode='dev')

        # 存储每一轮验证集的损失
        model_statistics["training_losses"].append(training_total_loss)
        model_statistics["dev_losses"].append(dev_total_loss)
        model_statistics["epochs"] = epoch

        if training_total_loss < model_statistics["best_training_loss"]:
            model_statistics["best_training_loss"] = training_total_loss
        if dev_total_loss < model_statistics["best_dev_loss"]:
            model_statistics["best_dev_loss"] = dev_total_loss
            model_statistics["best_epoch"] = epoch

        logger.info("{}th epoch, training loss: {} dev loss: {} "
                    "best epoch: {} best training loss: {} best dev loss: {}".format(
            model_statistics["epochs"], training_total_loss, dev_total_loss,
            model_statistics["best_epoch"], model_statistics["best_training_loss"], model_statistics["best_dev_loss"]))

        # 将训练好的模型参数存入本地文件
        if not os.path.exists(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        # temporarily do not choose best model, just collect all epochs
        torch.save(model.state_dict(), params.checkpoint_file)
        logger.info('第{}轮的模型参数已经保存至{}'.format(epoch, params.checkpoint_file))
        torch.save(model_statistics, params.model_statistics_file)
        logger.info('The {}th epoch model_statistics is saved to {}'.format(
            epoch, params.model_statistics_file))

        # calculate BLEU scores
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir)

        def write_output_file(mode, epoch, type, lst):
            orig_filename = params.gold_file if type == "gold" else params.pred_file
            filename = "{}_{}_{}.txt".format(orig_filename.replace(".txt", ""), mode, epoch)

            f = open(filename, 'w')
            for sentence in lst:
                f.write(sentence + '\n')
            f.close()

            return filename

        # 依次写入文件
        epoch_train_pred_file = write_output_file("train", epoch, "pred", train_output_list["pred"])
        epoch_train_gold_file = write_output_file("train", epoch, "gold", train_output_list["gold"])
        epoch_dev_pred_file = write_output_file("dev", epoch, "pred", dev_output_list["pred"])
        epoch_dev_gold_file = write_output_file("dev", epoch, "gold", dev_output_list["gold"])

        bleu = eval(epoch_train_pred_file, epoch_train_gold_file, epoch_train_gold_file,
                          epoch_dev_pred_file, epoch_dev_gold_file, epoch_dev_gold_file)

        model_statistics["bleu"].append(bleu)

        # save score/bleu to tensorboard
        writer.add_scalars('bleu', bleu, epoch)
        # save loss to tensorboard
        writer.add_scalars('loss', {'train': training_total_loss, 'val': dev_total_loss}, epoch)
        writer.flush()

def train_epoch(params, vocab, loader, model, optimizer, epoch, model_statistics):
    ''' Epoch operation in training phase'''
    model.train()

    # 记录训练/验证的总样例数
    total_examples = 0
    # 记录训练/验证的总损失
    total_loss = 0

    model_statistics["sampling_training"][epoch] = []

    # store a list of output sentence for pred and gold
    output_list = {"gold" : [], "pred" : []}

    # 每一个batch的训练/验证
    for batch_index, batch in enumerate(tqdm(loader)):
        # 从数据中读取模型的输入和输出
        input_indices = batch[0].to(params.device)
        output_indices = batch[1].to(params.device)
        if torch.is_tensor(batch[2]):
            answer_indices = batch[2].to(params.device)
        else:
            answer_indices = None
        # input_indices: [batch_size, input_seq_len]
        # output_indices: [batch_size, output_seq_len]
        # answer_indices: [batch_size, output_seq_len]

        # 模型:通过模型输入来预测真实输出,即
        #  <s>  1   2   3
        #  --------------->
        #   1   2   3  </s>
        # 真实输出是在原始数据的基础上"去头"(decoder部分的输出)
        # 原始数据: <s> 1 2 3 </s>
        # 真实输出: 1 2 3 </s>
        output_indices_gold = output_indices[:, 1:]
        # 模型输入是在原始数据的基础上"去尾"(decoder部分的输入)
        # 原始数据: <s> 1 2 3 </s>
        # 真实输出: <s> 1 2 3
        output_indices = output_indices[:, :-1]

        # forward
        optimizer.zero_grad()
        
        # TODO: add answer in model
        output_indices_pred = model(input_indices, output_indices)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            output_indices_pred, output_indices_gold, 
            vocab.convert_word2index("<pad>"), 
            eps=params.label_smoothing_eps, smoothing=params.label_smoothing) 
        
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        num_examples = input_indices.size(0)
        total_examples += num_examples
        total_loss += loss.item() * num_examples
        
        # track prediction
        sentences_pred = track_predictions(params, vocab, epoch, batch_index,
                          input_indices, answer_indices, output_indices,
                          output_indices_pred, output_indices_gold,
                          output_list, model_statistics, "train")

    total_loss = total_loss / total_examples

    return model, sentences_pred, total_loss, output_list

def eval_epoch(params, vocab, loader, model, optimizer, epoch, model_statistics):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    # 记录训练/验证的总样例数
    total_examples = 0
    # 记录训练/验证的总损失
    total_loss = 0

    model_statistics["sampling_training"][epoch] = []

    # store a list of output sentence for pred and gold
    output_list = {"gold" : [], "pred" : []}

    with torch.no_grad():
        # 每一个batch的训练/验证
        for batch_index, batch in enumerate(tqdm(loader)):
            # 从数据中读取模型的输入和输出
            input_indices = batch[0].to(params.device)
            output_indices = batch[1].to(params.device)
            if torch.is_tensor(batch[2]):
                answer_indices = batch[2].to(params.device)
            else:
                answer_indices = None
            # input_indices: [batch_size, input_seq_len]
            # output_indices: [batch_size, output_seq_len]
            # answer_indices: [batch_size, output_seq_len]

            output_indices_gold = output_indices[:, 1:]
            output_indices = output_indices[:, :-1]

            # forward
            optimizer.zero_grad()

            # TODO: add answer in transformer
            output_indices_pred = model(input_indices, output_indices)
            
            loss, n_correct, n_word = cal_performance(
                output_indices_pred, output_indices_gold, 
                vocab.convert_word2index("<pad>"), 
                eps=params.label_smoothing_eps, smoothing=params.label_smoothing)

            # note keeping
            num_examples = input_indices.size(0)
            total_examples += num_examples
            total_loss += loss.item() * num_examples
            
            # track prediction
            sentences_pred = track_predictions(params, vocab, epoch, batch_index,
                              input_indices, answer_indices, output_indices,
                              output_indices_pred, output_indices_gold,
                              output_list, model_statistics, "dev")

    total_loss = total_loss / total_examples

    return model, sentences_pred, total_loss, output_list

def cal_performance(pred, gold, trg_pad_idx, eps, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, eps, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, eps, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def track_predictions(params, vocab, epoch, batch_index,
                      input_indices, answer_indices, output_indices,
                      output_indices_pred, output_indices_gold,
                      output_list, model_statistics, mode):
    # input shape: (batch_size * src_len, vocab_size)
    
    sentences_pred = []

    pred = output_indices_pred.reshape(params.batch_size, -1, len(vocab))    

    # 将基于vocab的概率分布,通过取最大值的方式得到预测的输出序列
    indices_pred = torch.max(pred, dim=-1)[1]
    # indices_pred: [batch_size, output_seq_len]

    # 输出预测序列
    for indices in indices_pred:
        # full: True表示输出完整序列
        #       False表示遇到</s>就停止(只输出到</s>前的序列)
        sentence = vocab.convert_index2sentence(indices, full=False)
        sentences_pred.append(' '.join(sentence))
            
    input_gold = ' '.join(vocab.convert_index2sentence(input_indices[-1]))
    answer = None
    if torch.is_tensor(answer_indices):
        answer = answer_indices[-1] * input_indices[-1]
        answer = ' '.join(vocab.convert_index2sentence(answer, full=True))
    output_gold = ' '.join(vocab.convert_index2sentence(output_indices[-1]))
    output_pred = sentences_pred[-1]

    if params.print_results:
        logger.info('真实输入序列 : {}'.format(input_gold))
        if torch.is_tensor(answer_indices):
            logger.info('真实答案序列 : {}'.format(answer))
        logger.info('真实输出序列 : {}'.format(output_gold))
        logger.info('预测输出序列 : {}'.format(output_pred))

    if batch_index % model_statistics["sampling_frequency"] == 0:
        sample = {
            "epoch" : epoch,
            "input_gold" : input_gold,
            "answer" : answer,
            "output_gold" : output_gold,
            "output_pred" : output_pred
        }
        if mode == "train":
            model_statistics["sampling_training"][epoch].append(sample)
        if mode == "dev":
            model_statistics["sampling_dev"][epoch].append(sample)

    output_list["gold"].append(output_gold)
    output_list["pred"].append(output_pred)

    return sentences_pred

# TODO: change to params

if __name__ == '__main__':

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()

    # save input tensorboard_dir
    writer = SummaryWriter("./runs/" + params.tensorboard_dir)

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(params.temp_pt_file)

    vocab = data['vocab']
    params = data['params']

    # params.d_model = 128
    # params.num_heads = 1
    # params.d_k = 64
    # params.dropout = 0.5
    # params.num_layers = 2
    # params.num_epochs = 5

    model_statistics = torch.load(params.model_statistics_file)

    print("model statistics: \n{}\n".format(model_statistics))

    params.device = "cpu"

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    train_loader, dev_loader = prepare_dataloaders(params, data)

    # 训练模型
    train_model(params, vocab, train_loader, dev_loader, model_statistics, writer)

    writer.close()