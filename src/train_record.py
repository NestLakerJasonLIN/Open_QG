#!/usr/bin/env python
# encoding: utf-8
'''
数据训练:
(1)将模型训练集/验证集的输入输出数据构造为batch
(2)将训练集数据输入模型,进行模型训练
(3)将验证集数据输入模型,进行模型验证
(4)根据验证集损失最小的原则,选择训练最好的模型参数进行保存
'''
__author__ = 'yanwenl'

import sys
sys.path.append('evaluate')

import json
import os
import shutil
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from logger import logger
from params import params
from vocab import Vocab
from dataset import Dataset
from optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from eval import eval as bleu_eval
from beam import Generator
from torch.nn.utils.rnn import pad_sequence


class Sample:
    def __init__(self, batch):
        input_indices, output_indices, answers_indices, vocab = zip(*batch)

        # input
        self.inputs_lens = [len(x) for x in input_indices]
        self.inputs = pad_sequence([torch.tensor(x) for x in input_indices], batch_first=True)

        # output/target
        outputs_text = [torch.tensor(text[:-1]) for text in output_indices]
        targets_text = [torch.tensor(text[1:]) for text in output_indices]

        self.outputs_lens = [y.size(0) for y in outputs_text]
        self.targets_lens = [y.size(0) for y in targets_text]

        self.outputs = pad_sequence(outputs_text, batch_first=True)
        self.targets = pad_sequence(targets_text, batch_first=True)

        # answer
        self.answers = torch.zeros_like(self.inputs)

        for index, single_data in enumerate(answers_indices):
            answer = single_data
            # shift by 1
            answer_start = answer[0] + 1
            answer_end = answer[1] + 1
            self.answers[index][answer_start: answer_end] = 1

    def pin_memory(self):
        self.inputs = self.inputs.pin_memory()
        self.outputs = self.outputs.pin_memory()
        self.targets = self.targets.pin_memory()
        return self

def collate_fn(batch):
    return Sample(batch)

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
    '''
    作用:
    训练和验证模型

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    train_loader: 训练集的dataloader
    dev_loader: 验证集的dataloader
    '''

    logger.info('正在加载模型,即将开始训练')

    # 定义模型
    model = Model(params, vocab).to(params.device)

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

    # 定义优化器
    optimizer = Optimizer(params, model)

    # # 存储每一轮验证集的损失,根据验证集损失最小来挑选最好的模型进行保存
    # total_loss_epochs = []

    curr_epoch = model_statistics["epochs"]

    # 每一轮的训练和验证
    for epoch in range(1+curr_epoch, params.num_epochs + 1 + curr_epoch):
        # 一轮模型训练
        model, _, training_total_loss, train_output_list = one_epoch_train(params, vocab, train_loader, model, optimizer, epoch, model_statistics)
        # 一轮模型验证
        model, sentences_pred, dev_total_loss, dev_output_list = one_epoch_dev(params, vocab, dev_loader, model, optimizer, epoch, model_statistics)

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
        # # 根据验证集损失最小来挑选最好的模型进行保存
        # if total_loss == min(total_loss_epochs):
        #     torch.save(model.state_dict(), params.checkpoint_file)
        #     logger.info('第{}轮的模型参数已经保存至{}'.format(epoch, params.checkpoint_file))

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
        epoch_dev_pred_generate_file = write_output_file("dev_g", epoch, "pred", dev_output_list["pred_generate"])
        epoch_dev_gold_file = write_output_file("dev", epoch, "gold", dev_output_list["gold"])

        bleu_train = eval("train", epoch_train_pred_file, epoch_train_gold_file, epoch_train_gold_file)
        bleu_dev = eval("dev", epoch_dev_pred_file, epoch_dev_gold_file, epoch_dev_gold_file)
        bleu_dev_g = eval("dev_g", epoch_dev_pred_generate_file, epoch_dev_gold_file, epoch_dev_gold_file)

        bleu = {**bleu_train, **bleu_dev, **bleu_dev_g}
        model_statistics["bleu"].append(bleu)

        # save bleu to tensorboard
        writer.add_scalars('bleu', bleu, epoch)
        # save loss to tensorboard
        writer.add_scalars('loss', {'train': training_total_loss, 'val': dev_total_loss}, epoch)
        writer.flush()

def one_epoch_dev(params, vocab, loader, model, optimizer, epoch, model_statistics):
    '''
    作用:
    每一轮的训练/验证

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    loader: 训练集/验证集的dataloader
    model: 当前使用模型
    optimizer: 当前使用优化器
    epoch: 当前轮数
    mode: train表示是训练阶段
          dev表示是验证阶段

    输出参数:
    model: 训练/验证结束时得到的模型
    sentences_pred: 验证结束时得到的预测序列集合(只有验证时有内容,训练时为空list)
    total_loss: 总损失
    '''

    # 断言: mode值一定在['train', 'dev']范围内
    logger.info('验证阶段,第{}轮'.format(epoch))

    with torch.no_grad():
        model.eval()

        # 对于验证阶段,我们保存所有得到的输出序列
        sentences_pred = []
        sentences_pred_generate = []

        # 记录训练/验证的总样例数
        total_examples = 0
        # 记录训练/验证的总损失
        total_loss = 0

        model_statistics["sampling_dev"][epoch] = []

        # store a list of output sentence for pred and gold
        output_list = {"gold" : [], "pred" : [], "pred_generate" : []}

        # 每一个batch的训练/验证
        for batch_index, sample in enumerate(tqdm(loader)):
            # 从数据中读取模型的输入和输出
            inputs, inputs_lens = sample.inputs, sample.inputs_lens
            outputs, outputs_lens = sample.outputs, sample.outputs_lens
            targets, targets_lens = sample.targets, sample.targets_lens
            answers = sample.answers

            inputs, outputs, targets, answers = inputs.to(params.device), outputs.to(params.device), \
                                                targets.to(params.device), answers.to(params.device)
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
            output_indices_gold = targets
            # 模型输入是在原始数据的基础上"去尾"(decoder部分的输入)
            # 原始数据: <s> 1 2 3 </s>
            # 真实输出: <s> 1 2 3
            output_indices = outputs
            input_indices = inputs
            answer_indices = answers

            # 将输入数据导入模型,得到预测的输出数据
            output_indices_pred = model(input_indices, output_indices, answer_indices=answer_indices)
            # output_indices_pred: [batch_size, output_seq_len, vocab_size]

            # 将基于vocab的概率分布,通过取最大值的方式得到预测的输出序列
            indices_pred = torch.max(output_indices_pred, dim=-1)[1]
            # indices_pred: [batch_size, output_seq_len]

            # 输出预测序列
            for indices in indices_pred:
                # full: True表示输出完整序列
                #       False表示遇到</s>就停止(只输出到</s>前的序列)
                sentence = vocab.convert_index2sentence(indices, full=False)
                sentences_pred.append(' '.join(sentence))

            # 利用预测输出和真实输出计算损失
            # softmax在模型中已经做了,因此还需要自己做一下log
            # output_indices_pred = F.log_softmax(output_indices_pred, dim=-1)
            output_indices_pred = torch.log(output_indices_pred)
            # output_indices_pred: [batch_size, output_seq_len, vocab_size]
            # output_indices_gold: [batch_size, output_seq_len]
            if params.label_smoothing:
                # 自己编写损失函数
                # 使用标签平滑归一化
                batch_size = output_indices_pred.size(0)
                output_seq_len = output_indices_pred.size(1)
                vocab_size = output_indices_pred.size(2)

                # 调整维度
                output_indices_pred = output_indices_pred.contiguous().view(batch_size * output_seq_len, vocab_size)
                output_indices_gold = output_indices_gold.contiguous().view(batch_size * output_seq_len).unsqueeze(1)
                # output_indices_pred: [batch_size * output_seq_len, vocab_size]
                # output_indices_gold: [batch_size * output_seq_len, 1]

                # 计算损失
                nll_loss = -output_indices_pred.gather(dim=-1, index=output_indices_gold)
                smooth_loss = -output_indices_pred.sum(dim=-1, keepdim=True)
                # nll_loss: [batch_size * output_seq_len]
                # smooth_loss: [batch_size * output_seq_len]

                # 通过取平均的方式得到损失
                nll_loss = nll_loss.mean()
                smooth_loss = smooth_loss.mean()

                # 使用标签平滑归一化,得到最终损失
                eps_i = params.label_smoothing_eps / vocab_size
                loss = (1 - params.label_smoothing_eps) * nll_loss + eps_i * smooth_loss
            else:
                # 使用内置的损失函数
                # NLLLoss(x,y)的两个参数:
                # x: [batch_size, num_classes, ……], 类型为LongTensor, 是预测输出
                # y: [batch_size, ……], 类型为LongTensor, 是真实输出
                criterion = torch.nn.NLLLoss(ignore_index=vocab.word2index['<pad>'])

                output_indices_pred = output_indices_pred.permute(0, 2, 1)
                # output_indices_pred: [batch_size, vocab_size, output_seq_len]
                # output_indices_gold: [batch_size, output_seq_len]
                loss = criterion(output_indices_pred, output_indices_gold)

            # 计算到当前为止的总样例数和总损失
            num_examples = input_indices.size(0)
            total_examples += num_examples
            total_loss += loss.item() * num_examples

            # generate mode
            generator = Generator(params, model)
            indices_pred, scores_pred = generator.generate_batch(input_indices, src_ans=answer_indices)

            # 输出预测序列
            for indices in indices_pred:
                # indices[0]表示beam_size中分数最高的那个输出
                sentence = vocab.convert_index2sentence(indices[0])
                sentences_pred_generate.append(' '.join(sentence))

            # 如果参数中设置了打印模型损失,则打印模型损失
            if params.print_loss:
                logger.info('Epoch : {}, batch : {}/{}, loss : {}'.format(epoch, batch_index, len(loader), loss))

            # 为了便于测试,在训练/验证阶段也可以把预测序列打印出来
            input_gold = ' '.join(vocab.convert_index2sentence(input_indices[-1]))
            answer = None
            if torch.is_tensor(answer_indices):
                answer = answer_indices[-1] * input_indices[-1]
                answer = ' '.join(vocab.convert_index2sentence(answer, full=True))
            output_gold = ' '.join(vocab.convert_index2sentence(output_indices[-1]))
            output_pred = sentences_pred[-1]
            output_pred_generate = sentences_pred_generate[-1]

            if batch_index % 50 == 0:
                logger.info('真实输入序列 : {}'.format(input_gold))
                if torch.is_tensor(answer_indices):
                    logger.info('真实答案序列 : {}'.format(answer))
                logger.info('真实输出序列 : {}'.format(output_gold))
                logger.info('预测输出序列 : {}'.format(output_pred))
                logger.info('generated : {}'.format(output_pred_generate))
                logger.info("")

            if batch_index % model_statistics["sampling_frequency"] == 0:
                sample = {
                    "epoch" : epoch,
                    "input_gold" : input_gold,
                    "answer" : answer,
                    "output_gold" : output_gold,
                    "output_pred" : output_pred
                }

                model_statistics["sampling_dev"][epoch].append(sample)

            output_list["gold"].append(output_gold)
            output_list["pred"].append(output_pred)
            output_list["pred_generate"].append(output_pred_generate)

    # 计算总损失
    total_loss = total_loss / total_examples

    logger.info("dev loss: {}".format(total_loss))

    return model, sentences_pred, total_loss, output_list

def one_epoch_train(params, vocab, loader, model, optimizer, epoch, model_statistics):
    '''
    作用:
    每一轮的训练/验证

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    loader: 训练集/验证集的dataloader
    model: 当前使用模型
    optimizer: 当前使用优化器
    epoch: 当前轮数
    mode: train表示是训练阶段
          dev表示是验证阶段

    输出参数:
    model: 训练/验证结束时得到的模型
    sentences_pred: 验证结束时得到的预测序列集合(只有验证时有内容,训练时为空list)
    total_loss: 总损失
    '''

    # 断言: mode值一定在['train', 'dev']范围内
    logger.info('训练阶段,第{}轮'.format(epoch))
    model.train()

    # 对于验证阶段,我们保存所有得到的输出序列
    sentences_pred = []
    # 记录训练/验证的总样例数
    total_examples = 0
    # 记录训练/验证的总损失
    total_loss = 0

    model_statistics["sampling_training"][epoch] = []

    # store a list of output sentence for pred and gold
    output_list = {"gold" : [], "pred" : []}

    # 每一个batch的训练/验证
    for batch_index, sample in enumerate(tqdm(loader)):
        # 从数据中读取模型的输入和输出
        inputs, inputs_lens = sample.inputs, sample.inputs_lens
        outputs, outputs_lens = sample.outputs, sample.outputs_lens
        targets, targets_lens = sample.targets, sample.targets_lens
        answers = sample.answers

        inputs, outputs, targets, answers = inputs.to(params.device), outputs.to(params.device), \
                                            targets.to(params.device), answers.to(params.device)
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
        output_indices_gold = targets
        # 模型输入是在原始数据的基础上"去尾"(decoder部分的输入)
        # 原始数据: <s> 1 2 3 </s>
        # 真实输出: <s> 1 2 3
        output_indices = outputs
        input_indices = inputs
        answer_indices = answers

        # 将输入数据导入模型,得到预测的输出数据
        output_indices_pred = model(input_indices, output_indices, answer_indices=answer_indices)
        # output_indices_pred: [batch_size, output_seq_len, vocab_size]

        # 将基于vocab的概率分布,通过取最大值的方式得到预测的输出序列
        indices_pred = torch.max(output_indices_pred, dim=-1)[1]
        # indices_pred: [batch_size, output_seq_len]

        # 输出预测序列
        for indices in indices_pred:
            # full: True表示输出完整序列
            #       False表示遇到</s>就停止(只输出到</s>前的序列)
            sentence = vocab.convert_index2sentence(indices, full=False)
            sentences_pred.append(' '.join(sentence))

        # 利用预测输出和真实输出计算损失
        # softmax在模型中已经做了,因此还需要自己做一下log
        # output_indices_pred = F.log_softmax(output_indices_pred, dim=-1)
        output_indices_pred = torch.log(output_indices_pred)
        # output_indices_pred: [batch_size, output_seq_len, vocab_size]
        # output_indices_gold: [batch_size, output_seq_len]
        if params.label_smoothing:
            # 自己编写损失函数
            # 使用标签平滑归一化
            batch_size = output_indices_pred.size(0)
            output_seq_len = output_indices_pred.size(1)
            vocab_size = output_indices_pred.size(2)

            # 调整维度
            output_indices_pred = output_indices_pred.contiguous().view(batch_size * output_seq_len, vocab_size)
            output_indices_gold = output_indices_gold.contiguous().view(batch_size * output_seq_len).unsqueeze(1)
            # output_indices_pred: [batch_size * output_seq_len, vocab_size]
            # output_indices_gold: [batch_size * output_seq_len, 1]

            # 计算损失
            nll_loss = -output_indices_pred.gather(dim=-1, index=output_indices_gold)
            smooth_loss = -output_indices_pred.sum(dim=-1, keepdim=True)
            # nll_loss: [batch_size * output_seq_len]
            # smooth_loss: [batch_size * output_seq_len]

            # 通过取平均的方式得到损失
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()

            # 使用标签平滑归一化,得到最终损失
            eps_i = params.label_smoothing_eps / vocab_size
            loss = (1 - params.label_smoothing_eps) * nll_loss + eps_i * smooth_loss
        else:
            # 使用内置的损失函数
            # NLLLoss(x,y)的两个参数:
            # x: [batch_size, num_classes, ……], 类型为LongTensor, 是预测输出
            # y: [batch_size, ……], 类型为LongTensor, 是真实输出
            criterion = torch.nn.NLLLoss(ignore_index=vocab.word2index['<pad>'])

            output_indices_pred = output_indices_pred.permute(0, 2, 1)
            # output_indices_pred: [batch_size, vocab_size, output_seq_len]
            # output_indices_gold: [batch_size, output_seq_len]
            loss = criterion(output_indices_pred, output_indices_gold)

        # 计算到当前为止的总样例数和总损失
        num_examples = input_indices.size(0)
        total_examples += num_examples
        total_loss += loss.item() * num_examples

        # 如果参数中设置了打印模型损失,则打印模型损失
        if params.print_loss:
            logger.info('Epoch : {}, batch : {}/{}, loss : {}'.format(epoch, batch_index, len(loader), loss))

        # 如果是训练阶段,就利用优化器进行BP反向传播,更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 为了便于测试,在训练/验证阶段也可以把预测序列打印出来
        input_gold = ' '.join(vocab.convert_index2sentence(input_indices[-1]))
        answer = None
        if torch.is_tensor(answer_indices):
            answer = answer_indices[-1] * input_indices[-1]
            answer = ' '.join(vocab.convert_index2sentence(answer, full=True))
        output_gold = ' '.join(vocab.convert_index2sentence(output_indices[-1]))
        output_pred = sentences_pred[-1]

        if batch_index % 500 == 0:
            logger.info('真实输入序列 : {}'.format(input_gold))
            if torch.is_tensor(answer_indices):
                logger.info('真实答案序列 : {}'.format(answer))
            logger.info('真实输出序列 : {}'.format(output_gold))
            logger.info('预测输出序列 : {}'.format(output_pred))
            logger.info("")

        if batch_index % model_statistics["sampling_frequency"] == 0:
            sample = {
                "epoch" : epoch,
                "input_gold" : input_gold,
                "answer" : answer,
                "output_gold" : output_gold,
                "output_pred" : output_pred
            }

            model_statistics["sampling_training"][epoch].append(sample)

        output_list["gold"].append(output_gold)
        output_list["pred"].append(output_pred)

    # 计算总损失
    total_loss = total_loss / total_examples

    logger.info("train loss: {}".format(total_loss))

    return model, sentences_pred, total_loss, output_list

# a wrapper for bleu eval to make result as a dict
def eval(mode, out_file, src_file, tgt_file, isDIn=False, num_pairs=500):
    print("{}:".format(mode))
    train_result = bleu_eval(out_file, src_file, tgt_file, isDIn, num_pairs)

    ret = {"{}_Bleu_1".format(mode) : train_result[0] * 100,
           "{}_Bleu_2".format(mode) : train_result[1] * 100,
           "{}_Bleu_3".format(mode) : train_result[2] * 100,
           "{}_Bleu_4".format(mode) : train_result[3] * 100,
           }

    return ret

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

    if params.rnnsearch:
        from rnnsearch import Model
    else:
        from transformer import Model

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    # 根据加载数据构造batch(使用pytorch中的datasets类)
    train_loader, dev_loader = prepare_dataloaders(params, data)

    # 训练模型
    train_model(params, vocab, train_loader, dev_loader, model_statistics, writer)

    writer.close()
