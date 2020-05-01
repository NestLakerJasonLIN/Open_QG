#!/usr/bin/env python
# encoding: utf-8
'''
数据测试:
(1)将模型测试集的输入输出数据构造为batch
(2)将测试集数据输入模型,进行模型测试
(3)保存模型测试的预测结果
'''
__author__ = 'qjzhzw'

import json
import os
import shutil
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from logger import logger
from params import params
from vocab import Vocab
from dataset import Dataset
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
    将模型测试集的输入输出数据构造为batch

    输入参数:
    params: 参数集合
    data: 输入的数据

    输出参数:
    test_loader: 测试集的dataloader
    '''

    logger.info('正在从{}中读取数据'.format(params.dataset_dir))

    # 构造test_loader
    if params.test_on_train:
        test_dataset = Dataset(params, data, mode='test_on_train')
    else:
        test_dataset = Dataset(params, data, mode='test')
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        num_workers = params.num_workers,
        batch_size = params.batch_size,
        collate_fn = collate_fn,
        shuffle = False
    )
    logger.info('正在构造test_loader,共有{}个batch'.format(len(test_loader)))

    return test_loader


def test_model(params, vocab, test_loader):
    '''
    作用:
    测试模型

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    test_loader: 测试集的dataloader
    '''

    logger.info('正在加载模型,即将开始测试')

    # 定义模型
    model = Model(params, vocab).to(params.device)

    # 如果参数中设置了打印模型结构,则打印模型结构
    if params.print_model:
        logger.info(model)

    # 加载模型参数
    if os.path.exists(params.checkpoint_file):
        model_params = torch.load(params.checkpoint_file, map_location=params.device)
        model.load_state_dict(model_params)
        logger.info('正在从{}中读取已经训练好的模型参数'.format(params.checkpoint_file))
    else:
        logger.info('注意!!!没有训练好的模型参数,正在使用随机初始化模型参数进行测试')

    # 一轮模型测试
    sentences_pred = one_epoch(params, vocab, test_loader, model)

    # 将预测结果存入本地文件
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    # 依次写入文件
    f_pred = open(params.pred_file, 'w')
    for sentence_pred in sentences_pred:
        f_pred.write(sentence_pred + '\n')
    f_pred.close()
    logger.info('测试阶段的预测结果已经保存至{}'.format(params.pred_file))

    # 原始的真实输出文件,需要从数据目录移动
    if params.test_on_train:
        shutil.copyfile(params.train_question_file, params.gold_file)
    else:
        shutil.copyfile(params.test_question_file, params.gold_file)

    # 使用multi-bleu.perl的脚本对结果进行评估
    os.system('evaluate/multi-bleu.perl %s < %s' %(params.gold_file, params.pred_file))


def one_epoch(params, vocab, loader, model):
    '''
    作用:
    每一轮的测试

    输入参数:
    params: 参数集合
    vocab: 从pt文件中读取的该任务所使用的vocab
    loader: 测试集的dataloader
    model: 当前使用模型

    输出参数:
    sentences_pred: 测试结束时得到的预测序列集合
    '''

    logger.info('测试阶段')

    with torch.no_grad():
        model.eval()

        # 我们保存所有得到的输出序列
        sentences_pred = []

        # 每一个batch的测试
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

            output_indices = outputs
            input_indices = inputs
            answer_indices = answers

            # 使用beam_search算法,以<s>作为开始符得到完整的预测序列
            generator = Generator(params, model)
            indices_pred, scores_pred = generator.generate_batch(input_indices, src_ans=answer_indices)
            # indices_pred: [batch_size, beam_size, output_seq_len]

            # 输出预测序列
            for indices in indices_pred:
                # indices[0]表示beam_size中分数最高的那个输出
                sentence = vocab.convert_index2sentence(indices[0])
                sentences_pred.append(' '.join(sentence))

            # 为了便于测试,在测试阶段也可以把预测序列打印出来
            if batch_index % 50 == 0:
                input_gold = ' '.join(vocab.convert_index2sentence(input_indices[-1]))
                logger.info('真实输入序列 : {}'.format(input_gold))

                if torch.is_tensor(answer_indices):
                    answer = answer_indices[-1] * input_indices[-1]
                    answer = ' '.join(vocab.convert_index2sentence(answer, full=True))
                    logger.info('真实答案序列 : {}'.format(answer))

                output_gold = ' '.join(vocab.convert_index2sentence(output_indices[-1]))
                logger.info('真实输出序列 : {}'.format(output_gold))

                output_pred = sentences_pred[-1]
                logger.info('预测输出序列 : {}'.format(output_pred))

    return sentences_pred


if __name__ == '__main__':

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()

    test_on_train = params.test_on_train
    pred_file = params.pred_file
    gold_file = params.gold_file

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']
    params = data['params']

    if (test_on_train):
        params.test_on_train = test_on_train
        assert "train" in pred_file
        assert "train" in gold_file
        params.pred_file = pred_file
        params.gold_file = gold_file

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
    test_loader = prepare_dataloaders(params, data)

    # 测试模型
    test_model(params, vocab, test_loader)
