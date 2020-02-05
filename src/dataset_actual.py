#!/usr/bin/env python
# encoding: utf-8
'''
Dataset类:
(1)使用pytorch的Dataset类,将模型的输入输出数据构造为batch
'''
__author__ = 'qjzhzw'

import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, params, data, mode='train'):
        '''
        Dataset类:
        使用pytorch的Dataset类,将模型的输入输出数据构造为batch

        输入参数:
        params: 参数集合
        data: 加载了的pt文件的内容
        mode: train表示是训练集的Dataset类构造
              dev表示是验证集的Dataset类构造
        '''

        # 从data中读取所有信息
        self.params = params
        self.mode = mode

        self.vocab = data['vocab']
        self.train_input_indices = data['train_input_indices']
        self.train_output_indices = data['train_output_indices']
        self.train_answers = data['train_answers']
        self.dev_input_indices = data['dev_input_indices']
        self.dev_output_indices = data['dev_output_indices']
        self.dev_answers = data['dev_answers']
        self.test_input_indices = data['test_input_indices']
        self.test_output_indices = data['test_output_indices']
        self.test_answers = data['test_answers']

        # 断言: mode值一定在['train', 'dev', 'test']范围内
        assert self.mode in ['train', 'dev', 'test']

        # 断言: 训练集/验证集/测试集的输入输出数量必须一致
        assert len(self.train_input_indices) == len(self.train_output_indices)
        assert len(self.dev_input_indices) == len(self.dev_output_indices)
        assert len(self.test_input_indices) == len(self.test_output_indices)
        if self.params.answer_embeddings:
            assert len(self.train_input_indices) == len(self.train_answers)
            assert len(self.dev_input_indices) == len(self.dev_answers)
            assert len(self.test_input_indices) == len(self.test_answers)

    def __getitem__(self, index):
        '''
        作用:
        这个方法必须实现,返回每个batch的内容

        输入参数:
        index: 每条数据的索引

        输出参数:
        input_indices: 输入序列
        output_indices: 输出序列
        answers: 答案
        vocab: Vocab类
        '''

        # 根据mode值的不同返回不同的内容
        # 如果答案为空,则该部分返回空
        if self.mode == 'train':
            return self.train_input_indices[index], \
                   self.train_output_indices[index], \
                   self.train_answers[index] if self.train_answers else None, \
                   self.vocab
        elif self.mode == 'dev':
            return self.dev_input_indices[index], \
                   self.dev_output_indices[index], \
                   self.dev_answers[index] if self.dev_answers else None, \
                   self.vocab
        elif self.mode == 'test':
            return self.test_input_indices[index], \
                   self.test_output_indices[index], \
                   self.test_answers[index] if self.test_answers else None, \
                   self.vocab

    def __len__(self):
        '''
        作用:
        这个方法必须实现,否则会报错:NotImplementedError

        输出参数:
        Dataset类的大小
        '''

        # 根据mode值的不同返回不同的内容
        if self.mode == 'train':
            return len(self.train_input_indices)
        elif self.mode == 'dev':
            return len(self.dev_input_indices)
        elif self.mode == 'test':
            return len(self.test_input_indices)


def collate_fn(data):
    '''
    作用:
    构造batch

    输入参数:
    data: 输入的数据(即上面的__getitem__传下来的),原始的一个batch的内容
          是一个二维list,第一维是batch_size,第二维是batch中每个句子(不等长)

    输出参数:
    batch_input: 输入序列的batch
    batch_output: 输出序列的batch
    batch_answer: 答案的batch
    '''

    # 对模型的[输入序列/输出序列/答案]分别构造batch
    batch_input = get_batch(data, mode=0)
    batch_output = get_batch(data, mode=1)

    # 如果答案部分不存在,则返回的batch_answer为None
    batch_answer = get_batch(data, mode=2, batch_input=batch_input)

    return batch_input, batch_output, batch_answer


def get_batch(data, mode=0, batch_input=None):
    '''
    作用:
    构造batch

    输入参数:
    data: 输入的数据(即上面的__getitem__传下来的),原始的一个batch的内容
          是一个二维list,第一维是batch_size,第二维是batch中每个句子(不等长)
    mode: 0表示是输入序列
          1表示是输出序列
          2表示是答案
    batch_input: 在构造答案时需要构造一个大小和输入序列完全一样的tensor,然后根据答案位置填0和1

    输出参数:
    batch: 构造好的batch
    '''

    # 断言: mode值一定在[0, 1, 2]范围内
    assert mode in [0, 1, 2]

    # 输入序列/输出序列的处理方式
    if mode == 0 or mode == 1:
        # 获得该batch中的最大句长
        max_len = 0
        # 这里的single_data是上面的__getitem__传下来的,原始的一个batch的内容
        # 获得single_data以后,要通过索引的方式获取其中每个需要的部分
        # 例如single_data[0]是输入序列,single_data[1]是输出序列
        for single_data in data:
            sentence = single_data[mode]
            if len(sentence) > max_len:
                max_len = len(sentence)

        # 构造batch,不足的部分用<pad>对应的索引填充
        batch = []
        for single_data in data:
            sentence = single_data[mode]
            vocab = single_data[-1]
            batch_sentence = sentence + [vocab.word2index['<pad>']] * (max_len - len(sentence))
            batch.append(batch_sentence)

        # 将二维list的batch直接转换为tensor的形式
        batch = torch.tensor(batch)

    elif mode == 2:
        # 如果答案部分不为空,对答案部分构造batch,否则返回None
        if data[0][mode] != None:
            # 在构造答案时需要构造一个大小和输入序列完全一样的tensor
            batch = torch.zeros_like(batch_input)

            # 构造batch,然后根据答案位置填0和1
            for index, single_data in enumerate(data):
                answer = single_data[mode]
                # 因为输入序列有<s>作为起始符,因此真实答案位置需要后移一位
                answer_start = answer[0] + 1
                answer_end = answer[1] + 1
                batch[index][answer_start: answer_end] = 1
        else:
            return None

    return batch
