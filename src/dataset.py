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
from torch.nn.utils.rnn import pad_sequence


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
        self.train_answer_indices = data['train_answer_indices']
        self.dev_input_indices = data['dev_input_indices']
        self.dev_output_indices = data['dev_output_indices']
        self.dev_answer_indices = data['dev_answer_indices']
        self.test_input_indices = data['test_input_indices']
        self.test_output_indices = data['test_output_indices']
        self.test_answer_indices = data['test_answer_indices']

        # 断言: mode值一定在['train', 'dev', 'test']范围内
        assert self.mode in ['train', 'dev', 'test', 'test_on_train']

        # 断言: 训练集/验证集/测试集的输入输出数量必须一致
        assert len(self.train_input_indices) == len(self.train_output_indices)
        assert len(self.dev_input_indices) == len(self.dev_output_indices)
        assert len(self.test_input_indices) == len(self.test_output_indices)
        assert len(self.train_input_indices) == len(self.train_answer_indices)
        assert len(self.dev_input_indices) == len(self.dev_answer_indices)
        assert len(self.test_input_indices) == len(self.test_answer_indices)

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
                   self.train_answer_indices[index] if self.train_answer_indices else None, \
                   self.vocab
        elif self.mode == 'dev':
            return self.dev_input_indices[index], \
                   self.dev_output_indices[index], \
                   self.dev_answer_indices[index] if self.dev_answer_indices else None, \
                   self.vocab
        elif self.mode == 'test':
            return self.test_input_indices[index], \
                   self.test_output_indices[index], \
                   self.test_answer_indices[index] if self.test_answer_indices else None, \
                   self.vocab
        elif self.mode == 'test_on_train':
            return self.train_input_indices[index], \
                   self.train_output_indices[index], \
                   self.train_answer_indices[index] if self.train_answer_indices else None, \
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
        elif self.mode == 'test_on_train':
            return len(self.test_input_indices)

class Sample:
    def __init__(self, batch):
        input_indices, question_indices, answer_indices, vocab = zip(*batch)
        vocab = vocab[0]

        compose_input_indices = [[vocab.word2index['<cls>']] + input_indices[i] + [vocab.word2index['<sep>']] + \
                                 answer_indices[i] + [vocab.word2index['<sep>']] for i in range(len(input_indices))]

        # input
        self.inputs_lens = [len(x) for x in compose_input_indices]
        self.inputs = pad_sequence([torch.tensor(x) for x in compose_input_indices], batch_first=True)
        self.input_segment_indices = torch.tensor([[0] * (1 + len(input_indices[i]) + 1) +
                                                   [1] * (len(answer_indices[i]) + 1) +
                                                   [2] * (self.inputs[i].size(0) - self.inputs_lens[i])
                                                   for i in range(len(input_indices))])

        # output/target
        question_indices = [[vocab.word2index['<s>']] + question_idx + [vocab.word2index['</s>']]
                            for question_idx in question_indices]
        outputs_text = [torch.tensor(text[:-1]) for text in question_indices]
        targets_text = [torch.tensor(text[1:]) for text in question_indices]

        self.outputs_lens = [y.size(0) for y in outputs_text]
        self.targets_lens = [y.size(0) for y in targets_text]

        self.outputs = pad_sequence(outputs_text, batch_first=True)
        self.targets = pad_sequence(targets_text, batch_first=True)

    def pin_memory(self):
        self.inputs = self.inputs.pin_memory()
        self.outputs = self.outputs.pin_memory()
        self.targets = self.targets.pin_memory()
        return self

def collate_fn(batch):
    return Sample(batch)