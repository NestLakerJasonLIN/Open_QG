#!/usr/bin/env python
# encoding: utf-8
'''
加载数据集:
(1)将json形式的SQuAD原始文件(由Song et al., 2018提供)转换为txt形式,便于transformer模型进行处理
'''
__author__ = 'yanwenl'

import json
import os

from logger import logger
from params import params


def load_dataset(params, origin_file, sentence_file, question_file, answer_file):
    # 将原始数据加载进来
    instances = json.loads(open(origin_file, 'r').read())

    # total和num用于判断有多少数据被成功处理
    total = len(instances)
    clipped_num = 0
    dropped_num = 0
    num = 0

    # 所有输出文件
    f_sentence = open(sentence_file, 'w')
    f_question = open(question_file, 'w')
    f_answer = open(answer_file, 'w')

    # 依次处理所有数据
    for instance in instances:
        # 加载原始文件中的[句子/问题/答案]三元组
        sentence = instance['annotation1']['toks'].strip().lower()
        question = instance['annotation2']['toks'].strip().lower()
        answer = instance['annotation3']['toks'].strip().lower()

        # 将str切分为list
        sentence = sentence.split()
        question = question.split()
        answer = answer.split()

        # clip sentences with too long length
        if (params.max_seq_len < len(sentence)):
            clipped_num += 1
        sentence = sentence[:params.max_seq_len]

        # drop sentence that does not contain answer
        answer_start = 0
        answer_end = 0
        for idx in range(len(sentence)):
            if answer == sentence[idx: idx + len(answer)]:
                answer_start = idx
                answer_end = idx + len(answer)
                break

        if (answer_start != 0 or answer_end != 0):
            f_sentence.write(' '.join(sentence) + '\n')
            f_question.write(' '.join(question) + '\n')
            f_answer.write(' '.join(answer) + '\n')
        else:
            dropped_num += 1

    # 关闭所有文件
    f_sentence.close()
    f_question.close()
    f_answer.close()

    logger.info('{} load: {}, clipped: {} dropped: {}'.format(origin_file, total, clipped_num, dropped_num))

if __name__ == '__main__':

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()

    # 判断子目录train/dev/test是否存在，若不存在则创建
    if not os.path.exists(params.train_dir):
        os.makedirs(params.train_dir)
    if not os.path.exists(params.dev_dir):
        os.makedirs(params.dev_dir)
    if not os.path.exists(params.test_dir):
        os.makedirs(params.test_dir)

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    load_dataset(params,
                params.origin_train_file,
                params.train_sentence_file,
                params.train_question_file,
                params.train_answer_file)
    load_dataset(params,
                params.origin_dev_file,
                params.dev_sentence_file,
                params.dev_question_file,
                params.dev_answer_file)
    load_dataset(params,
                params.origin_test_file,
                params.test_sentence_file,
                params.test_question_file,
                params.test_answer_file)