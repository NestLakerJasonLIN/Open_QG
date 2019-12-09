#!/usr/bin/env python
# encoding: utf-8
'''
测试demo:
(1)人为输入句子和答案
(2)将句子和答案输入模型进行预测,得到预测问题
(3)输出预测问题
'''
__author__ = 'qjzhzw'

import os
import torch

def init():
    '''
    作用:
    模型和参数初始化

    输出参数:
    logger: 日志输出器
    params: 参数集合
    vocab: Vocab类
    model: 当前使用模型
    generator: 当前使用生成器
    '''
    
    from logger import logger
    from params import params
    from vocab import Vocab
    from beam import Generator

    # 加载日志输出器和参数集合
    logger = logger()
    params = params()

    # 从已保存的pt文件中读取数据
    # 包括:vocab,训练集/验证集各自的输入/输出索引序列
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']
    vocab_pos = data['vocab_pos']
    vocab_ner = data['vocab_ner']
    params = data['params']
    params.lexical_feature = True

    if params.rnnsearch:
        from rnnsearch import Model
    else:
        from transformer import Model

    # 打印参数列表
    if params.print_params:
        logger.info('参数列表:{}'.format(params))

    # 定义模型
    model = Model(params, vocab, vocab_pos=vocab_pos, vocab_ner=vocab_ner).to(params.device)

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

    model.eval()

    # 定义生成器
    generator = Generator(params, model)

    return logger, params, vocab, vocab_pos, vocab_ner, model, generator


def demo(input_sentence, input_answer, input_ner, input_answer_ner, input_pos, input_answer_pos, logger, params, vocab, vocab_ner, vocab_pos, model, generator):

    '''
    作用:
    模型和参数初始化

    输入参数:
    input_sentence: 输入句子
    input_answer: 输入答案
    输出参数:
    logger: 日志输出器
    params: 参数集合
    vocab: Vocab类
    model: 当前使用模型
    generator: 当前使用生成器

    输出参数:
    output_question: 输出问题
    '''

    # 将输入句子和答案构造成统一文本
    input_sentence = '<cls> ' + input_sentence + ' <sep> ' + input_answer + ' <sep>'
    logger.info('输入句子的文本形式为 : {}'.format(input_sentence))

    input_ner = '<cls> ' + input_ner + ' <sep> ' + input_answer_ner + ' <sep>'
    input_pos = '<cls> ' + input_pos + ' <sep> ' + input_answer_pos + ' <sep>'

    # 将文本转化为list,并添加起止符<s>和</s>
    input_sentence = input_sentence.split()
    input_sentence = ['<s>'] + input_sentence + ['</s>']

    input_ner = input_ner.split()
    input_ner = ['<s>'] + input_ner + ['</s>']
    input_pos = input_pos.split()
    input_pos = ['<s>'] + input_pos + ['</s>']

    # 将输入句子转化为索引形式
    input_indices = vocab.convert_sentence2index(input_sentence)
    logger.info('输入句子的索引形式为 : {}'.format(input_indices))

    # 输入模型
    input_indices = torch.tensor(input_indices).to(params.device)
    # input_indices: [input_seq_len]
    input_indices = input_indices.unsqueeze(0)
    # input_indices: [1(batch_size), input_seq_len]

    # convert NER indices
    ner_input_indices = vocab_ner.convert_sentence2index(input_ner)
    ner_input_indices = torch.tensor(ner_input_indices).to(params.device)
    ner_input_indices = ner_input_indices.unsqueeze(0)

    # convert POS indices
    pos_input_indices = vocab_pos.convert_sentence2index(input_pos)
    pos_input_indices = torch.tensor(pos_input_indices).to(params.device)
    pos_input_indices = pos_input_indices.unsqueeze(0)

    # 使用beam_search算法,以<s>作为开始符得到完整的预测序列
    indices_pred, scores_pred = generator.generate_batch(input_indices, src_ans=None,
                                                        pos_input_indices=pos_input_indices,
                                                        ner_input_indices=ner_input_indices)

    # indices_pred: [batch_size, beam_size, output_seq_len]
    output_indices = indices_pred[0][0]
    # output_indices: [output_seq_len]
    logger.info('输出句子的索引形式为 : {}'.format(output_indices))

    # 将输出句子转化为文本形式
    output_question = vocab.convert_index2sentence(output_indices)
    output_question = ' '.join(output_question)
    logger.info('输出问题的文本形式为 : {}'.format(output_question))

    return output_question


if __name__ == '__main__':

    # 模型和参数初始化
    logger, params, vocab, vocab_ner, vocab_pos, model, generator = init()

    # 测试demo: 输入句子和答案,输出问题
    output_question = demo(input_sentence = 'There are 5000000 people in the united states .',
                    input_answer = '5000000',
                    input_ner = 'O O O O O O O O O',
                    input_answer_ner = 'O',
                    input_pos = 'O O O O O O O O O',
                    input_answer_pos = 'O',
                    logger = logger,
                    params = params,
                    vocab = vocab,
                    vocab_ner = vocab_ner,
                    vocab_pos = vocab_pos,
                    model = model,
                    generator = generator)
