import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerDecoderLayer
import numpy as np
from transformer import Utils
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, params, vocab):
        super(Model, self).__init__()
        self.model_type = 'Transformer'

        # model params
        # TODO: check all params are same for my model and torch model
        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # Transformer encoder
        self.encoder = TorchEncoder(self.params, self.vocab)

        # Transformer decoder
        self.decoder = TorchDecoder(self.params, self.vocab)

    def forward(self, input_indices, output_indices, answer_indices=None):
        encoder_hiddens = self.encoder(input_indices, answer_indices)
        output_indices = self.decoder(output_indices, input_indices, encoder_hiddens)

        return output_indices


# word_emb -> pos_emb -> +answer_emb -> [TorchEncoderLayer1, TorchEncoderLayer2, ...]
class TorchEncoder(nn.Module):
    def __init__(self, params, vocab):
        """
            word embedding + positional encoding + Transformer encoderlayer
        """
        super().__init__()

        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # 构造掩膜和位置信息
        self.utils = Utils(self.params)

        # word embedding, position encoding, answer encoding
        self.word_embedding_encoder = nn.Embedding(self.vocab_size, self.params.d_model)
        self.position_embedding_encoder = nn.Embedding(self.vocab_size, self.params.d_model)
        self.answer_embedding_encoder = nn.Embedding(2, self.params.d_model)

        # Encoder layers
        self.encoder_layers = \
            nn.ModuleList([TorchEncoderLayer(self.params) for _ in range(self.params.num_layers)])

    def forward(self, input_indices, answer_indices=None):
        bs, src_len = input_indices.size()

        input_indices_positions = self.utils.build_positions(input_indices)
        encoder_self_attention_masks = self.utils.build_pad_masks(query=input_indices, key=input_indices)
        # input_indices_positions: [batch_size, input_seq_len]
        # encoder_self_attention_masks: [batch_size, input_seq_len, input_seq_len]

        # add positional encodinng
        # TODO: update positional encoding
        input_indices = self.word_embedding_encoder(input_indices) * np.sqrt(self.params.d_model) + \
                        self.position_embedding_encoder(input_indices)
        # input_indices: [batch_size, input_seq_len, d_model]

        # add answer encoding
        if torch.is_tensor(answer_indices):
            input_indices += self.answer_embedding_encoder(answer_indices)

        # transformer encoding layers
        for encoder_layer in self.encoder_layers:
            input_indices = encoder_layer(input_indices,
                                          encoder_self_attention_masks)
        # input_indices: [batch_size, input_seq_len, d_model]
        assert list(input_indices.size()) == [bs, src_len, self.params.d_model]

        return input_indices


# a wrapper for torch.nn.TransformerEncoderLayer: self-attention -> FFN
class TorchEncoderLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=params.d_model,
            nhead=params.num_heads,
            dim_feedforward=params.d_ff,
            dropout=params.dropout,
            activation="relu",  # hardcode ReLU same as my model
        )

    def forward(self, input_indices, encoder_self_attention_masks):
        """
            - input_indices: [N, S, E]
            - encoder_self_attention_masks: [N, S, S] => key_padding_mask
            - src: (S, N, E)
            - src_mask: (S, S)
            - src_key_padding_mask: (N, S)
            S: source seq len, N: batch size, E: feature number
            Note:
                - [src/tgt/memory]_mask should be filled with
                    float('-inf') for the masked positions and float(0.0) else. These masks
                    ensure that predictions for position i depend only on the unmasked positions
                    j and are applied identically for each sequence in a batch.

                - [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
                    that should be masked with float('-inf') and False values will be unchanged.
                    This mask ensures that no information will be taken from position i if
                    it is masked, and has a separate mask for each sequence in a batch.

        """
        src = input_indices.permute(1, 0, 2)

        src_key_padding_mask = encoder_self_attention_masks[:, 0, :]
        src_key_padding_mask = \
            (torch.ones_like(src_key_padding_mask) - \
             src_key_padding_mask).type(torch.bool)  # convert to Byte Tensor and flip 1/0

        out = self.transformer_encoder_layer.forward(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask
        )

        # return shape = [batch_size, seq_len, d_model]
        return out.permute(1, 0, 2)


# word_emb -> pos_emb -> +answer_emb -> [TorchEncoderLayer1, TorchEncoderLayer2, ...]
class TorchDecoder(nn.Module):
    def __init__(self, params, vocab):
        super().__init__()
        self.params = params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # 构造掩膜和位置信息
        self.utils = Utils(self.params)

        # word embedding
        self.word_embedding_decoder = nn.Embedding(self.vocab_size, self.params.d_model)

        # positional embedding
        self.position_embedding_decoder = nn.Embedding(self.vocab_size, self.params.d_model)

        # decoder layers
        self.decoder_layers = nn.ModuleList([TorchDecoderLayer(self.params) for _ in range(self.params.num_layers)])

        # linear layer to produce vocab probability
        self.output = nn.Linear(self.params.d_model, self.vocab_size)

    def forward(self, output_indices, input_indices, encoder_hiddens):
        bs, tgt_len = output_indices.size()

        # 构造掩膜和位置信息
        output_indices_positions = self.utils.build_positions(output_indices)
        #         decoder_mutual_attention_masks = self.utils.build_pad_masks(query=output_indices, key=input_indices)
        #         decoder_self_attention_masks = (self.utils.build_pad_masks(query=output_indices, key=output_indices) * \
        #                                         self.utils.build_triu_masks(output_indices)).gt(0)

        tgt_mask = self.utils.build_triu_masks(output_indices)[0]
        tgt_mask[tgt_mask == 0] = float('-inf')
        tgt_mask[tgt_mask == 1] = 0.0

        memory_mask = None

        tgt_key_padding_mask = \
            self.utils.build_pad_masks(query=output_indices, key=output_indices)[:, 0, :]
        tgt_key_padding_mask = \
            (torch.ones_like(tgt_key_padding_mask) - \
             tgt_key_padding_mask).type(torch.bool)  # convert to Byte Tensor and flip 1/0

        memory_key_padding_mask = \
            self.utils.build_pad_masks(query=output_indices, key=input_indices)[:, 0, :]
        memory_key_padding_mask = \
            (torch.ones_like(memory_key_padding_mask) - \
             memory_key_padding_mask).type(torch.bool)  # convert to Byte Tensor and flip 1/0

        # output_indices_positions: [batch_size, output_seq_len]
        # decoder_mutual_attention_masks: [batch_size, output_seq_len, input_seq_len]
        # decoder_self_attention_masks: [batch_size, output_seq_len, output_seq_len]

        # 将索引/位置信息转换为词向量
        output_indices = self.word_embedding_decoder(output_indices) * np.sqrt(self.params.d_model) + \
                         self.position_embedding_decoder(output_indices)
        # output_indices: [batch_size, output_seq_len, d_model]

        # 经过多个相同子结构组成的decoder子层,层数为num_layers
        for decoder_layer in self.decoder_layers:
            output_indices = \
                decoder_layer(output_indices,
                              encoder_hiddens,
                              None,
                              None,
                              self.vocab,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        # output_indices: [batch_size, output_seq_len, d_model]
        # attention: [batch_size, output_seq_len, input_seq_len]
        # context_vector: [batch_size, output_seq_len, d_model]

        assert list(output_indices.size()) == [bs, tgt_len, self.params.d_model]

        # 经过输出层,将隐向量转换为模型最终输出:基于vocab的概率分布
        output_indices = self.output(output_indices)
        # output_indices: [batch_size, output_seq_len, vocab_size]

        # 使用softmax将模型输出转换为概率分布
        output_indices = F.softmax(output_indices, dim=-1)
        # # 将softmax后的结果控制在一定范围内,避免在计算log时出现log(0)的情况
        # output_indices = torch.clamp(output_indices, 1e-30, 1)

        return output_indices


# a wrapper for torch.nn.TransformerEncoderLayer:
# self-attention -> encoder-decoder attention -> FFN
class TorchDecoderLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # 构造掩膜和位置信息
        self.utils = Utils(self.params)

        self.transformer_decoder_layer = TransformerDecoderLayer(
            d_model=params.d_model,
            nhead=params.num_heads,
            dim_feedforward=params.d_ff,
            dropout=params.dropout,
            activation="relu"
        )

    def forward(self, output_indices, encoder_hiddens,
                decoder_self_attention_masks, decoder_mutual_attention_masks, vocab,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
            encoder_hiddens [S, N, E]
        """
        # generate masks manually since the original my model
        # will merge padding wand subsequent masks together

        assert tgt_mask is not None
        assert tgt_key_padding_mask is not None
        assert memory_key_padding_mask is not None

        # convert parameters to fit torch model
        tgt = output_indices.permute(1, 0, 2)
        memory = encoder_hiddens.permute(1, 0, 2)

        out = self.transformer_decoder_layer.forward(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # return shape = [batch_size, seq_len, d_model]
        return out.permute(1, 0, 2)