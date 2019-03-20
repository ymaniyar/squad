"""Top-level model classes.
Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class QANet(nn.Module):
    """
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1):
        super(QANet, self).__init__()

        # print("vectors: ", word_vectors)
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors = char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        self.enc_context = layers.Encoder(1, drop_prob/6, hidden_size, num_conv=4, num_heads=8)
        self.enc_query = layers.Encoder(1, drop_prob/6, hidden_size,  num_conv=4, num_heads=8)

        self.transformer = layers.Transformer(4*hidden_size, drop_prob)

        self.att = layers.BiDAFAttention(hidden_size= hidden_size,
                                         drop_prob=drop_prob)

        self.out = layers.TransformerOutput(hidden_size)

        self.hidden_size = hidden_size

    def make_sub_mask(self, seq):
        sz_b, len_s = seq.size()[0], seq.size()[1]
        subsequent_mask = torch.tril(torch.ones((len_s, len_s), device = seq.device, dtype = torch.uint8), diagonal=0)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) 
        # print("subsequent_mask: ", subsequent_mask)
        return subsequent_mask

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)



        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)
        
        c_max_len = cw_idxs.shape[1]
        q_max_len = qw_idxs.shape[1]

        c_emb = c_emb.view(-1, c_max_len, self.hidden_size)
        q_emb = q_emb.view(-1, q_max_len, self.hidden_size)

        batch_size_c = c_emb.size()[0]
        batch_size_q = q_emb.size()[0]

        c_mask = c_mask.unsqueeze(2)
        q_mask = q_mask.unsqueeze(2)

        c_mask_inv = 1 - c_mask
        q_mask_inv = 1 - q_mask

        c_enc = self.enc_context(c_emb, batch_size_c, c_mask_inv, c_max_len)
        q_enc = self.enc_query(q_emb, batch_size_q, q_mask_inv, q_max_len)

        cq_att = self.att(c_enc, q_enc, c_mask, q_mask)

        m_0, m_1, m_2 = self.transformer(cq_att, c_mask_inv, batch_size_c, c_max_len)

        out = self.out(m_0, m_1, m_2, c_mask.squeeze(dim=2))

        return out


class Transformer(nn.Module):
    """
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(Transformer, self).__init__()

        # print("vectors: ", word_vectors)
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors = char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.embed_size = 64 + 300 + 4


        self.transformer = layers.Transformer(hidden_size)

        self.att = layers.BiDAFAttention(hidden_size= hidden_size,
                                         drop_prob=drop_prob)

        self.out = layers.TransformerOutput(hidden_size)

        self.hidden_size = hidden_size

    def make_sub_mask(self, seq):
        sz_b, len_s = seq.size()[0], seq.size()[1]
        subsequent_mask = torch.tril(torch.ones((len_s, len_s), device = seq.device, dtype = torch.uint8), diagonal=0)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) 
        return subsequent_mask

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)


        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)
        
        c_max_len = cw_idxs.shape[1]
        q_max_len = qw_idxs.shape[1]

        c_emb = c_emb.view(-1, c_max_len, self.hidden_size)
        q_emb = q_emb.view(-1, q_max_len, self.hidden_size)

        batch_size_c = c_emb.size()[0]
        batch_size_q = q_emb.size()[0]

        c_mask = c_mask.unsqueeze(2)
        q_mask = q_mask.unsqueeze(2)

        c_mask_inv = 1 - c_mask
        q_mask_inv = 1 - q_mask

        c_0, c_1 = self.transformer(c_emb, c_mask_inv, batch_size_c, c_max_len)
        q_0, q_1 = self.transformer(q_emb, q_mask_inv, batch_size_q, q_max_len)

        m_0 = self.att(c_0, q_0, c_mask, q_mask)
        m_1 = self.att(c_1, q_1, c_mask, q_mask)

        out = self.out(m_0, m_1, c_mask.squeeze(dim=2))

        return out


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()

        # print("vectors: ", word_vectors)
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors = char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

        self.hidden_size = hidden_size

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        
        c_max_len = cw_idxs.shape[1]
        q_max_len = qw_idxs.shape[1]


        c_emb = c_emb.view(-1, c_max_len, self.hidden_size)
        q_emb = q_emb.view(-1, q_max_len, self.hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


