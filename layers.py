"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from util import masked_softmax
from cnn import CNN
import json
import math
import spacy
import numpy as np
from spacy.tokens import Doc
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
import copy

class Transformer(nn.Module):

    def __init__(self, hidden_size, dropout_prob = 0.1, num_layers = 6): 
        super(Transformer, self).__init__()

        self.dropout = nn.Dropout(p = dropout_prob)
        self.enc_layer = EncLayer(self.dropout, hidden_size)
        self.dec_layer = DecLayer(self.dropout, hidden_size)
        self.encoder = TransformerEncoder(self.enc_layer, num_layers)
        self.decoder = TransformerDecoder(self.dec_layer, num_layers)
        self.pos_enc = PositionalEncoder(hidden_size, self.dropout)
        # embedding: batch * seq_len * embed_size (word + chars + 4)

    def forward(self, x, pad_mask, batch_size, max_len):
        # print("x in transformer forward: ", x.size())
        x = self.pos_enc(x)
        print('x1: ', x)
        x = self.encoder(x, batch_size, pad_mask, max_len)
        print('x2: ', x)
        print("exiting encoder")
        return; 
        x = self.decoder(x, batch_size, pad_mask, max_len)

        m_0 = x
        # print('m_0: ', m_0)

        x = self.pos_enc(x)
        x = self.encoder(x, batch_size, pad_mask, max_len)
        x = self.decoder(x, batch_size, pad_mask, max_len)

        m_1 = x

        x = self.pos_enc(x)
        x = self.encoder(x, batch_size, pad_mask, max_len)
        x = self.decoder(x, batch_size, pad_mask, max_len)

        m_2 = x

        return m_0, m_1, m_2



class TransformerEncoder(nn.Module):

    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size, eps=1e-06)

    def forward(self, x, batch_size, pad_mask, max_len):
        for i, layer in enumerate(self.layers): 
            print('enc ', i)
            x = layer(x, batch_size, pad_mask, max_len)
            # print('x : ', x)
        # print('x encoder: ', x)
        x =  self.norm(x)
        return x




class TransformerDecoder(nn.Module):
    def __init__(self, layer, N):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size, eps=1e-06)

    def forward(self, x, batch_size, pad_mask, max_len):
        for i, layer in enumerate(self.layers):
            print('dec ', i)
            x = layer(x, pad_mask, batch_size, max_len)
        return self.norm(x)

class EncLayer(nn.Module):
    def __init__(self, dropout, hidden_size, num_headz = 8):
        super(EncLayer, self).__init__()

        self.self_att = MultiHeadSelfAttention(dropout, hidden_size)
        self.feed_fwd = FeedForward(hidden_size, dropout)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hidden_size, eps=1e-06)
        self.size = hidden_size

    def forward(self, x, batch_size, pad_mask, max_len):
        res1 = x
        x = self.self_att(x, batch_size, pad_mask, max_len)

        print("x selfatt IN ENC LAYER FORWARD: ", x)
        # return; 
        x = self.norm(x)
        print("x norm: ", x)
        x = self.dropout(x)
        # res2 = x
        res2 = res1 + x
        # print("x after res2: ", x.size())
        x = self.feed_fwd(x)
        print("x feedfwd: ", x)
        x = self.norm(x)

        x = self.dropout(x)
        x += res2 
        return x

def make_sub_mask(seq):
    print("seq: ", seq, type(seq))
    sz_b, len_s = seq.size()[0], seq.size()[1]
    # subsequent_mask = torch.tril(torch.ones((len_s, len_s), device = seq.device, dtype = torch.uint8), diagonal=0)
    subsequent_mask = torch.tril(torch.ones((len_s, 1), device = seq.device, dtype = torch.uint8), diagonal=0)
    print("sub mask in make mask: ", subsequent_mask)
    print("sub mask size in make mask: ", subsequent_mask.size())
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) 
    return subsequent_mask

class DecLayer(nn.Module):
    def __init__(self, dropout, hidden_size, num_headz = 8):
        super(DecLayer, self).__init__()

        self.masked_self_att = MultiHeadSelfAttention(dropout, hidden_size)
        self.self_att = MultiHeadSelfAttention(dropout, hidden_size)
        self.feed_fwd = FeedForward(hidden_size, dropout)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hidden_size, eps=1e-06)
        self.size = hidden_size

    def forward(self, x, pad_mask, batch_size, max_len):
        res1 = x
        sub_mask = make_sub_mask(x)
        print('sub_mask dec', sub_mask, sub_mask.shape) # (10, 203, 203)
        print('pad_mask dec', pad_mask, pad_mask.shape) # (10, 203, 203)

        # print("sub_mask before bit operation: ", sub_mask.size())
        sub_mask = 1 - sub_mask
        mask = sub_mask | pad_mask # (10, 203, 203)
        print('mask dec', mask, mask.shape) 
        
        # combination of pad_mask and sub_mask
        # print("mask dim: ", mask.size())

        #mask = pad_mask & sub_mask 
        # print("batch_size: ", batch_size)
        # print("pad mask: ", pad_mask)
        x = self.masked_self_att(x, batch_size, pad_mask, max_len, mask)
        # x = self.masked_self_att(x, batch_size, pad_mask, max_len) # using this line removes NAN! 
        x = self.norm(x)
        x = self.dropout(x)
        res2 = res1 + x
        print('res2 ', res2)
        print('padmask ', pad_mask)
        x = res2
        x = self.self_att(x, batch_size, pad_mask, max_len)
        x = self.norm(x)
        x = self.dropout(x)
        res3 = res2 + x
        x = res3
        x = self.feed_fwd(x)
        x = self.norm(x)
        x = self.dropout(x)
        x += res3
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dropout, hidden_size, num_headz=8):
        super(MultiHeadSelfAttention, self).__init__()

        self.self_attn = SelfAttention(dropout, hidden_size, hidden_size)
        self.self_attn_list = nn.ModuleList([copy.deepcopy(self.self_attn) for _ in range(num_headz)])
        self.dropout = dropout
        self.W_O = nn.Linear(num_headz * hidden_size, hidden_size)


    def forward(self, x, batch_size, pad_mask, max_len, sub_mask = None):
        # if sub_mask is not None:
        #     sub_mask = sub_mask.unsqueeze(1)
        device = torch.device("cpu")
        # if sub_mask is not None: 
            # print("submask: ", sub_mask.size())

        att_heads = torch.tensor([[[]] * max_len] * batch_size, device=device)

        # print("att_heads size: ", att_heads.size())
        for i, self_att in enumerate(self.self_attn_list):
            print('attn ', i)
            att_head = self_att(x, pad_mask, sub_mask)
            # print("att_head (SINGULAR) size: ", att_head)
            print("att head in mhm forward: ", att_head, att_head.size())
            att_heads = torch.cat((att_heads, att_head), 2)

        print("att headSSSSS in mhm forward: ", att_heads, att_heads.size())
        # return; 
        # print("att_heads dim: ", att_heads.size())
        att = self.W_O(att_heads)
        print("att in mhm forward: ", att, att.size())
        # print("att final dim: ", att.size())

        if sub_mask is not None: 
            # sub_mask = sub_mask.unsqueeze(2)
            att = att.masked_fill(sub_mask, -np.inf)
            # print('sub_mask in dot: ', sub_mask, sub_mask.size())
            # print("x in dot: ", x, x.size())
            # return; 
        else: 
            att = att.masked_fill(pad_mask, -np.inf)
            # print("att in mhm forward!!!!!!: ", att, att.size())
        print("att in mhm forward final: ", att, att.size())
        # return; 
        return att; 
        # return att_heads # batch_size, max_seq_len, n_headz * d_k

class SelfAttention(nn.Module):
    def __init__(self, dropout, hidden_size, d_k):
        super(SelfAttention, self).__init__()
        self.W_v = nn.Linear(hidden_size, d_k) # check on the second dimension of this - potentially d_k? hidden size? unclear
        self.W_k = nn.Linear(hidden_size, d_k)
        self.W_q = nn.Linear(hidden_size, d_k)

        self.hidden_size = hidden_size
        self.scaled_dot_prod = ScaledDotProductAttention(d_k)

    def forward(self, x, pad_mask, sub_mask = None):
        K_x = self.W_k(x) # batch_size * max_seq_len * d_k
        V_x = self.W_v(x) # batch_size * max_seq_len * d_k
        Q_x = self.W_q(x) # batch_size * max_seq_len * d_k
        # if sub_mask is not None: 
            # print("sub_mask: in selfatten forward", sub_mask.size())

        att = self.scaled_dot_prod(K_x, V_x, Q_x, pad_mask, sub_mask)
        # print("att 1 size: ", att.size())
        # print("att self: ", att)
        return att # batch_size * max_seq_len, d_k




class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, k, v, q, pad_mask, sub_mask = None):
        x = torch.bmm(q, torch.transpose(k, 1, 2)) # (batch_size, max_seq_len, max_seq_len)
        # do mask thing???? 


        # print("x size: ", x.size())
        # if sub_mask is not None: 
            # print("submask size: ", sub_mask.size())
        # else: 
            # print("submask is none!")
        # print("padmask size: ", pad_mask.size())
        # print("k size: ", k.size())
        # print("v size: ", v.size())
        # print("q size: ", q.size())
        # pad_mask = pad_mask.squeeze(2)

        # print('pad_mask in dot: ', pad_mask, pad_mask.shape)
        # if sub_mask is not None: 
        #     # sub_mask = sub_mask.unsqueeze(2)
        #     x = x.masked_fill(sub_mask, -np.inf)
        #     print('sub_mask in dot: ', sub_mask, sub_mask.size())
        #     print("x in dot: ", x, x.size())
        #     # return; 
        # else: 
        #     x = x.masked_fill(pad_mask, -np.inf)


        x /= math.sqrt(self.d_k) # same dim as above
        print("x in dot 2: ", x, x.size())
        x = F.softmax(x) # same dim as above
        print("x in dot 3: ", x, x.size())

        # if sub_mask is not None: 
        #     # sub_mask = sub_mask.unsqueeze(2)
        #     x = x.masked_fill(sub_mask, -np.inf)
        #     print('sub_mask in dot: ', sub_mask, sub_mask.size())
        #     print("x in dot: ", x, x.size())
        #     # return; 
        # else: 
        #     x = x.masked_fill(pad_mask, -np.inf)

        print("x in dot 4: ", x, x.size())


        print("v: ", v, v.size())
        att = torch.bmm(x, v) # batch_size, max_seq_len , d_k
        print('x att: ', att, att.shape)

        # if sub_mask is not None: 
        #     # sub_mask = sub_mask.unsqueeze(2)
        #     att = att.masked_fill(sub_mask, -np.inf)
        #     # print('sub_mask in dot: ', sub_mask, sub_mask.size())
        #     # print("x in dot: ", x, x.size())
        #     # return; 
        # else: 
        #     att = att.masked_fill(pad_mask, -np.inf)
        #     print("att in pad_mask thing: ", att, att.size())
            # return; 
        print("att after doing operation: ", att, att.size())

        return att # batch_size, max_seq_len , d_k


class FeedForward(nn.Module):

    def __init__(self, hidden_size, dropout, ff_size = 128, ff_size2 = 64):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(hidden_size, ff_size)
        self.W2 = nn.Linear(ff_size, ff_size2)
        self.W3 = nn.Linear(ff_size2, hidden_size)
        self.dropout = dropout


    def forward(self, x):
        x_2 = self.dropout(F.relu(self.W1(x)))
        x_3 = self.dropout(F.relu(self.W2(x_2)))
        return self.W3(x_3)



class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, dropout, max_len=400):
        super(PositionalEncoder, self).__init__()
        

        #REPLACE d_model with 
        # pe = torch.zeros(max_len, e_size)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = dropout

    def forward(self, x):
        # x size: (batch_size, 292 (max_seq_len), hidden_size)
        # print("x in positional encoder: ", x.size())
        # print("pe thing: ", self.pe[:, :x.size(1)].size())
        # z = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # print("z: ", z.size())
        # y = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # print("y: ", y)
        return self.dropout(x + Variable(self.pe[:, :x.size(1)], requires_grad=False))

class TransformerOutput(nn.Module):
    def __init__(self, hidden_size):
        super(TransformerOutput, self).__init__()
        self.W1 = nn.Linear(hidden_size, 1)
        self.W2 = nn.Linear(hidden_size, 1)

    def forward(self, m_0, m_1, m_2):
        m_01 = torch.cat((m_0, m_1), 1)
        m_02 = torch.cat((m_0, m_2), 1)
        print('m_01: ', m_01)
        p1 = F.softmax(self.W1(m_01), 1)
        print('p1: ', p1)
        p2 = F.softmax(self.W1(m_02), 1)
        p1 = p1.squeeze(2)
        p2 = p2.squeeze(2)
        print('p1: ', p1)

        return p1, p2

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed_w = nn.Embedding.from_pretrained(word_vectors)
        self.embed_c = nn.Embedding.from_pretrained(char_vectors, freeze = False)

        # added 4 here bc we added 4 features (dep, dep head, pos, ner)
        self.proj = nn.Linear(2 * word_vectors.size(1) + 4, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)
        # print("hidden_size: ", hidden_size)
        self.e_char = 64
        self.e_word = 300
        self.m_word = 16
        self.cnn = CNN(self.e_char, self.e_word, self.m_word)

        # with open ('wordidx2posidx.json') as json_file: 
        #     self.wordidx2posidx = json.load(json_file)

        with open('idx2word.json') as json_file:
            self.idx2word = json.load(json_file)
        # print(len(self.idx2word))

        with open('./data/word2idx.json') as json_file:
            self.word2idx = json.load(json_file)
            self.word2idx[''] = -1

        with open('dep2idx.json') as json_file:
            self.dep2idx = json.load(json_file)
            self.dep2idx[''] = -1

        with open('pos2idx.json') as json_file:
            self.pos2idx = json.load(json_file)
            self.pos2idx[''] = -1

        # with open('ent2idx.json') as json_file:
        #     self.ent2idx = json.load(json_file)
        #     self.ent2idx[''] = -1

        # print(self.dep2idx)



    def forward(self, x_w, x_c):
        # batch_size = 64, seq_len = ?
        # print("x_w size: ", x_w.size()) # (batch_size, max_sen_len)
        # print("x_c size: ", x_c.size()) # (batch_size, max_sen_len, max_word_len)


        emb_w = self.embed_w(x_w)  # (batch_size, seq_len, embed_size)
        emb_c = self.embed_c(x_c) 
  
        x_reshaped = emb_c.permute(0, 1, 3, 2)
        x_reshaped_new = x_reshaped.contiguous().view(-1, self.e_char, self.m_word)
        x_conv_out_c = self.cnn(x_reshaped_new) 

        emb_w = emb_w.view(-1, self.e_word)

        # pos_list = [[float(self.wordidx2posidx[str(int(word))]) for word in sentence] for sentence in x_w]

        # NULL_IDX = self.word2idx["--NULL--"]
        # OOV_IDX = self.word2idx["--OOV--"]

        # sentences = [[self.idx2word[str(int(word_int))] for word_int in sentence_int if int(word_int) != NULL_IDX and int(word_int) != OOV_IDX] for sentence_int in x_w]
        
        sentences_split = [[self.idx2word[str(int(word_int))] for word_int in sentence_int] for sentence_int in x_w]

        space = " "
        sentences = [space.join(sentences_split[i]) for i in range(len(sentences_split))]

        # sentences = (batch_size, sentence_length)
        nlp = spacy.load('en')

        nlp_sentences = [nlp(space.join(sentences_split[i])) for i in range(len(sentences_split))]
        for s, sentence in enumerate(nlp_sentences):
            sent_split = sentences_split[s]
            if len(sentence)!= len(sent_split):
                for i in range(0, len(sent_split)):
                    if str(sentence[i])!=str(sent_split[i]):
                        new_sentence = Doc(sentence.vocab, words = [word.text for t, word in enumerate(sentence) if t!=i])
                        sentence = new_sentence
                nlp_sentences[s] = sentence


        tag_list = [[[float(self.dep2idx[token.dep_]) if token.dep_ in self.dep2idx else -1, float(self.word2idx[token.head.text]) if token.head.text in self.word2idx else -1, float(self.pos2idx[token.pos_]) if token.pos_ in self.pos2idx else -1, float(1) if str(token) in [str(e) for e in sentence.ents] else float(0)] for token in sentence] for sentence in nlp_sentences] 
        tags = torch.tensor(tag_list)

        # ent_list = [[1 if str(token) in [str(e) for e in sentence.ents] else 0 for token in sentence] for sentence in nlp_sentences]

        # pos = torch.tensor(pos_list)
        # pos = pos.view(pos.shape[0], pos.shape[1], -1)
        # pos = pos.view(-1, pos.shape[2])
        tags = tags.view(-1, tags.shape[2])        
        concatenated = torch.cat((x_conv_out_c, emb_w, tags), 1)

        emb = F.dropout(concatenated, self.drop_prob, self.training)
        emb = self.proj(concatenated)
        emb = self.hwy(emb)

        print("embedding size: ", emb.size())

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        # print("passing into rnn: ", x)
        # print("type: ", type(x))
        # print("passing into rnn size: ", x.shape)
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        # print("x size (in LSTM forward): ", x.size())
        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        # print("c size: ", c.size())
        # print("q size: ", q.size())
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        # print("similarity matrix size: ", s.size())
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

