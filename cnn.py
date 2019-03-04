#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i


### END YOUR CODE

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence



import random

class CNN(nn.Module):

	def __init__(self, e_char, e_word, m_word, k = 5):
		
		super(CNN, self).__init__()
		# self.w = nn.Conv1d((e_char, m_word), (f, m_word - k + 1), bias = True)
		self.w = nn.Conv1d(e_char, e_word, kernel_size = k, bias = True)


		# num_windows = m_word - k + 1
		self.maxPool = nn.MaxPool1d(m_word - k + 1)

    	

	def forward(self, x_reshaped):
 		# expecting: x_reshaped = (batch_size * sentence_length, m_word, e_char)
 		# returning: x_conv_out = (batch_size * sentence_length, e_word)

 		# print("x_reshaped: ", x_reshaped.size())
 		x_conv = self.w(x_reshaped)
 		# print("x_conv: ", x_conv.size())
 		x_relu = F.relu(x_conv)
 		# print("x_relu: " , x_relu.size())
 		x_conv_out = self.maxPool(x_relu)
 		x_conv_out = torch.squeeze(x_conv_out, 2)
 		# dims = x_conv_out.size()
 		# x_conv_out = x_conv_out.view(dims[0], -1)
 		# print("returning x_conv_out: ", x_conv_out.size())
 	
 		return x_conv_out

