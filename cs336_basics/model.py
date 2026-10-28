import torch
import torch.nn as nn
import math

class Linear(nn.Module):
	def __init__(self, d_in, d_out):
		super().__init__()

		self.d_in = d_in
		self.d_out = d_out
		self.weights = nn.Parameter(torch.empty(d_out, d_in))
		std = math.sqrt(2 / (d_in + d_out))
		nn.init.trunc_normal_(self.weights, mean = 0, std = std, a = -3*std, b = 3*std)
	def forward(self, x):

		y = x @ self.weights.transpose(-2, -1)

		return y


class Embedding(nn.Module):
	def __init__(self, vocab_size, d_model):
		super().__init__()
		self.vocab_size = vocab_size
		self.d_model = d_model

		self.embedding = nn.Parameter(torch.empty(vocab_size, d_model))
		nn.init.trunc_normal_(self.embedding, mean = 0, std = 1, a = -3, b = 3)
	
	def forward(self, token_ids):
		result = self.embedding[token_ids]
		return result