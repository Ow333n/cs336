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

class RMSNorm(nn.Module):
	def __init__(self, d_model, eps=1e-5):
		super().__init__()
		self.d_model = d_model
		self.eps = eps
		self.weights = nn.Parameter(torch.ones(d_model,))
	
	def forward(self, x):
		x_original = x.dtype
		x = x.to(dtype = torch.float32)
		m = (x**2).mean(dim = -1, keepdim = True)
		x = x / torch.sqrt(m + self.eps)
		result = x * self.weights
		result = result.to(dtype = x_original)
		return result

def silu(x):
	sig = torch.sigmoid(x)
	result = x * sig
	return result


class SwiGLU(nn.Module):
	def __init__(self, d_model, d_ff):
		super().__init__()
		self.d_model = d_model
		self.d_ff = d_ff
		self.w1 = Linear(d_model, d_ff)
		self.w3 = Linear(d_model, d_ff)
		self.w2 = Linear(d_ff, d_model)
	
	def forward(self, x):
		
		W1 = self.w1(x)
		W3 = self.w3(x)
		W1 = silu(W1)
		W2 = W1 * W3

		result = self.w2(W2)
		return result