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

def softmax(x, dim = -1):
    x_max = torch.amax(x, dim = dim, keepdim = True)
    x = x - x_max
    exp_x = torch.exp(x)
    result = exp_x / torch.sum(exp_x, dim = dim, keepdim=True)
    return result

def scaled_dot_product_attention(Q, K, V, mask=None):
    dk = Q.shape[-1]
    product = Q @ K.transpose(-2,-1) / math.sqrt(dk)
    if mask is not None:
        product = product.masked_fill(~mask, float('-inf'))
    attention = softmax(product)
    result = attention @ V

    return result

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len)
        else:
            self.rope = None
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, x, token_positions = None):
        B, N, D = x.shape
        Q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        if token_positions is None:
            token_positions = torch.arange(N, device = x.device).unsqueeze(0).unsqueeze(0)
        else:
            token_positions = token_positions.unsqueeze(1)
        if self.rope is not None:
            q_rope = self.rope(Q, token_positions)
            k_rope = self.rope(K, token_positions)
        else:
            q_rope = Q
            k_rope = K
        mask = torch.tril(torch.ones(N, N, dtype=bool, device=x.device))
        attention = scaled_dot_product_attention(q_rope, k_rope, V, mask)

        result = attention.transpose(1,2).reshape(B, N, D)
        result = self.o_proj(result)
        return result


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        assert d_k % 2 == 0
        j = torch.arange(self.d_k // 2)
        p = torch.arange(max_seq_len)
        omega = theta ** (-2 * j / self.d_k)
        phi = p.unsqueeze(1) * omega
        self.register_buffer('cos_cache', torch.cos(phi), persistent = False)
        self.register_buffer('sin_cache', torch.sin(phi), persistent = False)
            
    def forward(self, x, token_positions):
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        even_rotated = x_even * cos - x_odd * sin

        odd_rotated = x_even * sin + x_odd * cos

        merged = torch.stack([even_rotated, odd_rotated], dim=-1).flatten(-2)

        return merged


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions = None):
        h = x + self.attn(self.norm1(x), token_positions)
        y = h + self.ffn(self.norm2(h))
        return y


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices):
        hs1 = self.token_embeddings(in_indices)
        for block in self.layers:
            hs1 = block(hs1)
        hs2 = self.norm(hs1)
        logits = self.lm_head(hs2)
        return logits