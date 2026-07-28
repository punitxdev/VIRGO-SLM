import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, d_k, max_seq_length=2048, base=10000.0):
        super(RotaryEmbedding, self).__init__()
        assert d_k % 2 == 0, "Head dimension must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_length).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, Q, K):
        seq_length = Q.size(-2)
        cos = self.cos[:, :, :seq_length, :].to(dtype=Q.dtype)
        sin = self.sin[:, :, :seq_length, :].to(dtype=Q.dtype)
        Q_even = Q[..., 0::2]
        Q_odd = Q[..., 1::2]
        K_even = K[..., 0::2]
        K_odd = K[..., 1::2]
        Q = torch.stack((Q_even * cos - Q_odd * sin, Q_even * sin + Q_odd * cos), dim=-1).flatten(-2)
        K = torch.stack((K_even * cos - K_odd * sin, K_even * sin + K_odd * cos), dim=-1).flatten(-2)
        return Q, K

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_length):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.d_k, max_seq_length)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        Q, K = self.rope(Q, K)
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=True)
        return self.W_o(self.combine_heads(attn_output))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_length, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, max_seq_length)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        norm_x = self.norm1(x)
        x = x + self.dropout(self.self_attn(norm_x))
        norm_x = self.norm2(x)
        x = x + self.dropout(self.feed_forward(norm_x))
        return x

class VirgoModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(VirgoModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_length, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds maximum {self.max_seq_length}")
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
