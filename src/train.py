import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, head_dim, base, block_size):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(block_size).float()
        freqs = torch.outer(t, inv_freq)                 # (block_size, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)          # (block_size, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x):                                # x: (B, nh, T, hd)
        T = x.size(2)
        cos = self.cos[:T][None, None]                   # (1, 1, T, hd)
        sin = self.sin[:T][None, None]
        return x * cos + rotate_half(x) * sin


class GQA(nn.Module):
  def __init__(self, config, is_local):
    super().__init__()
    assert config.n_head % config.n_kv_head == 0
    self.query = nn.Linear(config.n_embd, config.n_head    * config.head_dim, bias=False)
    self.key   = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
    self.value = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
    self.c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)

    self.n_head    = config.n_head
    self.n_kv_head = config.n_kv_head
    self.head_dim  = config.head_dim
    self.n_embd    = config.n_embd
    self.is_local  = is_local

    self.q_norm = nn.RMSNorm(config.head_dim)
    self.k_norm = nn.RMSNorm(config.head_dim)

    # RoPE only on local layers; global layers dont have any
    self.rope = RoPE(config.head_dim, config.rope_base, config.block_size) if is_local else None

    # sliding-window mask (local) or full-causal mask (global)
    bs = config.block_size
    causal = torch.tril(torch.ones(bs, bs))
    mask = causal - torch.tril(torch.ones(bs, bs), -config.window_size) if is_local else causal
    self.register_buffer("bias", mask.view(1, 1, bs, bs))

  def forward(self, x):
    B, T, C = x.size()
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    q = q.view(B, T, self.n_head,    self.head_dim).transpose(1, 2)   # (B, nh,  T, hs)
    k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)   # (B, nkv, T, hs)
    v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)   # (B, nkv, T, hs)

    q = self.q_norm(q)                               # QK-norm before RoPE
    k = self.k_norm(k)

    if self.rope is not None:                        # NoPE on global layers
      q = self.rope(q)
      k = self.rope(k)                               # k still nkv heads — cheaper

    heads_per_group = self.n_head // self.n_kv_head
    k = k.repeat_interleave(heads_per_group, dim=1)  # (B, nh, T, hs)
    v = v.repeat_interleave(heads_per_group, dim=1)

    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v                                      # (B, nh, T, hs)

    y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
    y = self.c_proj(y)
    return y
    