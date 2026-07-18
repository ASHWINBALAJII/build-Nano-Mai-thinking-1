from torch import topk
import math
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

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

    def forward(self, x):                                # x: (B, nh, T, hs)
        T = x.size(2)
        cos = self.cos[:T][None, None]                   # (1, 1, T, hs)
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

    # RoPE only on local layers , global layers dont have any
    self.rope = RoPE(config.head_dim, config.rope_base, config.block_size) if is_local else None

    # sliding-window mask (local) or full-causal mask (global)
 
    causal = torch.tril(torch.ones(config.block_size, config.block_size))
    mask = causal - torch.tril(torch.ones(config.block_size, config.block_size), -config.window_size) if is_local else causal
    self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

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
      k = self.rope(k)                               

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

class MLP(nn.Module):                         # dense FFN 
    def __init__(self, config):
        super().__init__()
        hidden = config.dense_ffn // 2        # FFN = 2*hidden
        self.c_fc   = nn.Linear(config.n_embd, hidden, bias=False)   # up
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)   # gate
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)   # down
    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))

class Expert(nn.Module):                       
    def __init__(self, config):
        super().__init__()
        hidden = config.expert_ffn //2     #Expert FFN = 2*hidden
        self.c_fc   = nn.Linear(config.d_latent, hidden, bias=False)
        self.c_gate = nn.Linear(config.d_latent, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.d_latent, bias=False)
    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))


class Router(nn.Module):                      # softmax gating on the ORIGINaL representation
    def __init__(self, config):
        super().__init__()
        self.top_k, self.num_experts = config.top_k, config.num_experts
        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)
    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        top_probs, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)   # renormalize chosen gates
        return top_probs, top_idx, probs

class SparseMoE(nn.Module):                   # Latent MoE
    def __init__(self, config):
        super().__init__()
        self.num_experts, self.top_k, self.aux_alpha = config.num_experts, config.top_k, config.aux_alpha
        self.router  = Router(config)
        self.w_down  = nn.Linear(config.n_embd, config.d_latent, bias=False)   # shared squeeze
        self.w_up    = nn.Linear(config.d_latent, config.n_embd, bias=False)   # shared expand
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
    def forward(self, x):
        B, T, C = x.size()
        gates, indices, probs = self.router(x)         # router sees ORIGINAL x ( that is the actual embedding dim)
        latent = self.w_down(x)                        # squeeze becomes (B, T, d_latent)
        fl = latent.view(-1, latent.size(-1))
        fg, fi = gates.view(-1, self.top_k), indices.view(-1, self.top_k)
        N = fi.size(0)
        out = torch.zeros_like(fl)                     # dropless accumulation in latent space
        for i, expert in enumerate(self.experts):
            m = (fi == i); tm = m.any(-1)
            if tm.any():
                gv = (fg * m).sum(-1)[tm].unsqueeze(1)
                out[tm] += expert(fl[tm]) * gv
        out = self.w_up(out.view(B, T, -1))            # expand -> (B, T, n_embd)
        # GShard global-batch load-balance loss
        counts = torch.zeros(self.num_experts, device=x.device)
        counts.scatter_add_(0, fi.view(-1), torch.ones(fi.numel(), device=x.device))
        f_i = counts / (N * self.top_k)
        P_i = probs.view(-1, self.num_experts).mean(0)
        aux = self.num_experts * (f_i * P_i).sum()
        return out, self.aux_alpha * aux

class Block(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        is_local=(layer_idx+1)%config.global_every != 0
        self.attn=GQA(config,is_local)
        self.ln_2=nn.RMSNorm(config.n_embd)
        # first is MLP then alternatively MOE and MLP
        self.use_moe=(layer_idx != 0) and (layer_idx % 2 == 1)
        self.ffn=SparseMoE(config) if self.use_moe else MLP(config)
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        if self.use_moe:
            f,aux=self.ffn(self.ln_2(x))
        else:
            f,aux=self.ffn(self.ln_2(x)), torch.zeros((), device=x.device)
        x=x+f
        return x,aux








@dataclass
class MAIConfig:
    
    n_layer:     int = 12          # L — must be a multiple of 6

    block_size:  int = 1024
    vocab_size:  int = 200019 


    head_dim:    int = 128
    n_kv_head:   int = 8
    num_experts: int = 512
    top_k:       int = 8
    window_size: int = 512
    rope_base:   int = 10000
    global_every: int = 6          # every 6th layer is global
    aux_alpha: float = 0.01

    #derived from L (filled automatically) 
    n_embd:      int = 0
    n_head:      int = 0
    dense_ffn:   int = 0
    d_latent:    int = 0
    expert_ffn:  int = 0

    def __post_init__(self):
        assert self.n_layer % 6 == 0, "L must be a multiple of 6 (last layer must be global)"
        D = self.n_layer * 256 // 3              # D = L * 256/3
        self.n_embd     = D
        self.n_head     = 8 * round(self.n_layer/8) # round L up to nearest 8
        self.dense_ffn  = 2 * D                  # dense FFN expands 2x  (module does //2)
        self.d_latent   = D // 2                 # Latent MoE: 2x compression
        self.expert_ffn = (3 * self.d_latent)  # experts expand 3x

class MAI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte  = nn.Embedding(config.vocab_size, config.n_embd)
        self.h    = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight          # weight tying 
    def forward(self,idx,targets=None):
        B,T=idx.size()
        tok_emb=self.wte(idx)
        total_aux=torch.zeros((),device=idx.device)
        x=tok_emb
        for block in self.h:
            x,aux=block(x)
            total_aux=total_aux+aux
        x=self.ln_f(x)
        logits=self.lm_head(x)
        total_loss=None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss=lm_loss+total_aux  # add aux loss
        return logits,total_loss 




#auto detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

import tiktoken
enc = tiktoken.get_encoding('o200k_base')

model = MAI(MAIConfig())
model.eval()
model.to(device)


with open('input.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text[:1000])
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
buf = buf.to(device)
x = buf[:-1].view(B, T).to(device)
y = buf[1:].view(B, T).to(device)


#optimize
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
for i in range(100):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step{i}, loss:{loss.item()}")



num_return_sequences = 5
max_length = 100

tokens = enc.encode("Hello, Ashwin here , and I ")
tokens = torch.tensor(tokens, dtype=torch.long)
x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)   # (5, T)

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, -1, ix)                   # (B, 1)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    print(">", enc.decode(x[i, :max_length].tolist()))





        








