import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    base_width: int = 256
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        bsz, seqlen, width = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(width, dim=2)
        q = q.view(bsz, seqlen, self.n_head, width // self.n_head).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, width // self.n_head).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, width // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, width)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MuPGPT(nn.Module):
    """GPT with explicit muP-style width-aware initialization and optimizer groups.

    This is intentionally local rather than relying on a separate muP package, so
    the scaling assumptions are visible in the experiment. Width multiplier is
    `n_embd / base_width`; matrix-like hidden/readout learning rates are scaled
    by `1 / width_multiplier` in `param_groups`.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.width_mult = config.n_embd / config.base_width
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(self.width_mult))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        bsz, seqlen = idx.size()
        if seqlen > self.config.block_size:
            raise ValueError(f"Cannot forward sequence of length {seqlen}; block size is {self.config.block_size}")
        pos = torch.arange(0, seqlen, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def crop_block_size(self, block_size: int):
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def param_groups(self, lr: float, weight_decay: float):
        decay, no_decay, scaled = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "ln_" in name or "ln_f" in name:
                no_decay.append(param)
            elif name.endswith("c_proj.weight") or name.endswith("lm_head.weight"):
                scaled.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay, "mup_group": "hidden"},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0, "mup_group": "no_decay"},
            {
                "params": scaled,
                "lr": lr / max(self.width_mult, 1e-12),
                "weight_decay": weight_decay,
                "mup_group": "readout_or_residual_projection",
            },
        ]

    def estimate_mfu_params(self):
        return sum(p.numel() for p in self.parameters())
