"""
HW Debug AI — 10B Parameter Pretraining Script
Target: Google TPU v6e-8, PyTorch XLA
Project: rohith-llm-training
Checkpoints: gs://rohith-llm-checkpoints/
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import sentencepiece as spm
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────

GCS_BUCKET       = "gs://rohith-llm-checkpoints"
TOKENIZER_PATH   = f"{GCS_BUCKET}/hw_tok.model"
TOKEN_BIN        = f"{GCS_BUCKET}/train_tokens_10b.bin"
CHECKPOINT_DIR   = f"{GCS_BUCKET}/checkpoints"

N_LAYERS         = 48
N_EMBD           = 5120
N_HEADS          = 40
MLP_HIDDEN       = 20480       # SwiGLU intermediate
BLOCK_SIZE       = 4096
VOCAB_SIZE       = 32000
ROPE_THETA       = 10000.0

BATCH_SIZE       = 16          # per TPU core
GRAD_ACCUM       = 4
LR               = 3e-4
LR_MIN           = 3e-5
WARMUP_STEPS     = 2000
MAX_STEPS        = 500000
SAVE_EVERY       = 5000
LOG_EVERY        = 100
WEIGHT_DECAY     = 0.1
GRAD_CLIP        = 1.0
DTYPE            = torch.bfloat16

# ── RoPE ─────────────────────────────────────────────────────────────────────

def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ── Model Blocks ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        scores = torch.softmax(scores.float(), dim=-1).type_as(x)
        out = torch.matmul(scores, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, mlp_hidden):
        super().__init__()
        self.attn_norm = RMSNorm(n_embd)
        self.attn = Attention(n_embd, n_heads)
        self.mlp_norm = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, mlp_hidden)

    def forward(self, x, freqs_cis, mask=None):
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class HWDebugLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([
            Block(N_EMBD, N_HEADS, MLP_HIDDEN) for _ in range(N_LAYERS)
        ])
        self.norm = RMSNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.freqs_cis = precompute_freqs_cis(N_EMBD // N_HEADS, BLOCK_SIZE, ROPE_THETA)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)
        freqs_cis = self.freqs_cis[:T].to(x.device)
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        for block in self.blocks:
            x = block(x, freqs_cis, mask)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), targets.view(-1)
            )
        return logits, loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx:idx + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


# ── LR Schedule ───────────────────────────────────────────────────────────────

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(rank, args):
    device = xm.xla_device()
    model = HWDebugLM().to(device).to(DTYPE)

    # Load checkpoint if resuming
    start_step = 0
    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        start_step = state["step"]
        xm.master_print(f"Resumed from step {start_step}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

    dataset = TokenDataset(args.data_path, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    para_loader = pl.MpDeviceLoader(loader, device)

    model.train()
    step = start_step
    t0 = time.time()

    for x, y in para_loader:
        if step >= MAX_STEPS:
            break

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        logits, loss = model(x, y)
        loss = loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            xm.reduce_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            xm.mark_step()

        if step % LOG_EVERY == 0:
            xm.master_print(
                f"step {step} | loss {loss.item() * GRAD_ACCUM:.4f} | "
                f"lr {lr:.2e} | {(time.time() - t0):.1f}s"
            )
            t0 = time.time()

        if step % SAVE_EVERY == 0 and step > 0:
            xm.master_print(f"Saving checkpoint at step {step}")
            xm.save({
                "model": model.state_dict(),
                "step": step,
                "config": {
                    "n_layers": N_LAYERS, "n_embd": N_EMBD,
                    "n_heads": N_HEADS, "vocab_size": VOCAB_SIZE
                }
            }, os.path.join(args.checkpoint_dir, f"ckpt_{step:07d}.pt"))

        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=TOKEN_BIN)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    xmp.spawn(train, args=(args,), nprocs=8, start_method="fork")
