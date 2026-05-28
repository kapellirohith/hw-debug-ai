"""
HW Debug AI — Supervised Fine-Tuning Script
Trains on MASTER_FINETUNE.jsonl: hardware debug Q&A pairs
Format: {"prompt": "...", "completion": "..."}
"""

import os
import json
import math
import time
import argparse
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import sentencepiece as spm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from train.pretrain import HWDebugLM, RMSNorm, get_lr

# ── Config ────────────────────────────────────────────────────────────────────

GCS_BUCKET        = "gs://rohith-llm-checkpoints"
PRETRAIN_CKPT     = f"{GCS_BUCKET}/checkpoints/latest.pt"
FINETUNE_DATA     = f"{GCS_BUCKET}/MASTER_FINETUNE.jsonl"
FINETUNE_CKPT_DIR = f"{GCS_BUCKET}/finetune"
TOKENIZER_PATH    = f"{GCS_BUCKET}/hw_tok.model"

BLOCK_SIZE        = 2048
BATCH_SIZE        = 8
GRAD_ACCUM        = 4
LR                = 1e-5          # Lower LR for fine-tuning
LR_MIN            = 1e-6
WARMUP_STEPS      = 100
MAX_EPOCHS        = 3
SAVE_EVERY        = 500
LOG_EVERY         = 50
WEIGHT_DECAY      = 0.01
GRAD_CLIP         = 1.0
DTYPE             = torch.bfloat16

PROMPT_TEMPLATE   = "PROMPT: {prompt}\n\nRESPONSE:"
EOS_TOKEN         = 1             # </s> in hw_tok.model

# ── Dataset ───────────────────────────────────────────────────────────────────

class FinetuneDataset(Dataset):
    """
    Loads hardware debug Q&A pairs from JSONL.
    Each record: {"prompt": str, "completion": str}
    Tokenizes and packs into (input_ids, labels) where prompt tokens
    are masked (-100) so loss is computed only on completion tokens.
    """

    def __init__(self, path, tokenizer_path, block_size):
        self.tok = spm.SentencePieceProcessor()
        self.tok.Load(tokenizer_path)
        self.block_size = block_size
        self.examples = []

        with open(path) as f:
            for line in f:
                record = json.loads(line.strip())
                prompt_text = PROMPT_TEMPLATE.format(prompt=record["prompt"])
                completion_text = record["completion"]

                prompt_ids = self.tok.Encode(prompt_text)
                completion_ids = self.tok.Encode(completion_text) + [EOS_TOKEN]

                input_ids = prompt_ids + completion_ids
                labels = [-100] * len(prompt_ids) + completion_ids

                if len(input_ids) > block_size:
                    input_ids = input_ids[:block_size]
                    labels = labels[:block_size]
                else:
                    pad_len = block_size - len(input_ids)
                    input_ids = input_ids + [0] * pad_len
                    labels = labels + [-100] * pad_len

                self.examples.append((input_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ── Loss (masked) ─────────────────────────────────────────────────────────────

def masked_cross_entropy(logits, labels):
    B, T, V = logits.shape
    logits_flat = logits.view(-1, V)
    labels_flat = labels.view(-1)
    return nn.functional.cross_entropy(
        logits_flat, labels_flat, ignore_index=-100
    )


# ── Fine-Tuning Loop ──────────────────────────────────────────────────────────

def finetune(rank, args):
    device = xm.xla_device()

    # Load pretrained model
    model = HWDebugLM().to(device).to(DTYPE)
    state = torch.load(args.pretrain_ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    xm.master_print(f"Loaded pretrained checkpoint from step {state['step']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

    dataset = FinetuneDataset(args.data_path, args.tokenizer_path, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    para_loader = pl.MpDeviceLoader(loader, device)

    total_steps = MAX_EPOCHS * len(loader)
    step = 0
    model.train()
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        xm.master_print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
        for x, y in para_loader:
            lr = LR_MIN + 0.5 * (LR - LR_MIN) * (
                1 + math.cos(math.pi * max(0, step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS))
            ) if step >= WARMUP_STEPS else LR * step / max(1, WARMUP_STEPS)

            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits, _ = model(x)
            loss = masked_cross_entropy(logits, y) / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                xm.reduce_gradients(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()
                xm.mark_step()

            if step % LOG_EVERY == 0:
                xm.master_print(
                    f"[FT] epoch {epoch+1} step {step} | "
                    f"loss {loss.item() * GRAD_ACCUM:.4f} | "
                    f"lr {lr:.2e} | {time.time() - t0:.1f}s"
                )
                t0 = time.time()

            if step % SAVE_EVERY == 0 and step > 0:
                xm.save({
                    "model": model.state_dict(),
                    "step": step,
                    "epoch": epoch,
                }, os.path.join(args.checkpoint_dir, f"ft_ckpt_{step:06d}.pt"))

            step += 1

    # Save final
    xm.save({
        "model": model.state_dict(),
        "step": step,
        "epoch": MAX_EPOCHS,
    }, os.path.join(args.checkpoint_dir, "ft_final.pt"))
    xm.master_print("Fine-tuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_ckpt", default=PRETRAIN_CKPT)
    parser.add_argument("--data_path", default=FINETUNE_DATA)
    parser.add_argument("--tokenizer_path", default=TOKENIZER_PATH)
    parser.add_argument("--checkpoint_dir", default=FINETUNE_CKPT_DIR)
    args = parser.parse_args()

    xmp.spawn(finetune, args=(args,), nprocs=8, start_method="fork")
