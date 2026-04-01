# ============================================================
# FULL TRAINING PIPELINE — Real data, real model, checkpoints
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time
import sys
sys.stdout.reconfigure(line_buffering=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ── Load real data ────────────────────────────────────────
with open('data/combined.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset: {len(text):,} characters")

chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size} unique characters")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Train / Val split ─────────────────────────────────────
data = torch.tensor(encode(text), dtype=torch.long).to(device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

# ── Hyperparameters ───────────────────────────────────────
block_size    = 256
batch_size    = 16
n_embd        = 512
n_heads       = 8
n_layers      = 12
dropout       = 0.1
learning_rate = 3e-4
epochs        = 100000
eval_interval = 500

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,), device=device)
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(20)
        for k in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ── Model (same architecture, bigger) ────────────────────
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2,-1) * C**-0.5
        scores = scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)
        return scores @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ff   = FeedForward()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb    = nn.Embedding(vocab_size, n_embd)
        self.position_emb = nn.Embedding(block_size, n_embd)
        self.blocks       = nn.Sequential(*[TransformerBlock() for _ in range(n_layers)])
        self.ln_final     = nn.LayerNorm(n_embd)
        self.head         = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        x = self.token_emb(x) + self.position_emb(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            x_crop = x[:, -block_size:]
            logits, _ = self(x_crop)
            logits = logits[:, -1, :]
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_t], dim=1)
        return x

# ── Train ─────────────────────────────────────────────────
model = GPT().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
print("Training on Pride and Prejudice...\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_time = time.time()

for step in range(epochs):
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        torch.save({"model_state": model.state_dict(), "vocab_size": vocab_size, "stoi": stoi, "itos": itos}, "checkpoints/model_overnight.pt")
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"Step {step:4d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | time: {elapsed:.0f}s")

# ── Save checkpoint ───────────────────────────────────────
os.makedirs('checkpoints', exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'vocab_size':  vocab_size,
    'stoi': stoi,
    'itos': itos,
}, 'checkpoints/model_overnight.pt')
print("\nModel saved to checkpoints/model_overnight.pt")

# ── Generate ──────────────────────────────────────────────
print("\n--- Generated text ---")
start = torch.zeros((1,1), dtype=torch.long)
output = model.generate(start, max_tokens=300)
print(decode(output[0].tolist()))