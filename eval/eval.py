"""
HW Debug AI — Domain-Specific Evaluation Harness

Evaluates the model on hardware debugging prompts across categories:
- I2C / SPI / UART peripheral bugs
- Cortex-M hard faults and exception handling
- U-Boot / bootloader failures
- RTOS scheduling and timing issues
- Power supply and clock tree problems
- PCB signal integrity and EMC

Metrics:
- Perplexity on held-out hardware text
- Keyword hit rate (does output contain relevant debug terms?)
- Category accuracy (manual review subset)
"""

import os
import json
import math
import argparse
import torch
import sentencepiece as spm
from train.pretrain import HWDebugLM, BLOCK_SIZE, VOCAB_SIZE

# ── Evaluation Prompts ────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    {
        "category": "i2c",
        "prompt": "I2C bus hanging on STM32 after transmit, SCL stuck low. How do I debug this?",
        "keywords": ["SCL", "SDA", "clock stretching", "HAL_I2C", "bus reset", "BUSY flag", "I2CR"],
    },
    {
        "category": "uart",
        "prompt": "UART receiving garbage data on ESP32 after changing baud rate.",
        "keywords": ["baud", "clock divider", "APB", "UART_BRR", "FIFO", "framing error", "oversampling"],
    },
    {
        "category": "hard_fault",
        "prompt": "Hard fault exception on Cortex-M4 when accessing peripheral register at 0x40021000.",
        "keywords": ["CFSR", "HFSR", "BFSR", "MPU", "NVIC", "stack", "SCB", "fault handler"],
    },
    {
        "category": "uboot",
        "prompt": "U-Boot fails to boot Linux kernel, error: bad magic number.",
        "keywords": ["FIT image", "mkimage", "load address", "bootargs", "partition", "uImage", "zImage"],
    },
    {
        "category": "rtos",
        "prompt": "FreeRTOS task not running, watchdog reset occurring every 30 seconds.",
        "keywords": ["vTaskDelay", "watchdog", "priority", "stack overflow", "scheduler", "tick", "idle task"],
    },
    {
        "category": "spi",
        "prompt": "SPI communication failing on nRF52840, CS line toggling but no data received.",
        "keywords": ["MOSI", "MISO", "SCLK", "CS", "SPI mode", "CPOL", "CPHA", "DMA", "NSS"],
    },
    {
        "category": "clock",
        "prompt": "STM32H7 peripheral not working after changing system clock to 480MHz.",
        "keywords": ["PLL", "AHB", "APB", "prescaler", "HSE", "RCC", "clock divider", "HCLK"],
    },
    {
        "category": "dma",
        "prompt": "DMA transfer completing but data in buffer is corrupted on STM32.",
        "keywords": ["cache", "SCB_CleanDCache", "memory alignment", "DMA stream", "NDTR", "circular mode"],
    },
    {
        "category": "power",
        "prompt": "MCU brownout reset occurring randomly under load. Supply voltage looks stable on multimeter.",
        "keywords": ["decoupling capacitor", "BOR", "brownout", "bulk capacitor", "PDN", "VDD", "transient"],
    },
    {
        "category": "linker",
        "prompt": "Hard fault immediately on startup, before main() is reached on Cortex-M3.",
        "keywords": ["vector table", "startup", "stack pointer", "VTOR", "linker script", "BSS", "scatter file"],
    },
]


# ── Model Inference ───────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    model = HWDebugLM()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint (step {state.get('step', 'unknown')})")
    return model


def generate(model, tok, prompt, device, max_new_tokens=300, temperature=0.7, top_p=0.9):
    input_ids = tok.Encode(f"PROMPT: {prompt}\n\nRESPONSE:")
    x = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.shape[1] >= BLOCK_SIZE:
                break
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cumulative_probs > top_p] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token = sorted_idx[0, torch.multinomial(probs[0], 1)]

            if next_token.item() == 1:  # EOS
                break
            x = torch.cat([x, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    output_ids = x[0, len(input_ids):].tolist()
    return tok.Decode(output_ids)


def compute_perplexity(model, tok, text, device):
    ids = tok.Encode(text)
    if len(ids) < 2:
        return float("inf")
    ids = ids[:BLOCK_SIZE]
    x = torch.tensor([ids[:-1]], dtype=torch.long).to(device)
    y = torch.tensor([ids[1:]], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, loss = model(x, y)
    return math.exp(loss.item())


def keyword_hit_rate(output, keywords):
    hits = sum(1 for kw in keywords if kw.lower() in output.lower())
    return hits / len(keywords) if keywords else 0.0


# ── Main Eval ─────────────────────────────────────────────────────────────────

def run_eval(args):
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    tok = spm.SentencePieceProcessor()
    tok.Load(args.tokenizer_path)

    results = []
    total_kw_hit = 0.0

    print("\n" + "=" * 60)
    print("HW Debug AI — Evaluation Results")
    print("=" * 60)

    for item in EVAL_PROMPTS:
        output = generate(model, tok, item["prompt"], device,
                          max_new_tokens=args.max_tokens,
                          temperature=args.temperature)
        hit_rate = keyword_hit_rate(output, item["keywords"])
        total_kw_hit += hit_rate

        result = {
            "category": item["category"],
            "prompt": item["prompt"],
            "output": output,
            "keyword_hit_rate": hit_rate,
            "keywords_expected": item["keywords"],
        }
        results.append(result)

        print(f"\n[{item['category'].upper()}] hit_rate={hit_rate:.2f}")
        print(f"PROMPT: {item['prompt']}")
        print(f"OUTPUT: {output[:300]}...")
        print("-" * 60)

    avg_kw_hit = total_kw_hit / len(EVAL_PROMPTS)
    print(f"\nAverage keyword hit rate: {avg_kw_hit:.3f}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "avg_keyword_hit_rate": avg_kw_hit,
                "results": results,
            }, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", default="gs://rohith-llm-checkpoints/hw_tok.model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output_file", default="eval_results.json")
    args = parser.parse_args()

    run_eval(args)
