# HW Debug AI

> A 10B parameter language model trained from scratch on hardware and embedded systems debugging data.

![Model](https://img.shields.io/badge/Model-10B%20Parameters-blue) ![Corpus](https://img.shields.io/badge/Corpus-346GB-green) ![TPU](https://img.shields.io/badge/Compute-Google%20TPU%20v6e-orange) ![Status](https://img.shields.io/badge/Status-Pretraining-yellow)

## Overview

HW Debug AI is a domain-specific LLM for cross-layer hardware-software debugging in embedded systems. General-purpose models fail at this problem because they lack grounding in hardware internals: I2C HAL configurations, STM32 clock trees, U-Boot partition tables, Cortex-M fault registers. This model is trained exclusively on hardware-domain natural language to fix that.

**Problem it solves:** An engineer debugging a hard fault on a Cortex-M4 does not need a model that knows Shakespeare. They need one that understands NVIC priority grouping, MPU region conflicts, and stack overflow signatures. No existing open model is trained for this.

**Long-term goal:** Acquisition by a semiconductor or embedded tooling company (Qualcomm, NXP, TI, Bosch) or deployment as a standalone debugging assistant embedded in IDEs and JTAG toolchains.

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | 10B |
| Layers | 48 |
| Hidden dim | 5120 |
| Attention heads | 40 |
| MLP | SwiGLU (dim 20480) |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| Precision | bfloat16 |
| Context length | 4096 tokens |
| Vocabulary | 32K SentencePiece |

## Training Infrastructure

- **Compute:** Google TPU v6e-8, 64 chips, europe-west4-a (TRC research grant)
- **Project:** rohith-llm-training
- **Checkpoints:** gs://rohith-llm-checkpoints/
- **Tokenizer:** gs://rohith-llm-checkpoints/hw_tok.model

## Corpus

346GB of hardware-domain natural language. No general web crawl. No code dumps.

| Source | Description |
|--------|-------------|
| Electronics Stack Exchange | 3M+ Q&A on embedded, PCB, signal integrity |
| Linux Kernel Mailing Lists | Driver, HAL, and subsystem debug threads |
| Zephyr RTOS Issues | Real firmware bug reports and fixes |
| NXP / Segger Community Forums | MCU debugging and peripheral configuration |
| RTLCoder Verilog Dataset | Hardware description and synthesis discussions |
| EDA Corpus | PCB design rule violations, signal integrity |
| Arduino / RPi Stack Exchange | Embedded prototyping Q&A |
| IETF RFCs | Protocol specs for embedded networking |
| Robotics Stack Exchange | Sensor fusion, motor control, realtime OS |
| Ham Radio Stack Exchange | RF, signal processing, hardware interfacing |
| GitHub Issues (70+ repos) | Firmware bug reports from real embedded projects |
| ArXiv (EE/CS) | Academic papers on hardware-software co-design |

## Proof of Concept: 1.6B Model

Before scaling to 10B, a 1.6B parameter model was trained to validate the approach:

- 500,000 training iterations completed
- ~16B tokens processed
- Architecture: 24 layers, 2048 dim, 16 heads, SwiGLU, RoPE, RMSNorm
- Checkpoint: hw_debug_1b_iter376000.pt

### Sample Outputs

**PROMPT:** I2C bus hanging on STM32 after transmit, SCL stuck low.
Model correctly identifies I2C bus contention and HAL driver conflict, outputs SCL recovery sequence and clock stretching configuration.

**PROMPT:** UART receiving garbage data on ESP32 after changing baud rate.
Model generates structured debug steps covering clock source verification, UART divider register, and APB clock configuration.

**PROMPT:** Hard fault exception on Cortex-M4 when accessing peripheral register.
Model identifies HARDFAULT_HANDLER_SOFTWARE_FAULT root cause, outputs MPU configuration steps and CFSR register decode procedure.

**PROMPT:** U-Boot fails to boot Linux kernel, error: bad magic number.
Model identifies partition table mismatch, outputs correct FIT image load address and bootargs configuration.

## Current Status

- [x] 1.6B proof-of-concept trained to 500K iterations
- [x] 346GB domain corpus assembled and cleaned
- [x] 32K hardware-domain tokenizer trained
- [x] 10B architecture finalized
- [x] Tokenized binary for 10B run ready (train_tokens_10b.bin)
- [ ] 10B pretraining run — in progress, TPU access expires June 30 2026
- [ ] Evaluation on hardware debugging benchmarks
- [ ] Public model release

## Repository Structure

- train/pretrain.py — 10B pretraining script (TPU v6e, PyTorch XLA)
- train/finetune.py — Supervised fine-tuning on debug Q&A pairs
- data/corpus_sources.md — Full list of data sources and filtering criteria
- data/download_all.py — Scraper orchestration
- data/convert_se.py — Stack Exchange XML to JSONL pipeline
- model/config.py — Architecture definition
- tokenizer/train_tokenizer.py — 32K SentencePiece tokenizer training
- eval/eval.py — Domain-specific evaluation harness

## About

Built by Rohith Kapelli, ECE undergraduate at SRM Institute of Science and Technology, Chennai (2024-2028). Independent research project. Full IP ownership.

Google TRC research grant recipient. Emergent Ventures applicant.

Contact: kapellirohith@gmail.com
