# HW Debug AI — 1.34B Parameter LLM

Domain-specific language model for embedded systems debugging.
Trained from scratch on 18B tokens of hardware-specific data.

## Architecture
- 1.34B parameters, 24 layers, dim=2048, 16 heads
- Custom SentencePiece tokenizer (32K vocab)
- Trained on Google TPU v6e (TRC grant)

## Files
- `03_train.py` — pretraining script
- `finetune_hw_debug.py` — fine-tuning script  
- `convert_se_to_finetune.py` — Stack Exchange data pipeline
- `final_clean_finetune.py` — dataset cleaning pipeline
