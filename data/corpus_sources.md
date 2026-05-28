# HW Debug AI — Corpus Sources

Total size: **346GB** of hardware-domain natural language.

## Filtering Criteria

All data must satisfy:
- Natural language (not raw code dumps)
- Hardware or embedded systems domain
- Contains debugging, troubleshooting, or technical explanation content
- Minimum 50 words per record
- No general-purpose programming Q&A (StackOverflow Python/JS excluded)

Excluded:
- HuggingFace pre-built datasets (produced garbage in 1.34B run)
- Common Crawl (too noisy, domain signal too weak)
- Raw source code files

## Sources

### Stack Exchange XML Dumps (Official)
Downloaded from archive.org Stack Exchange data dump.

| Site | Records | Notes |
|------|---------|-------|
| Electronics Stack Exchange | 3.2M+ | Core source. PCB, MCU, signal integrity, power |
| Arduino Stack Exchange | 420K | Embedded prototyping, peripheral config |
| Raspberry Pi Stack Exchange | 380K | Linux embedded, GPIO, SPI/I2C |
| Unix & Linux Stack Exchange | 1.1M | Filtered for driver/kernel/embedded content |
| Robotics Stack Exchange | 210K | Motor control, sensor fusion, RTOS |
| Reverse Engineering Stack Exchange | 180K | Firmware analysis, disassembly, protocol decode |
| Signal Processing Stack Exchange | 290K | DSP, ADC, filters — hardware-relevant subset |
| Ham Radio Stack Exchange | 150K | RF hardware, antenna, SDR |
| IoT Stack Exchange | 95K | Embedded connectivity, protocol stacks |

### Linux Kernel Mailing Lists
- lkml.org archive (2000–2025)
- Filtered for: driver bugs, HAL issues, subsystem debug threads
- Estimated records: 800K+ threads

### Zephyr RTOS GitHub Issues
- All closed issues with >2 comments
- Covers: UART, SPI, I2C, CAN, USB, Bluetooth subsystems
- ~45K issues

### NXP Community Forum
- Scraped via sitemap
- Covers: i.MX series, Kinetis, LPC, S32K
- ~120K posts

### Segger Forum
- J-Link, Ozone, RTT, SystemView debug discussions
- ~35K posts

### RTLCoder Verilog Dataset
- Hardware description language discussions, synthesis errors
- EDA tool output interpretation

### EDA / PCB Corpus
- Altium, KiCad, Eagle community forums
- Design rule violations, impedance matching, thermal analysis

### GitHub Issues (70+ repos)
Repositories scraped for hardware-relevant bug reports:
- zephyr-project (kernel + drivers)
- esp-idf (ESP32 platform)
- stm32-hal (STM32 HAL)
- openocd (JTAG debugging)
- u-boot (bootloader)
- linux (driver subsystems: i2c, spi, usb, can, iio)
- FreeRTOS (RTOS bugs)
- mongoose-os, riot-os, mbed-os
- nuttx, chibios, contiki-ng
- ... and 60+ more

### ArXiv (EE/CS)
- Filtered for: embedded systems, FPGA, SoC design, hardware verification
- ~180K abstracts + selected full papers

### IETF RFCs
- Protocol specifications relevant to embedded networking
- CoAP, MQTT, 6LoWPAN, Thread, BLE specs

### Vendor Documentation Excerpts
- STM32 application notes (AN series)
- NXP application notes
- TI reference designs
- Qualcomm embedded SDK docs

## Pipeline

1. Download raw data
2. Parse to JSONL (one record per line: {text, source, quality_score})
3. Quality filter: length, language detection, domain keyword presence
4. Dedup with MinHash LSH (threshold 0.85)
5. Tokenize with hw_tok.model (32K SentencePiece)
6. Pack into train_tokens_10b.bin for TPU training

## Storage

- Raw: `~/new_data/` (Mac)
- Processed JSONL: `MASTER_CORPUS.jsonl` (Google Drive)
- Tokenized binary: `train_tokens_10b.bin` (GCS + Drive)
