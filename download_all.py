#!/usr/bin/env python3
"""
MASSIVE DATASET DOWNLOADER — Downloads 1.8TB directly to Google Drive
Run this, go to class, come back to 1.8TB of training data.

Usage:
    python3 download_all.py

Make sure Google Drive for Desktop is installed and syncing.
"""

from datasets import load_dataset
import json, os, gc, time, sys

# ── CONFIGURATION ─────────────────────────────────────────
DRIVE = os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-krohithsai99@gmail.com/My Drive/llm-data"
)

os.makedirs(f"{DRIVE}/small", exist_ok=True)
os.makedirs(f"{DRIVE}/code", exist_ok=True)
os.makedirs(f"{DRIVE}/web", exist_ok=True)

total_bytes = 0
start_time = time.time()

def log(msg):
    elapsed = time.time() - start_time
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print(f"[{hrs}h{mins:02d}m] {msg}", flush=True)

def download_small(name, filename, max_rows=2000000):
    """Download a HuggingFace dataset to Drive"""
    global total_bytes
    filepath = f"{DRIVE}/small/{filename}"
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"⏭️  Exists: {filename} ({size/1024/1024:.1f} MB)")
        return
    
    try:
        log(f"⬇️  {name}...")
        ds = load_dataset(name, streaming=True)
        count = 0
        with open(filepath, 'w') as f:
            for split in ds:
                for item in ds[split]:
                    f.write(json.dumps(item, default=str) + '\n')
                    count += 1
                    if count >= max_rows:
                        break
                if count >= max_rows:
                    break
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"✅ {filename}: {count:,} rows — {size/1024/1024:.1f} MB")
        del ds
        gc.collect()
    except Exception as e:
        log(f"❌ {name}: {str(e)[:100]}")
        # Retry without streaming
        try:
            ds = load_dataset(name, split='train')
            count = 0
            with open(filepath, 'w') as f:
                for item in ds:
                    f.write(json.dumps(item, default=str) + '\n')
                    count += 1
            size = os.path.getsize(filepath)
            total_bytes += size
            log(f"✅ Retry worked: {count:,} rows — {size/1024/1024:.1f} MB")
            del ds
            gc.collect()
        except Exception as e2:
            log(f"❌ Skip {name}: {str(e2)[:80]}")


def download_code(lang, data_dir=None, max_files=5000000):
    """Download code from The Stack"""
    global total_bytes
    filename = f"the_stack_{lang}.jsonl"
    filepath = f"{DRIVE}/code/{filename}"
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"⏭️  Exists: {filename} ({size/1024/1024/1024:.2f} GB)")
        return
    
    actual_dir = data_dir if data_dir else f"data/{lang}"
    
    try:
        log(f"⬇️  The Stack — {lang} (this will take hours)...")
        ds = load_dataset(
            'bigcode/the-stack-dedup',
            data_dir=actual_dir,
            split='train',
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        with open(filepath, 'w') as f:
            for item in ds:
                content = item.get('content', '')
                if len(content) < 100 or len(content) > 100000:
                    continue
                f.write(json.dumps({
                    'content': content,
                    'lang': lang,
                    'path': item.get('path', ''),
                    'repo': item.get('repository_name', ''),
                }) + '\n')
                count += 1
                if count % 200000 == 0:
                    size = os.path.getsize(filepath)
                    total_bytes_now = total_bytes + size
                    log(f"  {lang}: {count:,} files — {size/1024/1024/1024:.2f} GB (total: {total_bytes_now/1024/1024/1024:.1f} GB)")
                if count >= max_files:
                    break
        
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"✅ {lang}: {count:,} files — {size/1024/1024/1024:.2f} GB")
        gc.collect()
    except Exception as e:
        log(f"❌ The Stack {lang}: {str(e)[:100]}")


def download_fineweb():
    """Download FineWeb filtered for technical content"""
    global total_bytes
    import re
    
    filepath = f"{DRIVE}/web/fineweb_technical.jsonl"
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"⏭️  Exists: fineweb_technical.jsonl ({size/1024/1024/1024:.2f} GB)")
        return
    
    try:
        log("⬇️  FineWeb — filtering for technical content (takes hours)...")
        ds = load_dataset(
            'HuggingFaceFW/fineweb',
            name='sample-10BT',
            split='train',
            streaming=True,
            trust_remote_code=True
        )
        
        keywords = re.compile(
            r'arduino|esp32|stm32|raspberry pi|microcontroller|firmware|embedded|'
            r'i2c|spi|uart|gpio|pwm|kernel|driver|interrupt|bootloader|fpga|verilog|'
            r'pcb|oscilloscope|schematic|buffer overflow|exploit|vulnerability|cve-|'
            r'reverse engineer|assembly language|registers|stack pointer|debug|segfault|'
            r'core dump|gdb|valgrind|linux kernel|device driver|kernel module|dmesg|'
            r'cuda|gpu programming|opencl|risc-v|arm cortex|iot|mqtt|zigbee|'
            r'bluetooth low energy|lora|cybersecurity|penetration testing|nmap|metasploit|'
            r'cryptography|aes|rsa|encryption|tcp/ip|socket programming|'
            r'operating system|rtos|freertos|zephyr|signal processing|dsp|fft|'
            r'power supply|voltage regulator|mosfet|transistor|sensor|accelerometer|'
            r'python|javascript|rust|golang|java|swift|kotlin|ruby|php|sql|'
            r'git|linux|unix|bash|shell|terminal|command line|database|server|'
            r'api|docker|kubernetes|aws|azure|http|dns|ssl|tls|'
            r'neural network|machine learning|deep learning|transformer|pytorch|tensorflow|'
            r'compiler|linker|makefile|cmake|gcc|clang|llvm|'
            r'algorithm|data structure|binary tree|hash table|sorting|'
            r'operating system|process|thread|mutex|semaphore|deadlock|'
            r'memory management|heap|stack|pointer|malloc|buffer|'
            r'network|packet|protocol|router|switch|firewall|vpn|proxy',
            re.IGNORECASE
        )
        
        count = 0
        checked = 0
        max_items = 2000000
        
        with open(filepath, 'w') as f:
            for item in ds:
                checked += 1
                text = item.get('text', '')
                if len(text) < 300:
                    continue
                if keywords.search(text[:3000]):
                    f.write(json.dumps({
                        'text': text[:10000],
                        'url': item.get('url', ''),
                    }) + '\n')
                    count += 1
                    if count % 100000 == 0:
                        size = os.path.getsize(filepath)
                        log(f"  FineWeb: {count:,} pages (checked {checked:,}) — {size/1024/1024/1024:.2f} GB")
                    if count >= max_items:
                        break
                if checked >= 100000000:
                    break
        
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"✅ FineWeb: {count:,} pages — {size/1024/1024/1024:.2f} GB")
        gc.collect()
    except Exception as e:
        log(f"❌ FineWeb: {str(e)[:100]}")


def download_starcoder():
    """Download StarCoderData — massive code dataset"""
    global total_bytes
    
    languages = ['python', 'c', 'cpp', 'javascript', 'java', 'rust', 'go', 'shell', 'assembly']
    
    for lang in languages:
        filename = f"starcoder_{lang}.jsonl"
        filepath = f"{DRIVE}/code/{filename}"
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            size = os.path.getsize(filepath)
            total_bytes += size
            log(f"⏭️  Exists: {filename} ({size/1024/1024/1024:.2f} GB)")
            continue
        
        try:
            log(f"⬇️  StarCoder — {lang}...")
            ds = load_dataset(
                'bigcode/starcoderdata',
                data_dir=lang,
                split='train',
                streaming=True,
                trust_remote_code=True
            )
            
            count = 0
            max_per = 3000000
            
            with open(filepath, 'w') as f:
                for item in ds:
                    content = item.get('content', '')
                    if len(content) < 100 or len(content) > 100000:
                        continue
                    f.write(json.dumps({
                        'content': content,
                        'lang': lang,
                    }) + '\n')
                    count += 1
                    if count % 500000 == 0:
                        size = os.path.getsize(filepath)
                        log(f"  StarCoder/{lang}: {count:,} files — {size/1024/1024/1024:.2f} GB")
                    if count >= max_per:
                        break
            
            size = os.path.getsize(filepath)
            total_bytes += size
            log(f"✅ StarCoder/{lang}: {count:,} files — {size/1024/1024/1024:.2f} GB")
            gc.collect()
        except Exception as e:
            log(f"❌ StarCoder/{lang}: {str(e)[:100]}")


def download_redpajama():
    """Download RedPajama code/technical subset"""
    global total_bytes
    
    filepath = f"{DRIVE}/web/redpajama_code.jsonl"
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"⏭️  Exists: redpajama_code.jsonl ({size/1024/1024/1024:.2f} GB)")
        return
    
    try:
        log("⬇️  RedPajama — code subset...")
        ds = load_dataset(
            'togethercomputer/RedPajama-Data-V2',
            name='default',
            split='train',
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        max_items = 2000000
        
        with open(filepath, 'w') as f:
            for item in ds:
                text = item.get('raw_content', '') or item.get('text', '')
                if len(text) < 200:
                    continue
                f.write(json.dumps({
                    'text': text[:10000],
                }) + '\n')
                count += 1
                if count % 200000 == 0:
                    size = os.path.getsize(filepath)
                    log(f"  RedPajama: {count:,} docs — {size/1024/1024/1024:.2f} GB")
                if count >= max_items:
                    break
        
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"✅ RedPajama: {count:,} docs — {size/1024/1024/1024:.2f} GB")
        gc.collect()
    except Exception as e:
        log(f"❌ RedPajama: {str(e)[:100]}")


# ══════════════════════════════════════════════════════════
#  MAIN — DOWNLOAD EVERYTHING
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    log("="*60)
    log("MASSIVE DOWNLOAD STARTING")
    log(f"Target: {DRIVE}")
    log("Go to class. Come back to 1.8TB of training data.")
    log("="*60)
    
    # ── PHASE 1: Small datasets (20-30 min) ───────────────
    log("\n📦 PHASE 1: Small Datasets")
    log("-"*40)
    
    small_datasets = [
        ('CyberNative/Code_Vulnerability_Security_DPO', 'code_vulnerability.jsonl'),
        ('darkknight25/Exploit_Database_Dataset', 'exploit_database.jsonl'),
        ('ChaoticNeutrals/Cybersecurity-ShareGPT', 'cybersec_sharegpt.jsonl'),
        ('ethanolivertroy/nist-cybersecurity-training', 'nist_cybersec.jsonl'),
        ('AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0', 'cybersec_fenrir.jsonl'),
        ('Bouquets/DeepSeek-V3-Distill-Cybersecurity-en', 'deepseek_cybersec.jsonl'),
        ('Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset', 'cybersec_instruct.jsonl'),
        ('atul10/reverse_engineering_code_dataset_O2_x64_O2', 'reverse_eng_x64.jsonl'),
        ('atul10/reverse_engineering_code_dataset_O2_arm_O2', 'reverse_eng_arm.jsonl'),
        ('atul10/reverse_engineering_code_dataset_O2_mips_O2', 'reverse_eng_mips.jsonl'),
        ('bshada/reverseengineering.stackexchange.com', 'reverse_eng_se.jsonl'),
        ('g4lihru/arduino-dataset', 'arduino_dataset.jsonl'),
        ('CJJones/Multiturn_Microcontroller-Arduino-LLM-Training', 'microcontroller_train.jsonl'),
        ('suneeldk/arduino-code-dataset', 'arduino_code.jsonl'),
        ('gavmac00/arduino-docs', 'arduino_docs.jsonl'),
        ('bshada/arduino.stackexchange.com', 'arduino_se.jsonl'),
        ('ewedubs/linux-kernel-commits-aireason-instruct', 'linux_kernel.jsonl'),
        ('fenar/iot-security', 'iot_security.jsonl'),
        ('sjhhg/fpga_verilog', 'fpga_verilog.jsonl'),
        ('AvistoTelecom/dataset_designs_fpga', 'fpga_designs.jsonl'),
        ('vinitvek/cybersecurityattacks', 'cybersec_attacks.jsonl'),
        ('AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1', 'cybersec_heimdall.jsonl'),
        ('maddyrucos/code_vulnerability_java', 'vuln_java.jsonl'),
        ('maddyrucos/code_vulnerability_python', 'vuln_python.jsonl'),
        ('atul10/recreated_reverse_engineering_code_dataset_O2_x86_O2', 'reverse_eng_x86.jsonl'),
    ]
    
    for name, filename in small_datasets:
        download_small(name, filename)
    
    log(f"\n📦 Phase 1 done. Total so far: {total_bytes/1024/1024/1024:.2f} GB")
    
    # ── PHASE 2: The Stack — Code (3-5 hours) ────────────
    log("\n💻 PHASE 2: The Stack (Code)")
    log("-"*40)
    
    code_languages = [
        ('c', None),
        ('c++', 'data/c++'),
        ('python', None),
        ('assembly', None),
        ('shell', None),
        ('makefile', None),
        ('rust', None),
        ('go', None),
        ('java', None),
        ('javascript', None),
        ('verilog', None),
        ('vhdl', None),
    ]
    
    for lang, data_dir in code_languages:
        download_code(lang, data_dir)
    
    log(f"\n💻 Phase 2 done. Total so far: {total_bytes/1024/1024/1024:.2f} GB")
    
    # ── PHASE 3: StarCoder Data (2-3 hours) ───────────────
    log("\n⭐ PHASE 3: StarCoder Data")
    log("-"*40)
    
    download_starcoder()
    
    log(f"\n⭐ Phase 3 done. Total so far: {total_bytes/1024/1024/1024:.2f} GB")
    
    # ── PHASE 4: FineWeb filtered (2-3 hours) ─────────────
    log("\n🌐 PHASE 4: FineWeb (Filtered Internet)")
    log("-"*40)
    
    download_fineweb()
    
    log(f"\n🌐 Phase 4 done. Total so far: {total_bytes/1024/1024/1024:.2f} GB")
    
    # ── PHASE 5: RedPajama (1-2 hours) ────────────────────
    log("\n📚 PHASE 5: RedPajama")
    log("-"*40)
    
    download_redpajama()
    
    # ── FINAL REPORT ──────────────────────────────────────
    elapsed = time.time() - start_time
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    
    log("\n" + "="*60)
    log("🎉 ALL DOWNLOADS COMPLETE!")
    log(f"Total size: {total_bytes/1024/1024/1024:.2f} GB")
    log(f"Time taken: {hrs}h {mins}m")
    log(f"Location: {DRIVE}")
    log("="*60)
    
    # List all files
    log("\nFiles downloaded:")
    for root, dirs, files in os.walk(DRIVE):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            log(f"  {f}: {size/1024/1024/1024:.2f} GB")
