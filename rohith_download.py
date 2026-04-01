#!/usr/bin/env python3
"""
ROHITH'S DOWNLOAD — CODE DATASETS ONLY (1.8TB)
The Stack + StarCoder = every programming language

Run and go to class. Come back to 1.8TB of code.
"""

from datasets import load_dataset
import json, os, gc, time

DRIVE = os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-krohithsai99@gmail.com/My Drive/llm-data"
)

os.makedirs(f"{DRIVE}/code", exist_ok=True)

total_bytes = 0
start_time = time.time()

def log(msg):
    elapsed = time.time() - start_time
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print(f"[{hrs}h{mins:02d}m] {msg}", flush=True)


def download_code(lang, data_dir=None, max_files=5000000):
    global total_bytes
    safe_name = lang.replace('+', 'plus')
    filename = f"the_stack_{safe_name}.jsonl"
    filepath = f"{DRIVE}/code/{filename}"
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"⏭️  Exists: {filename} ({size/1024/1024/1024:.2f} GB)")
        return
    
    actual_dir = data_dir if data_dir else f"data/{lang}"
    
    try:
        log(f"⬇️  The Stack — {lang}...")
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
                    log(f"  {lang}: {count:,} files — {size/1024/1024/1024:.2f} GB (total: {(total_bytes+size)/1024/1024/1024:.1f} GB)")
                if count >= max_files:
                    break
        
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"✅ {lang}: {count:,} files — {size/1024/1024/1024:.2f} GB")
        gc.collect()
    except Exception as e:
        log(f"❌ The Stack {lang}: {str(e)[:100]}")


def download_starcoder(lang, max_files=3000000):
    global total_bytes
    filename = f"starcoder_{lang}.jsonl"
    filepath = f"{DRIVE}/code/{filename}"
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"⏭️  Exists: {filename} ({size/1024/1024/1024:.2f} GB)")
        return
    
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
                if count >= max_files:
                    break
        
        size = os.path.getsize(filepath)
        total_bytes += size
        log(f"✅ StarCoder/{lang}: {count:,} files — {size/1024/1024/1024:.2f} GB")
        gc.collect()
    except Exception as e:
        log(f"❌ StarCoder/{lang}: {str(e)[:100]}")


if __name__ == '__main__':
    log("="*60)
    log("ROHITH'S CODE DOWNLOAD — 1.8TB")
    log(f"Target: {DRIVE}/code")
    log("Go to class. Come back to all the code in the world.")
    log("="*60)
    
    # ── PHASE 1: The Stack (biggest code dataset) ─────────
    log("\n💻 PHASE 1: The Stack — All Languages")
    log("-"*40)
    
    stack_languages = [
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
        ('typescript', None),
        ('ruby', None),
        ('php', None),
        ('swift', None),
        ('kotlin', None),
        ('scala', None),
        ('r', None),
        ('lua', None),
        ('perl', None),
        ('haskell', None),
        ('julia', None),
        ('dart', None),
        ('cmake', None),
    ]
    
    for lang, data_dir in stack_languages:
        download_code(lang, data_dir)
    
    log(f"\n💻 Phase 1 done. Total: {total_bytes/1024/1024/1024:.2f} GB")
    
    # ── PHASE 2: StarCoder ────────────────────────────────
    log("\n⭐ PHASE 2: StarCoder — All Languages")
    log("-"*40)
    
    starcoder_langs = [
        'python', 'c', 'cpp', 'javascript', 'java',
        'rust', 'go', 'shell', 'assembly', 'typescript',
        'ruby', 'php', 'swift', 'kotlin', 'scala',
        'r', 'lua', 'perl', 'haskell', 'julia',
    ]
    
    for lang in starcoder_langs:
        download_starcoder(lang)
    
    # ── FINAL REPORT ──────────────────────────────────────
    elapsed = time.time() - start_time
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    
    log("\n" + "="*60)
    log("🎉 ROHITH'S DOWNLOAD COMPLETE!")
    log(f"Total: {total_bytes/1024/1024/1024:.2f} GB")
    log(f"Time: {hrs}h {mins}m")
    log(f"Location: {DRIVE}/code")
    log("="*60)
    
    log("\nFiles:")
    code_dir = f"{DRIVE}/code"
    for f in sorted(os.listdir(code_dir)):
        size = os.path.getsize(os.path.join(code_dir, f))
        log(f"  {f}: {size/1024/1024/1024:.2f} GB")
