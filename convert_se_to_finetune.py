import json, os

INPUT  = os.path.expanduser("~/new_data/stackexchange_qa.jsonl")
OUTPUT = os.path.expanduser("~/new_data/se_finetune.jsonl")

total = 0
with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        q = d.get("question", "") or d.get("body", "")
        a = d.get("answer", "")
        title = d.get("title", "")
        if not q or not a: continue
        if len(a.strip()) < 50: continue
        instruction = f"{title}\n\n{q}".strip() if title else q.strip()
        record = {
            "instruction": instruction[:1000],
            "response": a.strip()[:2000]
        }
        fout.write(json.dumps(record) + "\n")
        total += 1

print(f"Total examples: {total:,}")
print(f"Output: {OUTPUT}")
