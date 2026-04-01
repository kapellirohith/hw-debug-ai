import json, os

INPUT  = "/Users/rohithkapelli/new_data/MASTER_FINETUNE.jsonl"
OUTPUT = "/Users/rohithkapelli/new_data/MASTER_FINETUNE_CLEAN.jsonl"

REMOVE_IF_CONTAINS = [
    "reconnaissance","penetration test","metasploit","sql injection",
    "xss attack","csrf","privilege escalation","jboss","tomcat hardening",
    "nmap scan","exploit framework","payload delivery","403 forbidden",
    "pentest","red team","blue team","ctf challenge","capture the flag",
    "buffer overflow exploit","shellcode","reverse shell","bind shell",
    "lateral movement","credential dumping","mimikatz","cobalt strike",
]

total_in = 0
total_removed = 0
total_out = 0

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        total_in += 1
        d = json.loads(line)
        inst = (d.get('instruction','') or '').lower()
        resp = (d.get('response','') or '').lower()
        text = inst + ' ' + resp
        if any(kw in text for kw in REMOVE_IF_CONTAINS):
            total_removed += 1
            continue
        fout.write(json.dumps(d) + "\n")
        total_out += 1

print(f"Input:   {total_in:,}")
print(f"Removed: {total_removed:,}")
print(f"Output:  {total_out:,}")
print(f"Saved:   {OUTPUT}")
