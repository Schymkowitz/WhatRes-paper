#!/usr/bin/env python3.14
"""Independent validation set (Fig. 3d): deposited resolution and structure files.

Input: dms_hero/val_ids.tsv (the 941 selected entries and their method; the selection record).
  val_resolution.tsv   deposited resolution per entry (RCSB GraphQL, resolution_combined)
  val_pdbs/<ID>.pdb    PDB-format file per entry (RCSB); fetched only when absent. Entries that
                       RCSB serves only as mmCIF have no PDB-format file and get no feature row (logged)
  val_list.txt         the files present, for the feature build

usage: validation_set.py --out <dir> [--check]
  --check  compare val_resolution.tsv (rows) and val_list.txt (entries) with the files in use."""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); V = f"{A}/dms_hero"

def graphql(ids):
    q = '{entries(entry_ids:%s){rcsb_id rcsb_entry_info{resolution_combined}}}' % json.dumps(ids)
    r = subprocess.run(["curl", "-sf", "--retry", "3", "-X", "POST", "https://data.rcsb.org/graphql", "-H", "Content-Type: application/json",
                        "-d", json.dumps({"query": q})], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["data"]["entries"]

def fetch(pid):
    f = f"{V}/val_pdbs/{pid}.pdb"
    if os.path.exists(f) and os.path.getsize(f) > 1000: return True
    r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", f, f"https://files.rcsb.org/download/{pid}.pdb"])
    if r.returncode != 0 or not os.path.exists(f) or os.path.getsize(f) <= 1000:
        if os.path.exists(f): os.remove(f)
        return False
    return True

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(f"{V}/val_pdbs", exist_ok=True)
    rows = [l.rstrip("\n").split("\t") for l in open(f"{V}/val_ids.tsv") if l.strip()]
    meth = {i: m for i, m in rows}; ids = [i for i, _ in rows]
    res = {}
    for k in range(0, len(ids), 100):
        for e in graphql(ids[k:k + 100]) or []:
            if e:
                r = e["rcsb_entry_info"].get("resolution_combined"); res[e["rcsb_id"]] = r[0] if r else None
    with open(f"{a.out}/val_resolution.tsv", "w") as f:
        f.write("id\texact_resolution\tmethod\n" + "".join(f"{i}\t{res[i]}\t{meth[i]}\n" for i in ids if i in res))
    with ThreadPoolExecutor(16) as ex: ok = dict(zip(ids, ex.map(fetch, ids)))
    have = [i for i in ids if ok[i]]
    open(f"{a.out}/val_list.txt", "w").write("".join(f"dms_hero/val_pdbs/{i}.pdb\n" for i in sorted(have)))
    print(f"[validation] {len(ids)} entries: resolution for {len(res)}; PDB-format file for {len(have)}; "
          f"without one (mmCIF only): {len(ids) - len(have)}")
    if a.check:
        rd = lambda p: sorted(l for l in open(p).read().split("\n")[1:] if l)
        r_ok = rd(f"{a.out}/val_resolution.tsv") == rd(f"{V}/val_resolution.tsv")
        st = lambda p: {l.strip() for l in open(p) if l.strip()}
        l_ok = st(f"{a.out}/val_list.txt") == st(f"{V}/val_list.txt")
        print(f"  [check] val_resolution.tsv {'identical rows' if r_ok else 'DIFFERS'}; val_list.txt {'same entries' if l_ok else 'DIFFERS'}")
        sys.exit(0 if r_ok and l_ok else 1)

if __name__ == "__main__":
    main()
