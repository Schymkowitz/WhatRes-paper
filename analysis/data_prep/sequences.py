#!/usr/bin/env python3.14
"""Sequence clusters for grouped cross-validation (MMseqs2, 30% identity, 80% coverage).

Sequences, one per structure (the longest chain):
  training structures   benchmark experimental structures with a deposited resolution
                        (exact_resolution.tsv): amino acids modelled in the coordinates, modified
                        residues as their parent (gemmi one-letter code), UNK as X
  validation structures dms_hero/val_ids.tsv: RCSB FASTA of the entry; raw downloads are cached in
                        seq_clusters/rcsb_fasta/ so the input is fixed once fetched
  structures whose longest chain has fewer than 10 residues are left out (logged)
Then: mmseqs easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0.

The deposited seq_clusters/all_sequences.fasta (2026-10-26) is the clustering input. RCSB's entity
order is not stable over time, so for an entry whose longest chains tie in length, a fresh download
can pick a different chain: the extraction below is therefore a CHECK of the deposited FASTA (every
difference is listed), and the clusters are computed from the deposited FASTA.

usage: sequences.py --out <dir> [--check]     ($MMSEQS = mmseqs binary)
  --check  exit non-zero unless the clusters reproduce exactly and every FASTA difference is a
           tie between equally long chains."""
import argparse, glob, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import gemmi

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = os.path.dirname(A)
SC = f"{A}/seq_clusters"; CACHE = f"{SC}/rcsb_fasta"
MMSEQS = os.environ.get("MMSEQS", "mmseqs")   # on PATH unless set

def clean(s): return "".join(c for c in s.upper() if c.isalpha())

def coord_seq(p):
    best = ""
    for ch in gemmi.read_structure(p)[0]:
        s = ""
        for r in ch:
            t = gemmi.find_tabulated_residue(r.name)
            if t and t.is_amino_acid(): s += t.one_letter_code
        s = clean(s)
        if len(s) > len(best): best = s
    return best

def rcsb_seq(pid):
    f = f"{CACHE}/{pid}.fasta"
    if not os.path.exists(f):
        r = subprocess.run(["curl", "-sf", "--retry", "3", f"https://www.rcsb.org/fasta/entry/{pid}"], capture_output=True, text=True)
        if r.returncode != 0: return None
        open(f, "w").write(r.stdout)
    seqs, cur = [], ""
    for ln in open(f):
        if ln.startswith(">"):
            if cur: seqs.append(cur); cur = ""
        else: cur += ln.strip()
    if cur: seqs.append(cur)
    return clean(max(seqs, key=len)) if seqs else ""   # ties: first in the file

def read_fasta(p):
    d, cur = {}, None
    for ln in open(p):
        ln = ln.strip()
        if ln.startswith(">"): cur = ln[1:]; d[cur] = ""
        elif cur: d[cur] += ln
    return d

def partition(p):
    g = {}
    for ln in open(p):
        if ln.strip(): rep, mem = ln.split(); g.setdefault(rep, set()).add(mem)
    return {frozenset(v) for v in g.values()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    tr_ids = sorted(l.split("\t")[0].upper() for l in open(f"{A}/exact_resolution.tsv").read().split("\n")[1:] if l)
    train, short = {}, []
    for pid in tr_ids:
        s = coord_seq(f"{H}/benchmark/pdbs/{pid.lower()}.pdb")
        (train.__setitem__(pid, s) if len(s) >= 10 else short.append(pid))
    va_ids = sorted({l.split("\t")[0].upper() for l in open(f"{A}/dms_hero/val_ids.tsv") if l.strip()} - set(train))
    with ThreadPoolExecutor(8) as ex: vs = dict(zip(va_ids, ex.map(rcsb_seq, va_ids)))
    failed = [k for k, v in vs.items() if v is None]
    if failed: sys.exit(f"STOP: RCSB FASTA not available for {failed[:10]}")
    val = {k: v for k, v in vs.items() if len(v) >= 10}; vshort = sorted(set(vs) - set(val))
    print(f"[sequences] training {len(tr_ids)} -> {len(train)} (longest chain < 10 residues: {len(short)} {short[:6]}...)")
    print(f"[sequences] validation {len(va_ids)} -> {len(val)} (< 10 residues: {len(vshort)} {vshort})")
    fa = f"{a.out}/all_sequences_fresh.fasta"
    with open(fa, "w") as f:
        for d in (train, val):
            for k, v in d.items(): f.write(f">{k}\n{v}\n")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([MMSEQS, "easy-cluster", f"{SC}/all_sequences.fasta", f"{td}/clust", f"{td}/tmp", "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(f"{td}/clust_cluster.tsv", f"{a.out}/mmseqs_clusters_id30_cov80.tsv")
    P = partition(f"{a.out}/mmseqs_clusters_id30_cov80.tsv")
    print(f"[clusters] {sum(len(x) for x in P)} sequences in {len(P)} clusters")
    old, new = read_fasta(f"{SC}/all_sequences.fasta"), read_fasta(fa)
    diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    tie = [k for k in diff if k in old and k in new and len(old[k]) == len(new[k])]
    print(f"[check] fresh extraction vs deposited FASTA: {len(old) - len(diff)} of {len(old)} identical; "
          f"different: {len(diff)} {diff} (equal-length tie: {tie})")
    lines = lambda p: sorted(l for l in open(p).read().split("\n") if l)   # MMseqs2 writes in thread order
    b_ok = lines(f"{a.out}/mmseqs_clusters_id30_cov80.tsv") == lines(f"{SC}/mmseqs_clusters_id30_cov80.tsv")
    print(f"[check] clusters from the deposited FASTA: {'identical (every member, same representative; line order aside)' if b_ok else 'DIFFER'}")
    if a.check: sys.exit(0 if b_ok and set(diff) == set(tie) else 1)

if __name__ == "__main__":
    main()
