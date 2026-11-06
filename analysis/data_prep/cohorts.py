#!/usr/bin/env python3.14
"""Derived cohort tables, from the structures and the deposited cohort definitions.

  benchmark_existing.txt   benchmark/list_all_benchmark.txt entries whose file is on disk
                           (the analysed benchmark; entries without a file are logged, by category)
  exact_resolution.tsv     deposited resolution of every benchmark experimental structure
                           (REMARK 2 of benchmark/pdbs/*.pdb)
  ai_topmodel_meta.tsv     top-ranked AlphaFold3 sample (ranking_score) and Boltz2 model
  ai_list.txt              (confidence_score) per target, from the predictors' confidence summaries
  era_years.tsv            deposition year of the 100 era structures (HEADER record)

usage: cohorts.py --out <dir> [--check]
  --check  compare every table with the one in use (content, order-independent where the
           consumers merge by id) and exit non-zero on any difference.
Inputs: benchmark/predictor_confidence/ (the AlphaFold3 and Boltz2 confidence summaries) and the
model files in benchmark/models_af3/pdb and benchmark/models_boltz2/pdb."""
import argparse, csv, glob, hashlib, os, re, sys, collections

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = os.path.dirname(A); B = os.path.join(H, "benchmark")
CONF = f"{B}/predictor_confidence"   # af3_confidence.tsv, boltz2_confidence.tsv (the predictors' own summaries)
CUR = {"benchmark_existing.txt": f"{A}/refbuild/benchmark_existing.txt", "exact_resolution.tsv": f"{A}/exact_resolution.tsv",
       "ai_topmodel_meta.tsv": f"{A}/ai_topmodel_meta.tsv", "ai_list.txt": f"{A}/refbuild/ai_list.txt",
       "era_years.tsv": f"{A}/era_before_after_1995/era_years.tsv"}

def coords(p):
    return hashlib.md5("".join(l[30:54] for l in open(p) if l.startswith("ATOM")).encode()).hexdigest()

def benchmark(out):
    lst = [l.strip() for l in open(f"{B}/list_all_benchmark.txt") if l.strip()]
    keep = [p for p in lst if os.path.exists(os.path.join(B, p))]
    open(f"{out}/benchmark_existing.txt", "w").write("\n".join(keep) + "\n")
    man = {r["id"].upper(): r["category"] for r in csv.DictReader(open(f"{B}/benchmark_manifest.tsv"), delimiter="\t")}
    miss = sorted(set(lst) - set(keep))
    by = collections.Counter(man.get(os.path.splitext(os.path.basename(p))[0].upper(), "?") for p in miss)
    print(f"[benchmark] {len(lst)} list entries ({len(set(lst))} unique) -> {len(keep)} with a file "
          f"({len(set(keep))} unique); without a file: {len(miss)} {dict(by)}")

def exact_resolution(out):
    rx = re.compile(r"REMARK\s+2\s+RESOLUTION\.\s+([0-9.]+)\s+ANGSTROM"); rows = []; n = 0
    for p in sorted(glob.glob(f"{B}/pdbs/*.pdb")):
        n += 1; res = None
        for ln in open(p, errors="ignore"):
            if ln.startswith("REMARK   2"):
                m = rx.search(ln)
                if m:
                    try: res = float(m.group(1))
                    except ValueError: pass
                    break
            if ln.startswith("ATOM"): break
        if res is not None and res > 0: rows.append((os.path.splitext(os.path.basename(p))[0].lower(), res))
    open(f"{out}/exact_resolution.tsv", "w").write("id\texact_resolution\n" + "\n".join(f"{i}\t{r}" for i, r in rows) + "\n")
    print(f"[exact_resolution] {n} benchmark/pdbs files -> {len(rows)} with a REMARK 2 resolution ({n - len(rows)} without: NMR or none)")

def ai(out):
    meta, lst, bad = [], [], []
    best = {}
    for r in csv.DictReader(open(f"{CONF}/af3_confidence.tsv"), delimiter="\t"):
        n, s = r["name"].lower(), float(r["ranking_score"])
        if n not in best or s > best[n][0]: best[n] = (s, r["sample"], r.get("has_clash"), r.get("ptm"), r.get("fraction_disordered"))
    for n, (s, samp, clash, ptm, fdis) in best.items():
        src = f"{B}/models_af3/pdb/{n}_sample{samp}.pdb"
        if os.path.exists(src):
            meta.append((n, "af3", round(s, 3), clash, ptm, fdis)); lst.append(f"ai/af3_{n}.pdb")
            if coords(src) != coords(f"{B}/ai/af3_{n}.pdb"): bad.append(f"af3_{n}")
    best = {}
    for r in csv.DictReader(open(f"{CONF}/boltz2_confidence.tsv"), delimiter="\t"):
        n, s = r["name"].lower(), float(r["confidence_score"])
        if n not in best or s > best[n][0]: best[n] = (s, r["model"], r.get("complex_plddt"), r.get("ptm"))
    for n, (s, mdl, cpl, ptm) in best.items():
        src = f"{B}/models_boltz2/pdb/{n}_model_{mdl}.pdb"
        if os.path.exists(src):
            meta.append((n, "boltz2", round(s, 3), "", ptm, cpl)); lst.append(f"ai/boltz2_{n}.pdb")
            if coords(src) != coords(f"{B}/ai/boltz2_{n}.pdb"): bad.append(f"boltz2_{n}")
    with open(f"{out}/ai_topmodel_meta.tsv", "w") as f:
        f.write("target\tpredictor\tconfidence\taf3_has_clash\tptm\textra\n")
        for m in meta: f.write("\t".join(str(x) for x in m) + "\n")
    open(f"{out}/ai_list.txt", "w").write("\n".join(sorted(lst)) + "\n")
    print(f"[ai] top models: {sum(m[1] == 'af3' for m in meta)} AlphaFold3, {sum(m[1] == 'boltz2' for m in meta)} Boltz2; "
          f"coordinates differing from benchmark/ai: {len(bad)} {bad[:5]}")
    if bad: sys.exit("STOP: selected models do not match benchmark/ai")

def era(out):
    E = f"{H}/structures/era_before_after_1995"
    ids = [l.split("\t")[0] for l in open(f"{E}/MANIFEST.tsv").read().split("\n")[1:] if l.strip()]
    rows = []
    for pid in ids:
        yr = None
        for ln in open(f"{E}/{pid}.pdb", errors="ignore"):
            if ln.startswith("HEADER"):
                m = re.search(r"(\d{2})-([A-Z]{3})-(\d{2})", ln)
                if m: yy = int(m.group(3)); yr = 1900 + yy if yy >= 70 else 2000 + yy
                break
        rows.append((pid, yr))
    open(f"{out}/era_years.tsv", "w").write("id\tyear\n" + "\n".join(f"{i}\t{y}" for i, y in rows) + "\n")
    print(f"[era] {len(rows)} structures, years {min(y for _, y in rows)}-{max(y for _, y in rows)}; without a HEADER date: {sum(y is None for _, y in rows)}")

def table(p):
    lines = [l for l in open(p).read().split("\n") if l.strip()]
    return lines[0], sorted(lines[1:]) if "\t" in lines[0] else lines

def check(out):
    bad = 0
    for name, cur in CUR.items():
        new = f"{out}/{name}"
        if name.endswith(".txt"):
            same = open(new).read() == open(cur).read(); how = "byte-identical"
        else:
            same = table(new) == table(cur); how = "same rows (order-independent)"
        print(f"  [check] {name:24s} {'OK, ' + how if same else 'DIFFERS from ' + os.path.relpath(cur, H)}")
        bad += not same
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    benchmark(a.out); exact_resolution(a.out); ai(a.out); era(a.out)
    if a.check: check(a.out)
