#!/usr/bin/env python3.14
"""Check every deposited cohort list against the criteria it was selected by.

The lists are the record of these selections (drawn once; a new draw would give a different set),
so instead of re-drawing them this script verifies each one and fails on any violation.

  reference DB    refbuild/ref_list_2000.txt: 2,000 distinct X-ray structures, deposited
                  resolution 2.0 A or better (REMARK 2 of the file)
  benchmark       benchmark_manifest.tsv design; every analysed experimental structure inside its
                  method-and-resolution bin (exact_resolution.tsv vs bin_label)
  AI set          120 targets; 10 AlphaFold3 and 10 Boltz2 models each (ai_all_list.txt); the
                  training subset is the top-ranked model of each (ai_list.txt, see cohorts.py)
  era             100 X-ray structures at 2.0-2.5 A (fetch MANIFEST.tsv), deposition years split
                  at 1995 as stated
  validation      no structure used in Fig. 3d is a benchmark entry; methods X-ray or cryo-EM

usage: check_cohorts.py     (exit 1 on any violation)"""
import collections, csv, os, re, sys
import pandas as pd

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = os.path.dirname(A)
bad = []
def report(name, ok, detail):
    print(f"[{'ok' if ok else 'FAIL':4s}] {name}: {detail}")
    if not ok: bad.append(name)

rx = re.compile(r"REMARK\s+2\s+RESOLUTION\.\s+([0-9.]+)\s+ANGSTROM")
def header(p):
    res, meth = None, None
    for ln in open(p, errors="ignore"):
        if ln.startswith("EXPDTA"): meth = ln[10:].strip()
        if ln.startswith("REMARK   2") and res is None:
            m = rx.search(ln)
            if m: res = float(m.group(1))
        if ln.startswith("ATOM"): break
    return res, meth

# reference DB
ref = [l.strip() for l in open(f"{A}/refbuild/ref_list_2000.txt") if l.strip()]
hd = [header(os.path.join(H, p)) for p in ref]
nx = sum(1 for _, m in hd if m and m.startswith("X-RAY")); worst = max(r for r, _ in hd if r is not None)
nores = sum(1 for r, _ in hd if r is None)
report("reference DB", len(ref) == 2000 and len(set(ref)) == 2000 and nx == 2000 and worst <= 2.0 and nores == 0,
       f"{len(ref)} entries ({len(set(ref))} distinct), X-ray {nx}, worst resolution {worst:.2f} A, without resolution {nores}")

# benchmark bins
man = pd.read_csv(f"{H}/benchmark/benchmark_manifest.tsv", sep="\t"); man["k"] = man.id.str.lower()
er = dict(pd.read_csv(f"{A}/exact_resolution.tsv", sep="\t").values)
have = {os.path.splitext(os.path.basename(l.strip()))[0].lower() for l in open(f"{A}/refbuild/benchmark_existing.txt") if l.strip().startswith("pdbs/")}
out = []
for k, lab in zip(man.k, man.bin_label):
    m = re.search(r"([0-9.]+)-([0-9.]+)A", str(lab))
    if k in have and m and k in er:
        lo, hi = float(m.group(1)), float(m.group(2))
        if not (lo <= er[k] <= hi): out.append((k, lab, er[k]))
nb = sum(1 for k, lab in zip(man.k, man.bin_label) if k in have and re.search(r"[0-9.]+-[0-9.]+A", str(lab)) and k in er)
report("benchmark bins", not out, f"{nb} analysed experimental structures with a resolution bin; outside their bin: {len(out)} {out[:5]}")
print(f"       design per bin (manifest): {dict(collections.Counter(man.bin_label))}")

# AI set
ai_all = [l.strip() for l in open(f"{A}/refbuild/ai_all_list.txt") if l.strip()]
tg = collections.Counter((("af3" if "af3" in p else "boltz2"), re.sub(r"_(sample|model_)?\d+\.pdb$", "", os.path.basename(p)).lower()) for p in ai_all)   # Boltz2 names some targets in upper case
per = collections.Counter(tg.values()); targets = {t for _, t in tg}
ai = [l.strip() for l in open(f"{A}/refbuild/ai_list.txt") if l.strip()]
report("AI set", len(targets) == 120 and per == collections.Counter({10: 240}) and len(ai) == 240,
       f"{len(targets)} targets; models per (predictor, target): {dict(per)}; top-model list {len(ai)}")

# era
em = list(csv.DictReader(open(f"{H}/structures/era_before_after_1995/MANIFEST.tsv"), delimiter="\t"))
ey = pd.read_csv(f"{A}/era_before_after_1995/era_years.tsv", sep="\t")
rs = [float(r["resolution"]) for r in em]; xr = sum(1 for r in em if "X-RAY" in r["method"].upper())
pre, post = int((ey.year < 1995).sum()), int((ey.year >= 1995).sum())
report("era", len(em) == 100 and xr == 100 and min(rs) >= 2.0 and max(rs) <= 2.5 and set(ey.id.str.lower()) == {r["id"] for r in em},
       f"{len(em)} structures, X-ray {xr}, resolution {min(rs):.2f}-{max(rs):.2f} A; pre-1995 {pre}, 1995+ {post}")

# validation: nothing used in Fig. 3d is a benchmark entry (status from ms_fig3.py)
vr = pd.read_csv(f"{A}/dms_hero/val_resolution.tsv", sep="\t"); vk = set(vr.id.str.lower())
vs = pd.read_csv(f"{A}/native_pipeline/data/validation_status.tsv", sep="\t")
used = set(vs.loc[vs["Used in Fig 3d"] == "yes", "PDB ID"].str.lower())
shared = vk & set(man.k); meth = collections.Counter(vr.method)
report("validation", not (used & set(man.k)) and not (shared & have) and set(meth) <= {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"},
       f"{len(vr)} entries, {len(used)} used in Fig. 3d; used and in the benchmark design: {len(used & set(man.k))}; "
       f"in the benchmark design at all: {len(shared)} (none analysed, all excluded from Fig. 3d); methods {dict(meth)}")

print(f"\n{len(bad)} cohort(s) failing: {bad}" if bad else "\nall cohorts satisfy their stated criteria")
sys.exit(1 if bad else 0)
