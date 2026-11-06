#!/usr/bin/env python3.14
"""MolProbity-score baseline on the apparent-resolution training structures (Fig. 3 comparison).

Structures: benchmark X-ray and cryo-EM entries with a file (benchmark_existing.txt) and a deposited
resolution (exact_resolution.tsv), i.e. the apparent-resolution training set. Each is scored with
molprobity_baseline/mp_genuine_pipeline.py (reduce + probe clashscore, Top8000 rotamer and
Ramachandran grids, Chen et al. 2010 formula), 8 in parallel.

Writes <out>/molprobity_train_genuine.tsv (scored structures) and <out>/molprobity_failed.tsv
(structures the pipeline could not score, with the error). The baseline comparison uses the scored
structures; the failures are listed, never dropped silently.

usage: molprobity_benchmark.py --out <dir> [--check]
  --check  compare with molprobity_baseline/molprobity_train_genuine.tsv: same structures, every
           value equal within 1e-9 relative."""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = os.path.dirname(A)
sys.path.insert(0, f"{A}/molprobity_baseline"); import mp_genuine_pipeline as MP
COLS = ["clashscore", "n_atoms", "rota_out", "rama_fav", "mpscore", "n_rot", "n_ram", "k", "res", "category", "error"]

def manifest():
    man = pd.read_csv(f"{H}/benchmark/benchmark_manifest.tsv", sep="\t"); man["k"] = man.id.str.lower()
    er = pd.read_csv(f"{A}/exact_resolution.tsv", sep="\t").rename(columns={"id": "k", "exact_resolution": "res"})
    have = [os.path.splitext(os.path.basename(l.strip()))[0].lower() for l in open(f"{A}/refbuild/benchmark_existing.txt") if l.strip().startswith("pdbs/")]
    m = man[man.k.isin(have) & man.category.isin(["xray", "cryoem"])][["k", "category"]].merge(er, on="k", how="inner")
    m["path"] = [f"{H}/benchmark/pdbs/{k}.pdb" for k in m.k]
    return m.sort_values("k").reset_index(drop=True)

def one(r):
    try:
        d = MP.mpscore(r["path"]); d.update(k=r["k"], res=r["res"], category=r["category"], error=None); return d
    except Exception as e:
        return dict(k=r["k"], res=r["res"], category=r["category"], error=f"{type(e).__name__}: {e}"[:120])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    m = manifest(); t0 = time.time()
    with ThreadPoolExecutor(8) as ex: rows = list(ex.map(one, m.to_dict("records")))
    df = pd.DataFrame(rows).reindex(columns=COLS)
    ok, bad = df[df.error.isna()], df[df.error.notna()]
    ok.to_csv(f"{a.out}/molprobity_train_genuine.tsv", sep="\t", index=False)
    bad[["k", "category", "res", "error"]].to_csv(f"{a.out}/molprobity_failed.tsv", sep="\t", index=False)
    print(f"[molprobity] {len(m)} structures ({(m.category == 'xray').sum()} X-ray, {(m.category == 'cryoem').sum()} cryo-EM) "
          f"in {time.time() - t0:.0f} s: scored {len(ok)}, not scored {len(bad)}")
    for _, r in bad.iterrows(): print(f"  not scored: {r.k} ({r.category}) {r.error}")
    if a.check:
        old = pd.read_csv(f"{A}/molprobity_baseline/molprobity_train_genuine.tsv", sep="\t").set_index("k")
        new = ok.set_index("k")
        same_ids = set(old.index) == set(new.index)
        num = ["clashscore", "n_atoms", "rota_out", "rama_fav", "mpscore", "n_rot", "n_ram", "res"]
        j = old[num].join(new[num], rsuffix="_new", how="inner")
        worst = max(((j[c] - j[c + "_new"]).abs() / j[c].abs().clip(lower=1e-12)).max() for c in num)
        cat = (old.loc[j.index, "category"] == new.loc[j.index, "category"]).all()
        print(f"  [check] same structures: {same_ids} (deposited {len(old)}, now {len(new)}); "
              f"max relative difference {worst:.1e}; categories equal: {cat}")
        sys.exit(0 if same_ids and worst < 1e-9 and cat else 1)

if __name__ == "__main__":
    main()
