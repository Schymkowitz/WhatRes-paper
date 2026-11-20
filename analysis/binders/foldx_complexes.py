#!/usr/bin/env python3.14
"""FoldX AnalyseComplex on the raw (unrepaired) binder complexes: binder (chain A) against the target chains.

Sets: af3_complex and design_complex, the PDB files written by score_binders.py. Every structure runs in its own
temporary folder under the data folder (never in a code repository), with rotabase.txt next to it. Output per set:
foldx_<set>.tsv (sid, interaction energy and its terms) and foldx_failed.tsv (sid, set, reason).

usage: foldx_complexes.py [--data ~/data/binder_meta_2025] [--foldx <HighResPDB>/bin/foldx] [--jobs 8]"""
import argparse, json, os, shutil, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

H = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # HighResPDB

def run(sid, s, chains, data, foldx):
    src = f"{data}/whatres/{s}/{sid}.pdb"
    with tempfile.TemporaryDirectory(dir=f"{data}/foldx/tmp") as td:
        shutil.copy(src, f"{td}/{sid}.pdb"); shutil.copy(f"{H}/rotabase.txt", td)
        r = subprocess.run([foldx, "--command=AnalyseComplex", f"--pdb={sid}.pdb", f"--analyseComplexChains=A,{chains}"],
                           cwd=td, capture_output=True, text=True, timeout=900)
        out = f"{td}/Interaction_{sid}_AC.fxout"
        if not os.path.exists(out): return sid, s, None, f"no output (exit {r.returncode}): {(r.stdout + r.stderr)[-150:]}"
        lines = [l.rstrip("\n").split("\t") for l in open(out) if l.strip()]
        hdr = next(l for l in lines if l[0] == "Pdb"); row = lines[lines.index(hdr) + 1]
        return sid, s, dict(zip(hdr, row)), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.expanduser("~/data/binder_meta_2025"))
    ap.add_argument("--foldx", default=f"{H}/bin/foldx"); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    os.makedirs(f"{a.data}/foldx/tmp", exist_ok=True)
    ids = pd.read_csv(f"{a.data}/whatres/ids.tsv", sep="\t")
    tc = pd.read_csv(f"{a.data}/raw/final_dataset.csv", low_memory=False)[["binder_id", "target_chains"]].drop_duplicates("binder_id")
    m = ids.merge(tc, on="binder_id", how="left")
    m["chains"] = m.target_chains.map(lambda x: "".join(json.loads(x)))
    jobs = [(r.sid, s, r.chains) for r in m.itertuples() for s in ("af3_complex", "design_complex")]
    res, fail = {"af3_complex": [], "design_complex": []}, []
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run, sid, s, ch, a.data, a.foldx) for sid, s, ch in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            try: sid, s, row, err = f.result()
            except Exception as e: fail.append(("?", "?", str(e)[:150])); continue
            if err: fail.append((sid, s, err))
            else: row["sid"] = sid; res[s].append(row)
            if k % 1000 == 0: print(f"[foldx] {k}/{len(jobs)}", flush=True)
    for s, rows in res.items():
        df = pd.DataFrame(rows); num = [c for c in df.columns if c not in ("Pdb", "Group1", "Group2", "sid")]
        df[num] = df[num].apply(pd.to_numeric, errors="coerce"); df.to_csv(f"{a.data}/foldx/foldx_{s}.tsv", sep="\t", index=False)
        print(f"[foldx] {s}: {len(df)} complexes")
    pd.DataFrame(fail, columns=["sid", "set", "reason"]).to_csv(f"{a.data}/foldx/foldx_failed.tsv", sep="\t", index=False)
    print(f"[foldx] failed: {len(fail)} (foldx_failed.tsv)")

if __name__ == "__main__":
    main()
