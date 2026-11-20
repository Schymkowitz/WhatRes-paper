#!/usr/bin/env python3.14
"""WhatRes scores for the de novo binder meta-analysis (Overath et al. 2025; fetch_binder_dataset.py).

Four structure sets per labelled design (binder = chain A in every file):
  design_binder   design model as deposited by the designers, binder chain only
  design_complex  design model, whole complex
  af3_binder      top-ranked AlphaFold3 model (their re-prediction), binder chain only
  af3_complex     top-ranked AlphaFold3 model, whole complex
Structures are written as PDB under short ids (b00001...; map in ids.tsv) into a space-free work folder,
featurised with the WhatRes tool (python -m whatres.features, the public v1.0.0 code) in parallel, and
scored with whatres.apparent_resolution / realism. Every design that cannot be prepared or scored is listed
in dropped.tsv with the reason.

usage: score_binders.py [--data ~/data/binder_meta_2025] [--tool ~/whatres_export/WhatRes] [--jobs 4]"""
import argparse, os, subprocess, sys, glob, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd, gemmi

SETS = ["design_binder", "design_complex", "af3_binder", "af3_complex"]

def prepare(row, data, work):
    """write the four PDB files for one design; return (sid, {set: ok/error})"""
    sid, bid = row["sid"], row["binder_id"]; out = {}
    srcs = {"design": f"{data}/input_pdbs/input_pdbs/{bid}.pdb",
            "af3": (glob.glob(f"{data}/AF3_outputs/AF3_outputs/{bid}/*_model.cif") or [None])[0]}
    for kind, src in srcs.items():
        try:
            if not src or not os.path.exists(src): raise FileNotFoundError(f"{kind} structure missing")
            st = gemmi.read_structure(src); st.setup_entities(); st.remove_ligands_and_waters()
            while len(st) > 1: del st[1]
            full = st.clone(); full.write_pdb(f"{work}/{kind}_complex/{sid}.pdb")
            for ch in [c.name for c in st[0] if c.name != "A"]: st[0].remove_chain(ch)
            if len(st[0]) == 0: raise ValueError("no chain A")
            st.write_pdb(f"{work}/{kind}_binder/{sid}.pdb")
            out[f"{kind}_binder"] = out[f"{kind}_complex"] = "ok"
        except Exception as e:
            out[f"{kind}_binder"] = out[f"{kind}_complex"] = f"{type(e).__name__}: {e}"[:120]
    return sid, out

def featurise(s, work, tool, opt):
    lst = f"{work}/{s}_list.txt"
    ids = sorted(os.listdir(f"{work}/{s}")); open(lst, "w").write("".join(f"{work}/{s}/{f}\n" for f in ids))
    env = dict(os.environ, PYTHONPATH=tool)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "whatres.features", "--bin", f"{tool}/whatres-features/whatres-features", "--list", lst,
                        "--opt", opt, "--root", "/", "--out", f"{work}/{s}_features.tsv", "--workdir", f"{work}/{s}_work",
                        "--scratch", f"{work}/{s}_scratch"], env=env, capture_output=True, text=True)
    open(f"{work}/{s}_features.log", "w").write(r.stdout + r.stderr)
    return s, r.returncode, len(ids), time.time() - t0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.expanduser("~/data/binder_meta_2025"))
    ap.add_argument("--tool", default=os.path.expanduser("~/whatres_export/WhatRes")); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(); work = f"{a.data}/whatres"
    for s in SETS: os.makedirs(f"{work}/{s}", exist_ok=True)
    d = pd.read_csv(f"{a.data}/raw/final_dataset.csv", low_memory=False)
    n0 = len(d); d = d[d.binder.notna()]; n1 = len(d); d = d.drop_duplicates("binder_id", keep=False); n2 = len(d)
    print(f"[binders] {n0} rows -> {n1} with an outcome label -> {n2} after removing duplicated binder_id (all copies)")
    d = d.reset_index(drop=True); d["sid"] = [f"b{i + 1:05d}" for i in range(len(d))]
    d[["sid", "binder_id", "target_id", "source", "binder", "A_length"]].to_csv(f"{work}/ids.tsv", sep="\t", index=False)
    status = {}
    with ProcessPoolExecutor(8) as ex:
        for f in as_completed([ex.submit(prepare, r, a.data, work) for r in d.to_dict("records")]):
            sid, st = f.result(); status[sid] = st
    drop = [(sid, s, e) for sid, st in status.items() for s, e in st.items() if e != "ok"]
    print(f"[prepare] {len(d)} designs x 4 sets; not prepared: {len(drop)}")
    opt = f"{work}/Options.txt"; s = open(f"{a.tool}/data/Options_features.txt").read()
    open(opt, "w").write("\n".join(f"<qc_reference_db>{a.tool}/data/native49_structure_reference.bin;" if l.startswith("<qc_reference_db>") else l for l in s.split("\n")))
    with ProcessPoolExecutor(a.jobs) as ex:
        for f in as_completed([ex.submit(featurise, s, work, a.tool, opt) for s in SETS]):
            s_, rc, n, dt = f.result(); print(f"[features] {s_}: {n} structures, exit {rc}, {dt / 60:.0f} min", flush=True)
    sys.path.insert(0, a.tool); import whatres
    rows = []
    for s in SETS:
        fp = f"{work}/{s}_features.tsv"
        if not os.path.exists(fp): drop += [(sid, s, "feature build failed") for sid in d.sid]; continue
        fm = pd.read_csv(fp, sep="\t"); fm["sid"] = fm["id"].astype(str).str.replace(r"\.pdb$", "", regex=True)
        fm["apparent_resolution"] = whatres.apparent_resolution(fm); fm["realism"] = whatres.realism_score(whatres.realism_load(fm))
        fm["set"] = s; rows.append(fm[["sid", "set", "apparent_resolution", "realism", "qc_n_residues"]])
        miss = sorted(set(sid for sid, st in status.items() if st[s] == "ok") - set(fm.sid))
        drop += [(sid, s, "no feature row") for sid in miss]
    sc = pd.concat(rows).merge(d[["sid", "binder_id", "target_id", "source", "binder", "A_length"]], on="sid")
    sc.to_csv(f"{work}/scores.tsv", sep="\t", index=False)
    pd.DataFrame(drop, columns=["sid", "set", "reason"]).to_csv(f"{work}/dropped.tsv", sep="\t", index=False)
    print(f"[scores] {len(sc)} rows -> {work}/scores.tsv; dropped {len(drop)} (dropped.tsv)")
    print(sc.groupby("set")[["apparent_resolution", "realism"]].median().round(2).to_string())

if __name__ == "__main__":
    main()
