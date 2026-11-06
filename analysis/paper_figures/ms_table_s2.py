#!/usr/bin/env python3.14
"""Supplementary Table 2 (datasets), generated from the cohort files; nothing is copied from an
earlier version of the table.

Sheets and sources
  Summary                one row per dataset; every count from the sheets below
  Benchmark              benchmark_existing.txt (each file once) x benchmark_manifest.tsv x exact_resolution.tsv
  AI_prediction_targets  prediction_set/prediction_manifest.tsv
  Reference_cohort       refbuild/ref_list_2000.txt; resolution from REMARK 2; note if absent from the survey
  Era_cohort             era_years.tsv (deposition year; 'pre-1995' / '1995+')
  Over_regularisation    over_regularisation_geometry/geom_structures.tsv (structures actually read)
  Benchmark_IDs          as Benchmark, one row per structure
  Validation_IDs         native_pipeline/data/validation_status.tsv (written by ms_fig3.py)

usage: ms_table_s2.py [--compare <existing xlsx>]   (compare: per-sheet row differences, as sets)"""
import argparse, os, re, sys
import pandas as pd
from ms_common import AN, DATA

H = os.path.dirname(AN); T = os.path.join(AN, "native_pipeline", "tables"); os.makedirs(T, exist_ok=True)
OUT = os.path.join(T, "WhatRes Suppl Table 2 datasets.xlsx")
rx = re.compile(r"REMARK\s+2\s+RESOLUTION\.\s+([0-9.]+)\s+ANGSTROM")

def remark2(p):
    for ln in open(p, errors="ignore"):
        if ln.startswith("REMARK   2"):
            m = rx.search(ln)
            if m: return float(m.group(1))       # the first REMARK 2 line is blank
        if ln.startswith("ATOM"): return None
    return None

def build():
    er = pd.read_csv(f"{AN}/exact_resolution.tsv", sep="\t"); RES = dict(zip(er.id.str.upper(), er.exact_resolution))
    man = pd.read_csv(f"{H}/benchmark/benchmark_manifest.tsv", sep="\t"); CAT = dict(zip(man.id.str.upper(), man.category))
    # ---- benchmark: each file once ----
    files = list(dict.fromkeys(l.strip() for l in open(f"{AN}/refbuild/benchmark_existing.txt") if l.strip()))
    ids = []
    for p in files:
        k = os.path.splitext(os.path.basename(p))[0].upper()
        cat = "AlphaFold2" if p.startswith("alphafold/") else CAT[k]
        ids.append((k, cat, RES.get(k)))
    BID = pd.DataFrame(ids, columns=["PDB/model ID", "Category", "Resolution (Å)"]).sort_values("PDB/model ID").reset_index(drop=True)
    lab = {"xray": "X-ray", "cryoem": "cryo-EM", "nmr": "NMR", "AlphaFold2": "AlphaFold2"}
    rows = []
    for c in ["xray", "cryoem", "nmr", "AlphaFold2"]:
        s = BID[BID.Category == c]; r = s["Resolution (Å)"].dropna()
        rows.append([lab[c], len(s), len(r), f"{r.min():.2f} to {r.max():.2f}" if len(r) else "n/a (no deposited resolution)"])
    BEN = pd.DataFrame(rows, columns=["Method", "n structures", "n with deposited resolution", "Resolution range (A)"])
    rall = BID["Resolution (Å)"].dropna()
    # ---- AI prediction targets ----
    pm = pd.read_csv(f"{AN}/prediction_set/prediction_manifest.tsv", sep="\t").drop_duplicates("target_id")
    AIT = pm[["target_id", "uniprot", "seq_len", "temporal_tier", "homology_tier", "has_expt", "expt_pdb", "expt_method", "expt_resolution"]].rename(columns={
        "target_id": "Target", "uniprot": "UniProt", "seq_len": "Length", "temporal_tier": "Temporal tier", "homology_tier": "Homology tier",
        "has_expt": "Has experimental", "expt_pdb": "Experimental PDB", "expt_method": "Experimental method", "expt_resolution": "Experimental resolution (A)"})
    n_ai = sum(1 for l in open(f"{AN}/refbuild/ai_all_list.txt") if l.strip())
    # ---- reference cohort ----
    ref = [l.strip() for l in open(f"{AN}/refbuild/ref_list_2000.txt") if l.strip()]
    REF = pd.DataFrame([(os.path.splitext(os.path.basename(p))[0].upper(), remark2(os.path.join(H, p))) for p in ref], columns=["PDB ID", "Resolution (Å)"])
    summ = open(f"{AN}/refbuild/native49_structure_survey_summary.txt").readline()
    npdb, nres = map(int, re.search(r"(\d+) PDBs, (\d+) residues", summ).groups())
    if npdb == len(ref): REF["Note"] = None                      # every listed structure is in the reference DB
    else:                                                          # name the ones the survey lacks (needs the per-residue survey table)
        surveyed = set(pd.read_csv(f"{AN}/refbuild/native49_structure_survey.tsv", sep="\t", usecols=["source_pdb"]).source_pdb.astype(str)
                       .str.replace(r"\.pdb$", "", regex=True).str.upper())
        REF["Note"] = [None if i in surveyed else "not in the reference survey" for i in REF["PDB ID"]]
    REF = REF.sort_values("PDB ID").reset_index(drop=True)
    # ---- era ----
    ey = pd.read_csv(f"{AN}/era_before_after_1995/era_years.tsv", sep="\t")
    ERA = pd.DataFrame({"PDB ID": ey.id.str.upper(), "Deposition year": ey.year,
                        "Era": ["pre-1995" if y < 1995 else "1995+" for y in ey.year], "Deposited resolution window": "2.0-2.5 Å"})
    ERA = ERA.sort_values(["Deposition year", "PDB ID"]).reset_index(drop=True)
    # ---- over-regularisation ----
    gs = pd.read_csv(f"{AN}/over_regularisation_geometry/geom_structures.tsv", sep="\t")
    CL = {"<1A X-ray": "X-ray <1.0 Å (ultra-high-res reference)", ">2.5A X-ray": "X-ray >2.5 Å", "cryo-EM": "cryo-EM 2.0-3.0 Å",
          "AF3": "AlphaFold3 model (120 prediction targets)", "Boltz2": "Boltz2 model (120 prediction targets)"}
    gid = gs.id.str.replace(r"^(af3|boltz2)_", "", regex=True).str.upper()
    OVR = pd.DataFrame({"PDB/target ID": gid, "Cohort": gs.cohort.map(CL),
                        "Resolution (Å)": [None if c in ("AF3", "Boltz2") else RES.get(i) for i, c in zip(gid, gs.cohort)]})
    nov = gs.cohort.value_counts()
    # ---- validation ----
    VAL = pd.read_csv(f"{DATA}/validation_status.tsv", sep="\t")
    VAL.loc[VAL["Status / exclusion reason"].str.contains("singleton"), "MMseqs2 cluster"] = "singleton"
    used = VAL[VAL["Used in Fig 3d"] == "yes"]
    ux = used.Method.str.contains("X-RAY", case=False).sum(); ue = len(used) - ux
    st = VAL["Status / exclusion reason"]
    n_nofeat, n_shared = int((st == "excluded: no feature row").sum()), int((st == "excluded: MMseqs2 cluster shared with training").sum())
    n_other, n_single = int((st == "excluded: method or resolution").sum()), int(st.str.contains("singleton").sum())
    # ---- summary ----
    b = BEN.set_index("Method")["n structures"]
    pre, post = int((ERA.Era == "pre-1995").sum()), int((ERA.Era == "1995+").sum())
    SUM = pd.DataFrame([
        ["High-resolution reference cohort", "Per-residue reference distributions (percentile and surprise scoring)",
         "X-ray, 2.0 Å or better; reference stratified by amino-acid type and burial", f"{npdb:,} ({nres:,} residues)", "RCSB PDB (Reference_cohort sheet)"],
        ["Experimental benchmark", "Apparent-resolution training, realism calibration, method-of-origin, two-axis comparison",
         f"X-ray {b['X-ray']}, cryo-EM {b['cryo-EM']}, NMR {b['NMR']}, AlphaFold2 {b['AlphaFold2']}; {len(rall)} with deposited resolution ({rall.min():.2f} to {rall.max():.2f} Å)",
         f"{len(BID):,}", "RCSB PDB; AlphaFold Database (see Benchmark sheet)"],
        ["AI prediction set", "Two-axis comparison, realism fingerprints, model selection",
         f"AlphaFold3 and Boltz2, {len(AIT)} targets, {n_ai // (2 * len(AIT))} models each predictor", f"{n_ai:,}", "Generated; targets in AI_prediction_targets sheet (leakage-stratified)"],
        ["Refinement-era cohort", "Refinement-era robustness (before vs after 1995; Supp Fig 5)",
         f"X-ray at fixed 2.0-2.5 Å; {pre} pre-1995, {post} from 1995 onward ({ERA['Deposition year'].min()}-{ERA['Deposition year'].max()})", f"{len(ERA)}", "RCSB PDB (Era_cohort sheet)"],
        ["Over-regularisation cohorts", "Raw backbone geometry comparison (Figure 2, panels c-f)",
         f"X-ray <1 Å ({nov['<1A X-ray']}), X-ray >2.5 Å ({nov['>2.5A X-ray']}), cryo-EM 2.0-3.0 Å ({nov['cryo-EM']}), AlphaFold3 ({nov['AF3']}), Boltz2 ({nov['Boltz2']})",
         f"{len(OVR)}", "RCSB PDB + prediction set (Over_regularisation sheet)"],
        ["Independent validation set", "Apparent-resolution transfer test (Figure 3d)",
         f"X-ray {ux} and cryo-EM {ue}, {used['Deposited resolution (Å)'].min():.2f}-{used['Deposited resolution (Å)'].max():.2f} Å, no MMseqs2 sequence-cluster overlap with training (30% id, 80% cov)",
         f"{len(used)} used (of {len(VAL)} with deposited resolution; {n_nofeat} lacked feature rows, {n_shared + n_other} removed by MMseqs2 cluster/method/resolution filters; {n_single} are singletons with no cluster)",
         "RCSB PDB (Validation_IDs sheet)"]],
        columns=["Dataset", "Role in study", "Composition", "n (structures or mutations)", "Source / accession"])
    return [("Summary", SUM), ("Benchmark", BEN), ("AI_prediction_targets", AIT), ("Reference_cohort", REF), ("Era_cohort", ERA),
            ("Over_regularisation", OVR), ("Benchmark_IDs", BID), ("Validation_IDs", VAL)]

def norm(df):
    def v(x):
        if pd.isna(x): return ""
        if isinstance(x, float): return f"{x:.6g}"
        return str(x).strip()
    return {tuple(v(x) for x in r) for r in df.itertuples(index=False)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--compare"); a = ap.parse_args()
    sheets = build()
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        for name, df in sheets:
            df.to_excel(xw, sheet_name=name, index=False)
            for i, c in enumerate(df.columns):
                xw.sheets[name].column_dimensions[chr(65 + i)].width = min(60, max(12, int(df[c].fillna("").astype(str).str.len().quantile(0.9)) + 2))
    print(f"[ms_table_s2] {', '.join(f'{n} {len(d)}' for n, d in sheets)} -> {OUT}")
    if a.compare:
        old = pd.read_excel(a.compare, sheet_name=None)
        for name, df in sheets:
            if name not in old: print(f"  [compare] {name}: not in the existing table"); continue
            o, n = norm(old[name]), norm(df)
            print(f"  [compare] {name:22s} rows {len(old[name])} -> {len(df)}; identical rows {len(o & n)}; only old {len(o - n)}; only new {len(n - o)}")
            for r in sorted(o - n)[:3]: print("      old:", r)
            for r in sorted(n - o)[:3]: print("      new:", r)
        for name in set(old) - {n for n, _ in sheets}: print(f"  [compare] sheet {name} exists only in the old table")
