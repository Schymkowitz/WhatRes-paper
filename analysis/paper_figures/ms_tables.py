#!/usr/bin/env python3.14
"""Manuscript supplementary tables -> native_pipeline/tables/
S1 'WhatRes Suppl Table 1 Parameters.xlsx'          every descriptor with provenance (code/build_param_table.py)
S2 made by ms_table_s2.py
S3 'WhatRes Suppl Table 3 Realism parameters.xlsx'  the 72 realism descriptors: rho with apparent resolution
    (raw descriptor, experimental), Cohen's d (residualised descriptor, AF3+Boltz2 minus experimental), logistic weight
"""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, sys, subprocess, numpy as np, pandas as pd, joblib, openpyxl, re
from ms_common import fd, whatres, recorder, AN
from scipy.stats import spearmanr
rec = recorder(__file__); T = os.path.join(AN, "native_pipeline", "tables"); os.makedirs(T, exist_ok=True)

# ---- S1 ----
subprocess.run([sys.executable, os.path.join(AN, "code", "build_param_table.py")], check=True, cwd=AN, capture_output=True)
s1 = pd.read_excel(os.path.join(T, "WhatRes Suppl Table 1 Parameters.xlsx"), sheet_name=None)
from whatres.features import LEGACY_ENERGY_METRICS
hits = sum(int(df.astype(str).apply(lambda c: c.str.contains("|".join(LEGACY_ENERGY_METRICS), regex=True)).values.sum()) for df in s1.values())
assert hits == 0, f"legacy energy-term entries left in Table S1 ({hits} cells)"
rec("n_descriptors_table_s1", len(s1["Parameters"]), "descriptors listed in Supplementary Table S1")

# ---- S3 ----
M = joblib.load(os.path.join(AN, "models", "over_regularisation_model.joblib")); PAN, CO, CLF = M["panel"], M["appres_resid_coef"], M["clf"]
coef = dict(zip(PAN, CLF.named_steps["logisticregression"].coef_[0]))
D, _ = fd.origin_dataset(); D["appres"] = fd.honest_appres(D)
EXP = D.origin.isin(["X-ray", "cryo-EM"]); AI = D.origin.isin(["AlphaFold3", "Boltz2"])
def cohen(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return (a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
DESC = {"cb_deviation": "Cβ deviation from ideal position", "ca_chirality": "Cα improper dihedral (N–Cα–C–Cβ); L/D chirality",
        "omega": "Backbone ω (peptide) torsion", "peptide_planarity": "Peptide-bond planarity (|ω| from 0/180°)",
        "rotamer_outlier": "χ1 distance to nearest staggered well", "chi1": "Side-chain χ1 torsion", "chi2": "Side-chain χ2 torsion",
        "chi3": "Side-chain χ3 torsion (long side chains)", "chi4": "Side-chain χ4 torsion (Arg/Lys)",
        "cablam_mu_in": "CaBLAM μ_in Cα virtual dihedral", "cablam_mu_out": "CaBLAM μ_out Cα virtual dihedral",
        "cablam_nu": "CaBLAM ν carbonyl-O virtual dihedral", "cablam_ca_virtual": "CaBLAM Cα virtual angle",
        "phi": "Backbone φ torsion", "psi": "Backbone ψ torsion", "bond_N_CA": "N–Cα backbone bond length", "bond_CA_C": "Cα–C backbone bond length",
        "bond_C_O": "C=O carbonyl bond length", "bond_C_N": "C–N peptide bond length", "bond_CA_CB": "Cα–Cβ bond length",
        "angle_N_CA_C": "N–Cα–C backbone angle (τ)", "angle_CA_C_N": "Cα–C–N backbone angle", "angle_C_N_CA": "C–N–Cα backbone angle",
        "clashscore": "All-atom clashscore (MolProbity-style)"}
SUMM = {"mean_score": "mean surprise (−log₂ p)", "sd_pctile": "dispersion (SD of per-residue percentile)", "frac_extreme": "outlier fraction (pctile <5 or >95)"}
def split_feature(f):
    """qc_<descriptor>_<summary> -> (descriptor key, summary key)"""
    for sk in SUMM:
        if f.endswith("_" + sk) and f[3:-len(sk) - 1] in DESC: return f[3:-len(sk) - 1], sk
    raise KeyError(f)
# family and reference per descriptor: hand-written labels, kept in the repository (row order = table order)
LAB = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "realism_descriptor_labels.tsv"), sep="\t")
assert set(LAB.feature) == set(PAN) and len(LAB) == len(PAN), "label file and realism panel differ"
rows = []
for _, lb in LAB.iterrows():
    f = lb["feature"]; dk, sk = split_feature(f)
    r = {"Family": lb["family"], "Descriptor": DESC[dk], "Summary": SUMM[sk], "Reference": lb["reference"]}
    raw = pd.to_numeric(D[f], errors="coerce"); res_ = raw - whatres.resid_expected(CO[f], D["appres"].values)
    rows.append([r["Family"], r["Descriptor"], r["Summary"], round(spearmanr(raw[EXP], D.loc[EXP, "appres"], nan_policy="omit")[0], 2),
                 round(cohen(res_[AI], res_[EXP]), 2), round(float(coef[f]), 3), r["Reference"]])
    if f in ("qc_chi3_mean_score", "qc_chi4_mean_score", "qc_psi_mean_score"):
        rec(f"cohend_resid_{f}", cohen(res_[AI], res_[EXP]), f"Cohen's d, residualised {f}, AF3+Boltz2 minus X-ray+cryo-EM (Table S3 definition)")
assert len(rows) == 72 and len({(r[1], r[2]) for r in rows}) == 72
S3 = pd.DataFrame(rows, columns=["Family", "Descriptor", "Summary", "ρ(apparent res.)", "Cohen's d (AI−exp)", "Logistic weight", "Reference"])
p3 = os.path.join(T, "WhatRes Suppl Table 3 Realism parameters.xlsx")
with pd.ExcelWriter(p3, engine="openpyxl") as xw: S3.to_excel(xw, sheet_name="Realism parameters", index=False)

print(f"[ms_tables] S1 {len(s1['Parameters'])} descriptors, no legacy entries | S3 {len(S3)} rows -> {T}")
