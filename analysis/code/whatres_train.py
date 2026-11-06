"""whatres_train.py: the one implementation of how the two WhatRes models are trained.

Used by build_apparent_resolution.py and build_realism.py, and by any analysis that has to refit a
fold model (e.g. the size-subsampling figure), so training and analysis can never drift apart.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

AN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD, SIZE_MAX = 0.3, 0.35            # |rho with resolution| cutoff; |rho with residue count| size cutoff
N_TREES, MAX_FEAT, SEED = 600, 0.3, 0      # ExtraTrees design
C_REG = 0.003                              # realism logistic regularisation
# energy-metric columns present only in feature tables built before the pipeline became energy-free
LEGACY_ENERGY = {"total_energy", "backbone_hbond", "sidechain_hbond", "vdw", "electrostatics", "solv_polar",
 "solv_hydrophobic", "vdw_clashes", "entropy_sc", "entropy_mc", "sloop_entropy", "mloop_entropy", "cis_bond",
 "torsional_clash", "backbone_clash", "helix_dipole", "water_bridge", "disulfide", "electrostatic_kon",
 "partial_covalent", "energy_ionisation", "entropy_complex", "hetero_back_hbond", "hetero_side_hbond", "worst_score"}
REALISM_METRICS = ["cb_deviation", "ca_chirality", "omega", "peptide_planarity", "rotamer_outlier", "chi1", "chi2", "chi3", "chi4",
    "cablam_mu_in", "cablam_mu_out", "cablam_nu", "cablam_ca_virtual", "phi", "psi", "bond_N_CA", "bond_CA_C", "bond_C_O",
    "bond_C_N", "bond_CA_CB", "angle_N_CA_C", "angle_CA_C_N", "angle_C_N_CA", "clashscore"]

def is_legacy(c):
    core = c.replace("qc_", ""); return any(core.startswith(f + "_") or core == f for f in LEGACY_ENERGY)

def cluster_map():
    """PDB/target id (upper) -> MMseqs2 cluster representative (30% identity, 80% coverage)."""
    m = {}
    for ln in open(os.path.join(AN, "seq_clusters", "mmseqs_clusters_id30_cov80.tsv")):
        rep, mem = ln.rstrip("\n").split("\t"); m[mem.upper()] = rep.upper()
    return m

def group_of(ids, predicted=None):
    """Grouping for cross-validation: the MMseqs2 cluster of the entry; a predicted model (af3_X, boltz2_X)
    takes the cluster of its target X. Unclustered ids are their own singleton group."""
    m = cluster_map(); out = []
    for i in ids:
        i = str(i); e = i.split("_", 1)[1] if i.lower().startswith(("af3_", "boltz2_")) else i
        out.append(m.get(e.upper(), e.upper()))
    return np.array(out)

# ── apparent resolution ──
def appres_training():
    """(tr, y, grp, cand): X-ray + cryo-EM structures with exact deposited resolution, their groups and
    the candidate descriptor pool."""
    fm = pd.read_csv(os.path.join(AN, "feature_matrix.tsv"), sep="\t"); fm["id"] = fm["id"].str.lower()
    man = pd.read_csv(os.path.join(AN, "..", "benchmark", "benchmark_manifest.tsv"), sep="\t"); man["rid"] = man["id"].astype(str).str.lower()
    af = man["category"].str.startswith("alphafold"); man.loc[af, "rid"] = "af-" + man.loc[af, "id"].astype(str).str.lower()
    d = fm.merge(man[["rid", "bin_label"]].drop_duplicates("rid"), left_on="id", right_on="rid", how="inner")
    d["method"] = d["bin_label"].map(lambda b: "xray" if b.startswith("xray") else "cryoem" if b.startswith("cryoem") else "o")
    ex = pd.read_csv(os.path.join(AN, "exact_resolution.tsv"), sep="\t")
    d["res"] = d["id"].map(dict(zip(ex["id"].str.lower(), ex["exact_resolution"])))
    tr = d[d["method"].isin(["xray", "cryoem"]) & d["res"].notna()].copy().reset_index(drop=True)
    cand = [c for c in fm.columns if c not in ("id", "qc_n_residues") and not c.startswith("bf_")
            and not is_legacy(c) and pd.api.types.is_numeric_dtype(fm[c])]
    return tr, tr["res"].values, group_of(tr["id"]), cand

def select(tr, y, cand, rows, threshold=THRESHOLD):
    """(scale, geom, rho): descriptor selection using only the given training rows."""
    t = tr.iloc[rows]; yy = y[rows]; size = t["qc_n_residues"].values
    scale, geom, rho = [], [], {}
    for c in cand:
        v = t[c].values
        sr = spearmanr(v, size, nan_policy="omit")[0]; neutral = np.isnan(sr) or abs(sr) <= SIZE_MAX
        if neutral: geom.append(c)
        rr = spearmanr(v, yy, nan_policy="omit")[0]
        if rr is None or np.isnan(rr) or abs(rr) <= threshold or not neutral: continue
        scale.append(c); rho[c] = rr
    return sorted(scale), sorted(geom), rho

def Xof(df, cols): return df.reindex(columns=cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
def make_appres(): return ExtraTreesRegressor(n_estimators=N_TREES, max_features=MAX_FEAT, random_state=SEED, n_jobs=-1)

def predict(m, X):
    """Deterministic prediction: a forest sums its trees in parallel in completion order, which changes the
    result at 1e-15 between runs and so breaks exact ties (rank statistics then differ). Fitting stays parallel
    (every tree has its own fixed seed); only the summation is done in tree order."""
    nj = m.n_jobs; m.n_jobs = 1
    try: return m.predict(X)
    finally: m.n_jobs = nj

def fit_appres(tr, y, cand, rows):
    """Fit the apparent-resolution model on the given rows (selection inside). Returns (model, scale)."""
    scale, _, _ = select(tr, y, cand, rows)
    return make_appres().fit(Xof(tr.iloc[rows], scale), y[rows]), scale

# ── realism ──
def realism_panel(cols):
    return [f"qc_{m}_{s}" for m in REALISM_METRICS for s in ("mean_score", "sd_pctile", "frac_extreme") if f"qc_{m}_{s}" in cols]

def _expsat(x, a, b, k): return a + b * np.exp(-k * x)

def fit_curves(e, PAN):
    """Monotone expected-value curve a + b*exp(-k*appres) of each descriptor, on experimental rows e
    (column 'appres' must hold their honest apparent resolution)."""
    CO = {}
    for c in PAN:
        v = pd.to_numeric(e[c], errors="coerce"); v = v.fillna(v.median()).values; x = e["appres"].values
        try:
            b0 = (np.mean(v[x < np.percentile(x, 25)]) - np.mean(v[x > np.percentile(x, 75)])) or 1.0
            p, _ = curve_fit(_expsat, x, v, p0=[np.median(v), b0, 1.0], bounds=([-1e9, -1e9, 0.01], [1e9, 1e9, 10]), maxfev=20000)
            CO[c] = ("expsat", [float(p[0]), float(p[1]), float(p[2])])
        except Exception:
            CO[c] = ("poly", list(np.polyfit(x, v, 1)))
    return CO

def expected(spec, x):
    x = np.asarray(x, float); kind, p = spec
    return p[0] + p[1] * np.exp(-p[2] * x) if kind == "expsat" else np.polyval(p, x)

def resid(df, CO, PAN):
    return pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").values - expected(CO[c], df["appres"].values) for c in PAN}).fillna(0.0).values

def make_realism(): return make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=5000))

def fit_realism(rows_df, PAN):
    """Fit curves (on the experimental rows, cls==0) and classifier on rows_df (needs 'appres','cls')."""
    CO = fit_curves(rows_df[rows_df.cls == 0], PAN)
    return make_realism().fit(resid(rows_df, CO, PAN), rows_df.cls.values), CO
