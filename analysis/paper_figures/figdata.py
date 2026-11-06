"""Shared data loaders for the WhatRes paper figures (single source of truth).
Centralises: the energy-term exclusion rule, the geometry-only size-invariant descriptor set, the
method-of-origin labelled dataset, and the apparent-resolution training set + candidate pool.
All scores are GEOMETRY-ONLY (energy terms excluded), matching the deployed models.
"""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr

AN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../analysis
BENCH = os.path.join(AN, "..", "benchmark")

# per-residue energy terms (and their worst-score aggregate) excluded from every score: geometry only
ENERGY_TERMS = {"total_energy","backbone_hbond","sidechain_hbond","vdw","electrostatics","solv_polar",
 "solv_hydrophobic","vdw_clashes","entropy_sc","entropy_mc","sloop_entropy","mloop_entropy","cis_bond",
 "torsional_clash","backbone_clash","helix_dipole","water_bridge","disulfide","electrostatic_kon",
 "partial_covalent","energy_ionisation","entropy_complex","hetero_back_hbond","hetero_side_hbond","worst_score"}
def is_energy_term(c):
    core=c.replace("qc_",""); return any(core.startswith(f+"_") or core==f for f in ENERGY_TERMS)

def _manifest():
    m=pd.read_csv(os.path.join(BENCH,"benchmark_manifest.tsv"),sep="\t"); m["k"]=m["id"].astype(str).str.lower(); return m
def _exact_res():
    e=pd.read_csv(os.path.join(AN,"exact_resolution.tsv"),sep="\t"); e["k"]=e["id"].astype(str).str.lower(); return e

def _geom_candidates(df, nres):
    """size-invariant, geometry-only candidate descriptors (no bf_, no energy terms, |rho(n_res)|<=0.35)."""
    out=[]
    for c in df.columns:
        if c in ("id","k","category","bin_label","origin","res","qc_n_residues"): continue
        if c.startswith("bf_") or is_energy_term(c) or not pd.api.types.is_numeric_dtype(df[c]): continue
        v=pd.to_numeric(df[c],errors="coerce")
        if v.notna().sum()<len(df)*0.5: continue
        sr=spearmanr(v,nres,nan_policy="omit")[0]
        if np.isnan(sr) or abs(sr)<=0.35: out.append(c)
    return out

def origin_dataset():
    """Return (D, feats): all benchmark experimental + AI structures labelled by method of origin,
    with the geometry-only size-invariant feature list."""
    man=_manifest(); cat=dict(zip(man.k,man.category))
    fm=pd.read_csv(os.path.join(AN,"feature_matrix.tsv"),sep="\t")
    fm["k"]=fm["id"].astype(str).str.lower().str.replace(r'\.(pdb|ent)$','',regex=True)
    def lab(idl,k):
        if str(idl).lower().startswith("af-"): return "AlphaFold2"
        return {"xray":"X-ray","cryoem":"cryo-EM","nmr":"NMR"}.get(cat.get(k))
    fm["origin"]=[lab(i,k) for i,k in zip(fm["id"],fm["k"])]
    # the benchmark list repeats 28 AlphaFold2 files (identical paths and bytes); the feature builder
    # renamed the repeats <id>_1, <id>_2. Keep one copy of each model and say how many were removed.
    dup = fm["k"].str.match(r"^af-.*_\d+$") & fm["k"].str.replace(r"_\d+$", "", regex=True).isin(set(fm["k"]))
    if dup.any(): print(f"[figdata] removed {int(dup.sum())} duplicate AlphaFold2 rows (same file listed more than once)")
    fm = fm[~dup].copy()
    exp=fm[fm.origin.notna()].copy()
    ai=pd.read_csv(os.path.join(AN,"ai_feature_matrix.tsv"),sep="\t"); ai=ai[~ai["id"].astype(str).str.startswith("---")]
    ai["origin"]=ai["id"].astype(str).str.extract(r'^(af3|boltz2)')[0].map({"af3":"AlphaFold3","boltz2":"Boltz2"})
    ai=ai[ai.origin.notna()]
    D=pd.concat([exp,ai],ignore_index=True).reset_index(drop=True)
    feats=_geom_candidates(D, pd.to_numeric(D.get("qc_n_residues"),errors="coerce"))
    return D, feats

def seq_cluster_map():
    """ID (upper) -> MMseqs2 cluster representative (30% identity, 80% coverage). Structures without a
    usable sequence are absent and fall back to their own ID (singleton group)."""
    p=os.path.join(AN,"seq_clusters","mmseqs_clusters_id30_cov80.tsv")
    m2rep={}
    for ln in open(p):
        rep,mem=ln.rstrip("\n").split("\t"); m2rep[mem.upper()]=rep.upper()
    return m2rep

def appres_training():
    """Return (tr, y, grp, pool, rho_res): X-ray+cryo-EM with deposited resolution, the geometry-only
    candidate pool, and each candidate's Spearman with resolution. grp = MMseqs2 sequence cluster
    (30% identity, 80% coverage) so homologues are held out together in grouped cross-validation."""
    man=_manifest(); er=_exact_res()
    fm=pd.read_csv(os.path.join(AN,"feature_matrix.tsv"),sep="\t"); fm["id"]=fm["id"].str.lower()
    fm["k"]=fm["id"].str.replace(r'\.(pdb|ent)$','',regex=True)
    fm=fm.merge(man[["k","category"]].drop_duplicates("k"),on="k")
    tr=fm[fm.category.isin(["xray","cryoem"])].copy(); tr["res"]=tr["id"].map(dict(zip(er.k,er.exact_resolution)))
    tr=tr[tr.res.notna()].reset_index(drop=True)
    _rep=seq_cluster_map()
    grp=tr["id"].str.upper().map(lambda i:_rep.get(i,i)).values
    y=tr.res.values; nres=pd.to_numeric(tr["qc_n_residues"],errors="coerce")
    pool=_geom_candidates(tr, nres)
    rho_res={c:(0.0 if np.isnan(spearmanr(pd.to_numeric(tr[c],errors="coerce"),y,nan_policy="omit")[0])
                else spearmanr(pd.to_numeric(tr[c],errors="coerce"),y,nan_policy="omit")[0]) for c in pool}
    return tr, y, grp, pool, rho_res

def _key(df):
    return df["id"].astype(str).str.lower().str.replace(r'\.(pdb|ent)$', '', regex=True)

def _oof(fname, col):
    t = pd.read_csv(os.path.join(AN, "models", fname), sep="\t")
    return dict(zip(t["id"].astype(str).str.lower(), t[col]))

def honest_appres(df):
    """Apparent resolution with no memorised values: the out-of-fold prediction for every structure the
    deployed model was trained on (it reproduces their deposited resolution exactly), the deployed
    prediction for every other structure."""
    import whatres
    dep = np.asarray(whatres.apparent_resolution(df), float); oof = _oof("apparent_resolution_oof.tsv", "oof")
    return np.array([oof.get(k, a) for k, a in zip(_key(df), dep)])

def honest_realism(df):
    """Realism with no memorised values: the out-of-fold score for every structure the realism classifier
    was trained on (X-ray, cryo-EM, AlphaFold3, Boltz2 training sets), the deployed score otherwise."""
    import whatres
    dep = np.asarray(whatres.realism_score(whatres.realism_load(df)), float); oof = _oof("realism_oof.tsv", "realism_oof")
    return np.array([oof.get(k, r) for k, r in zip(_key(df), dep)])

# consistent colour / marker scheme across figures
ORIGIN_STYLE = {
 "X-ray":("#2ca02c","o"), "cryo-EM":("#1f77b4","o"), "NMR":("#7f7f7f","o"),
 "AlphaFold2":("#8c564b","^"), "AlphaFold3":("#d62728","s"), "Boltz2":("#e08214","D"),
}
ORIGIN_ORDER = ["X-ray","cryo-EM","NMR","AlphaFold2","AlphaFold3","Boltz2"]
