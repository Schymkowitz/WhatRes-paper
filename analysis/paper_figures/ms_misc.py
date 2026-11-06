#!/usr/bin/env python3.14
"""Manuscript numbers that belong to no single figure:
descriptor counts, reference-set size, the MolProbity comparison on the shared structures, the ridge
'resolution axis' (out-of-fold), and the shared-descriptor count."""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, re, numpy as np, pandas as pd
from ms_common import fd, whatres, recorder, AN
import whatres_train as WT
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
rec = recorder(__file__)

fm = pd.read_csv(os.path.join(AN, "feature_matrix.tsv"), sep="\t")
desc = [c for c in fm.columns if c != "id" and pd.api.types.is_numeric_dtype(fm[c])]
rec("n_descriptors", len(desc), "per-structure descriptors in the feature table (numeric columns)")
rec("n_descriptors_excl_bfactor", len([c for c in desc if not c.startswith("bf_")]), "per-structure descriptors excluding B-factor/pLDDT")
SF, RF = whatres.SCALE_FEATURES, whatres.REALISM_FEATURES
rec("n_shared_features", len(set(SF) & set(RF)), "descriptors shared by apparent resolution and realism")
s = open(os.path.join(AN, "refbuild", "native49_structure_survey_summary.txt")).readline()
npdb, nres = map(int, re.search(r"(\d+) PDBs, (\d+) residues", s).groups())
rec("refdb_n_structures", npdb, "high-resolution reference structures"); rec("refdb_n_residues", nres, "residues in the reference distributions")

# MolProbity comparison on the structures scored by both (honest out-of-fold apparent resolution)
mp = pd.read_csv(os.path.join(AN, "molprobity_baseline", "molprobity_train_genuine.tsv"), sep="\t"); mp["k"] = mp["k"].astype(str).str.lower()
oof = pd.read_csv(os.path.join(AN, "models", "apparent_resolution_oof.tsv"), sep="\t"); oof["k"] = oof["id"].astype(str).str.lower()
c = mp.dropna(subset=["mpscore"]).merge(oof[["k", "oof", "group"]], on="k").reset_index(drop=True)
rec("mp_n_common", len(c), "structures scored by both WhatRes and MolProbity")
rec("mp_whatres_mae", mean_absolute_error(c.res, c.oof), "WhatRes grouped-CV MAE on the shared structures", "A")
p = np.full(len(c), np.nan); X = c[["mpscore"]].values
for a, b in GroupKFold(5).split(X, c.res, c.group): p[b] = LinearRegression().fit(X[a], c.res.values[a]).predict(X[b])
rec("mp_molprobity_mae", mean_absolute_error(c.res, p), "MolProbity score, linear calibration, grouped-CV MAE on the shared structures", "A")
rec("mp_molprobity_mae_insample", mean_absolute_error(c.res, LinearRegression().fit(X, c.res).predict(X)), "MolProbity score, linear calibration fitted on all shared structures", "A")

# ridge 'resolution axis' (Methods): projection onto the unit ridge direction, out-of-fold
tr, y, grp, cand = WT.appres_training(); Z = [f for f in whatres.GEOMETRY_FEATURES if f in tr.columns]
Xz = WT.Xof(tr, Z); proj = np.full(len(y), np.nan)
for a, b in GroupKFold(5).split(Xz, y, grp):
    sc = StandardScaler().fit(Xz[a]); w = Ridge(alpha=10).fit(sc.transform(Xz[a]), y[a]).coef_; w = w / np.linalg.norm(w)
    proj[b] = sc.transform(Xz[b]) @ w
sc = StandardScaler().fit(Xz); w = Ridge(alpha=10).fit(sc.transform(Xz), y).coef_; w = w / np.linalg.norm(w)
rec("ridge_axis_spearman_oof", spearmanr(proj, y)[0], "Spearman of the ridge resolution-axis projection with deposited resolution, out-of-fold")
rec("ridge_axis_spearman_insample", spearmanr(sc.transform(Xz) @ w, y)[0], "same, fitted and projected on the same structures (earlier method)")
rec("ridge_axis_n_features", len(Z), "size-invariant geometry descriptors used for the resolution axis")
print(f"[ms_misc] descriptors {len(desc)} | shared {len(set(SF)&set(RF))} | refDB {npdb}/{nres} | MolProbity n={len(c)} WhatRes {mean_absolute_error(c.res,c.oof):.3f} vs MP {mean_absolute_error(c.res,p):.3f} | ridge axis OOF {spearmanr(proj,y)[0]:.3f}")
