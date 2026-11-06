#!/usr/bin/env python3.14
"""
build_apparent_resolution.py: apparent resolution (WhatRes).

An extremely-randomized-trees regressor (600 trees, each split drawn from a random 30% of the
descriptors) trained on experimental X-ray and cryo-EM structures to predict deposited resolution from
the descriptors that scale with it (|Spearman with resolution| > 0.3, excluding B-factor/pLDDT and
size-extensive descriptors). Training logic: whatres_train.py.

Honest evaluation: grouped 5-fold CV on MMseqs2 clusters with the descriptor selection repeated inside
every training fold. These out-of-fold predictions are THE apparent resolution of every training
structure in any analysis or figure: the deployed model, fitted on all of them, reproduces their
deposited resolution exactly (fully grown trees, no bootstrap) and must never be used to score them.

Outputs: models/apparent_resolution_model.joblib, models/apparent_resolution_oof.tsv,
         penalty_out/apparent_resolution_features.tsv, penalty_out/resolution_models.txt
"""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, sys, numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "code"))
sys.path.insert(0, os.path.join(HERE, "paper_figures")); import numbers_registry as NR
import whatres_train as WT
OUT = os.path.join(HERE, "penalty_out"); os.makedirs(OUT, exist_ok=True)
SRC = "code/build_apparent_resolution.py"
def rec(k, v, d, u=""): NR.record(k, v, d, SRC, u)
def log(*a): print(*a, flush=True)

tr, y, grp, cand = WT.appres_training()
log(f"[appres] training structures (X-ray+cryo-EM, exact resolution): {len(tr)}  | sequence groups: {len(set(grp))}")
scale, geom, rho_of = WT.select(tr, y, cand, np.arange(len(tr)))
log(f"[appres] resolution-scaling descriptors (all training data): {len(scale)}; size-invariant geometry: {len(geom)}")

gkf = GroupKFold(5)
oof = np.full(len(tr), np.nan); oof_glob = np.full(len(tr), np.nan); fold = np.full(len(tr), -1); nsel = []
for f, (a, b) in enumerate(gkf.split(tr, y, grp)):
    m, s_f = WT.fit_appres(tr, y, cand, a); nsel.append(len(s_f))
    oof[b] = WT.predict(m, WT.Xof(tr.iloc[b], s_f))
    oof_glob[b] = WT.predict(WT.make_appres().fit(WT.Xof(tr.iloc[a], scale), y[a]), WT.Xof(tr.iloc[b], scale))
    fold[b] = f
mae, r2, rho = mean_absolute_error(y, oof), r2_score(y, oof), spearmanr(y, oof)[0]; mae_g = mean_absolute_error(y, oof_glob)
log(f"[appres] grouped-CV (MMseqs2, selection inside folds): MAE {mae:.3f} A, R2 {r2:.3f}, Spearman {rho:.3f}; descriptors per fold {min(nsel)}-{max(nsel)}")
log(f"[appres]   (selection on all training data, earlier method: MAE {mae_g:.3f} A)")
pd.DataFrame({"id": tr["id"], "category": tr["method"], "res": y, "group": grp, "fold": fold, "oof": oof,
              "oof_selection_on_all": oof_glob}).to_csv(os.path.join(HERE, "models", "apparent_resolution_oof.tsv"), sep="\t", index=False)

reg = WT.make_appres().fit(WT.Xof(tr, scale), y)
joblib.dump({"regressor": reg, "scale_features": scale, "geometry_features": geom, "threshold": WT.THRESHOLD,
             "oof_file": "apparent_resolution_oof.tsv"}, os.path.join(HERE, "models", "apparent_resolution_model.joblib"))
log(f"[appres] saved models/apparent_resolution_model.joblib ({len(scale)} features) + models/apparent_resolution_oof.tsv")
rec("n_appres_features", len(scale), "resolution-scaling descriptors in the deployed apparent-resolution model")
rec("n_size_invariant_geometry", len(geom), "size-invariant geometry descriptors (training)")
rec("appres_cv_mae", mae, "grouped-CV MAE, MMseqs2 clusters, selection inside folds", "A")
rec("appres_cv_r2", r2, "grouped-CV R2, selection inside folds"); rec("appres_cv_spearman", rho, "grouped-CV Spearman, selection inside folds")
rec("appres_cv_mae_selection_on_all", mae_g, "grouped-CV MAE with selection on all training data (earlier method)", "A")
rec("appres_cv_nsel_min", min(nsel), "fewest descriptors selected in a fold"); rec("appres_cv_nsel_max", max(nsel), "most descriptors selected in a fold")
rec("appres_insample_mae", mean_absolute_error(y, WT.predict(reg, WT.Xof(tr, scale))), "deployed model on its own training structures (memorised)", "A")

fam = lambda c: "atom-type packing distance" if c.startswith("dist_") else "surface / cavity" if c.startswith("surf_") else "per-residue / whole-structure geometry"
ftab = pd.DataFrame({"descriptor": scale, "family": [fam(c) for c in scale], "spearman_with_resolution": [round(rho_of[c], 3) for c in scale]})
ftab.reindex(ftab["spearman_with_resolution"].abs().sort_values(ascending=False).index).to_csv(os.path.join(OUT, "apparent_resolution_features.tsv"), sep="\t", index=False)

def cv_mae(model, X):
    p = np.full(len(y), np.nan)
    for a, b in gkf.split(X, y, grp): p[b] = WT.predict(model.fit(X[a], y[a]), X[b]) if hasattr(model, "n_jobs") else model.fit(X[a], y[a]).predict(X[b])
    return mean_absolute_error(y, p), r2_score(y, p)
mae_rf = cv_mae(RandomForestRegressor(400, n_jobs=-1, random_state=42), WT.Xof(tr, scale))
mae_rdg = cv_mae(Ridge(alpha=10.0), StandardScaler().fit_transform(WT.Xof(tr, scale)))
with open(os.path.join(OUT, "resolution_models.txt"), "w") as f:
    f.write(f"Apparent-resolution model comparison (train X-ray+cryo-EM, n={len(y)}, GroupKFold(5) on MMseqs2 clusters)\n\n")
    f.write(f"{'model':46s} {'MAE(A)':>7s} {'R2':>6s}\n")
    f.write(f"{'ExtraTrees (deployed design), selection in folds':46s} {mae:7.3f} {r2:6.3f}\n")
    f.write(f"{'ExtraTrees, selection on all data':46s} {mae_g:7.3f} {r2_score(y, oof_glob):6.3f}\n")
    f.write(f"{'RandomForest, selection on all data':46s} {mae_rf[0]:7.3f} {mae_rf[1]:6.3f}\n")
    f.write(f"{'Ridge linear (scaled), selection on all data':46s} {mae_rdg[0]:7.3f} {mae_rdg[1]:6.3f}\n")
log(f"[appres] baselines: RF {mae_rf[0]:.3f} | Ridge {mae_rdg[0]:.3f}  -> penalty_out/resolution_models.txt")
