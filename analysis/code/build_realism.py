#!/usr/bin/env python3.14
"""build_realism.py: the realism score (WhatRes), geometry-only.

Each of 72 per-residue geometry descriptors is residualised on apparent resolution (monotone
a + b*exp(-k*x), fitted on experimental structures), then an L2-regularised logistic regression
(C = 0.003) separates experimental structures from predicted models (AlphaFold3, Boltz2) on the
standardised residuals. realism = 100 * (1 - P(model)). Training logic: whatres_train.py.

Apparent resolution here is never memorised: out-of-fold for the experimental structures the
apparent-resolution model was trained on (models/apparent_resolution_oof.tsv), the deployed prediction
for the predicted models (never in apparent-resolution training), i.e. what any new structure receives.

Honest evaluation: grouped 5-fold CV (MMseqs2 clusters; a predicted model takes its target's cluster),
with curves and classifier refitted inside every training fold. These out-of-fold scores are THE realism
of every training structure in any analysis or figure (models/realism_oof.tsv).

Outputs: models/over_regularisation_model.joblib {clf, panel, appres_resid_coef, C}, models/realism_oof.tsv
"""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, sys, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "code"))
sys.path.insert(0, os.path.join(HERE, "paper_figures")); import numbers_registry as NR
import whatres, whatres_train as WT
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
SRC = "code/build_realism.py"
def rec(k, v, d, u=""): NR.record(k, v, d, SRC, u)

man = pd.read_csv(os.path.join(HERE, "..", "benchmark", "benchmark_manifest.tsv"), sep="\t"); man["k"] = man["id"].astype(str).str.lower()
fm = pd.read_csv(os.path.join(HERE, "feature_matrix.tsv"), sep="\t"); fm["k"] = fm["id"].astype(str).str.lower().str.replace(r'\.(pdb|ent)$', '', regex=True)
fm = fm.merge(man[["k", "category"]], on="k", how="left")
exp = fm[fm.category.isin(["xray", "cryoem"])].copy().reset_index(drop=True)
ai = pd.read_csv(os.path.join(HERE, "ai_feature_matrix.tsv"), sep="\t"); ai = ai[~ai["id"].astype(str).str.startswith("---")]
ai["tool"] = ai["id"].astype(str).str.extract(r'^(af3|boltz2)')[0]; ai = ai[ai.tool.isin(["af3", "boltz2"])].copy().reset_index(drop=True)
PAN = [c for c in WT.realism_panel(exp.columns) if c in ai.columns]

oofap = pd.read_csv(os.path.join(HERE, "models", "apparent_resolution_oof.tsv"), sep="\t")
oofmap = dict(zip(oofap["id"].astype(str).str.lower(), oofap["oof"]))
dep = np.asarray(whatres.apparent_resolution(exp))
exp["appres"] = [oofmap.get(i, a) for i, a in zip(exp["id"].astype(str).str.lower(), dep)]
n_exp_oof = int(sum(i in oofmap for i in exp["id"].astype(str).str.lower()))
ai["appres"] = np.asarray(whatres.apparent_resolution(ai))

ALL = pd.concat([exp.assign(cls=0, tool=exp.category), ai.assign(cls=1)], ignore_index=True)
grp = WT.group_of(ALL["id"]); yy = ALL.cls.values
p_oof = np.full(len(ALL), np.nan); fold = np.full(len(ALL), -1)
for f, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(ALL, yy, grp)):
    clf_f, CO_f = WT.fit_realism(ALL.iloc[a], PAN)
    p_oof[b] = clf_f.predict_proba(WT.resid(ALL.iloc[b], CO_f, PAN))[:, 1]; fold[b] = f
auc = roc_auc_score(yy, p_oof)

clf, CO = WT.fit_realism(ALL, PAN)
X = WT.resid(ALL, CO, PAN)
auc_old = roc_auc_score(yy, cross_val_predict(WT.make_realism(), X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1])
joblib.dump({"clf": clf, "panel": PAN, "appres_resid_coef": CO, "C": WT.C_REG, "oof_file": "realism_oof.tsv"},
            os.path.join(HERE, "models", "over_regularisation_model.joblib"))
ALL["realism_oof"] = 100 * (1 - p_oof); ALL["fold"] = fold; ALL["group"] = grp
ALL[["id", "cls", "tool", "group", "fold", "appres", "realism_oof"]].to_csv(os.path.join(HERE, "models", "realism_oof.tsv"), sep="\t", index=False)
print(f"[realism] panel: {len(PAN)} geometry descriptors | experimental {len(exp)} ({n_exp_oof} with out-of-fold apparent resolution), predicted {len(ai)}")
print(f"[realism] grouped, nested CV AUC = {auc:.3f}  (earlier method, ungrouped and curves on all data: {auc_old:.3f})")
for t in ["xray", "cryoem", "af3", "boltz2"]:
    v = ALL.loc[ALL.tool == t, "realism_oof"]; print(f"[realism]   out-of-fold median {t:7s} {v.median():.1f}  (n={len(v)})")
rec("n_realism_features", len(PAN), "geometry descriptors in the realism model")
rec("realism_cv_auc", auc, "realism grouped (MMseqs2/target), nested 5-fold CV AUC, experimental vs AF3/Boltz2")
rec("realism_cv_auc_earlier_method", auc_old, "realism CV AUC, ungrouped folds, curves on all data (earlier method)")
rec("realism_C", WT.C_REG, "logistic regression regularisation C")
print("[realism] saved models/over_regularisation_model.joblib + models/realism_oof.tsv")
