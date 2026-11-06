#!/usr/bin/env python3.14
"""Remaining Methods numbers: per-method grouped-CV statistics of apparent resolution (out-of-fold),
the cryo-EM error range, cohort counts, and whether the out-of-fold median rises monotonically across the
deposited-resolution bins of each experimental method (the claim made for Figure 3e)."""
import os, numpy as np, pandas as pd
from ms_common import fd, recorder, AN
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
rec = recorder(__file__)
o = pd.read_csv(os.path.join(AN, "models", "apparent_resolution_oof.tsv"), sep="\t")
for m in ["xray", "cryoem"]:
    s = o[o.category == m]
    rec(f"appres_cv_spearman_{m}", spearmanr(s.res, s.oof)[0], f"grouped-CV Spearman, {m}")
    rec(f"appres_cv_slope_{m}", float(np.polyfit(s.res, s.oof, 1)[0]), f"grouped-CV slope of apparent vs deposited, {m}")
    rec(f"appres_cv_r2_{m}", r2_score(s.res, s.oof), f"grouped-CV R2, {m}"); rec(f"appres_cv_mae_{m}", mean_absolute_error(s.res, s.oof), f"grouped-CV MAE, {m}", "A")
man = fd._manifest(); o["k"] = o["id"].astype(str).str.lower()
o = o.merge(man[["k", "bin_label"]].drop_duplicates("k"), on="k", how="left")
for m in ["xray", "cryoem"]:
    s = o[o.category == m].copy(); s["lo"] = s.bin_label.str.extract(r"_(\d\.\d)-")[0].astype(float)
    med = s.groupby("lo").oof.median().sort_index()
    rec(f"fig3e_oof_bin_medians_{m}", ";".join(f"{k}:{v:.2f}" for k, v in med.items()), f"out-of-fold median per deposited-resolution bin, {m}")
    rec(f"fig3e_oof_monotonic_{m}", bool((np.diff(med.values) > 0).all()), f"out-of-fold median rises monotonically across bins, {m}")
D, _ = fd.origin_dataset()
for g in ["NMR", "AlphaFold2"]: rec(f"n_benchmark_{g}", int((D.origin == g).sum()), f"distinct {g} structures in the benchmark")
print("[ms_extra] done")
