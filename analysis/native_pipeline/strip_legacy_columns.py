#!/usr/bin/env python3.14
"""Remove the legacy columns that carry no information in the current pipeline, from the five
feature tables built by rebuild_all.sh / rebuild_era.sh before build_feature_matrix.py stopped
emitting them: the 24 legacy energy metrics (always pctile 50 / score 0) x 4 aggregations, and the
secondary-structure composition qc_frac_H/E/C (no secondary structure is assigned).
Same rule as build_feature_matrix.LEGACY_ENERGY_METRICS. Idempotent; reports what it dropped."""
import os, sys, pandas as pd
AN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(AN, "code"))
from whatres.features import LEGACY_ENERGY_METRICS
AGG = ("_mean_pctile", "_sd_pctile", "_frac_extreme", "_mean_score")
def legacy(c):
    if c in ("qc_frac_H", "qc_frac_E", "qc_frac_C"): return True
    if not c.startswith("qc_"): return False
    core = c[3:]
    return any(core == m + a for m in LEGACY_ENERGY_METRICS for a in AGG)
for f in ["feature_matrix.tsv", "ai_feature_matrix.tsv", "ai_all_feature_matrix.tsv",
          "dms_hero/val_feature_matrix.tsv", "era_before_after_1995/era_fm.tsv"]:
    p = os.path.join(AN, f); d = pd.read_csv(p, sep="\t", low_memory=False)
    drop = [c for c in d.columns if legacy(c)]
    if drop: d.drop(columns=drop).to_csv(p, sep="\t", index=False)
    print(f"[strip] {f}: {len(d)} rows, {d.shape[1]} -> {d.shape[1] - len(drop)} columns (dropped {len(drop)})")
