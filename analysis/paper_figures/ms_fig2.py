#!/usr/bin/env python3.14
"""Manuscript Figure 2: method of origin (a, b) and over-regularisation of backbone geometry (c-f).
a  Fig2a_LDA_projection   cross-validated LDA projection of the geometry-only descriptor space
b  Fig2b_confusion_matrix random-forest out-of-bag confusion (percent of each true-method row)
c-f Fig2cf_overregularisation  raw backbone geometry from coordinates (no descriptors, no model):
    produced by over_regularisation_geometry/make_geom_fig.py; copied here, numbers re-read.
"""
import os, shutil, numpy as np, pandas as pd
from ms_common import fs, fd, outdir, recorder, AN
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix

rec = recorder(__file__); O = outdir("Figure2")
D, feats = fd.origin_dataset()
X = D[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).values
y = D.origin.values
Xq = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(X)), random_state=0).fit_transform(X)
labels = fd.ORIGIN_ORDER
rec("fig2_n_structures", len(D), "structures in the method-of-origin dataset")
for g in labels: rec(f"fig2_n_{g}", int((y == g).sum()), f"{g} structures in the method-of-origin dataset")
rec("fig2_n_features", len(feats), "size-invariant geometry-only descriptors used for method of origin")

# a: cross-validated LDA projection
cv = cross_val_score(LDA(solver="eigen", shrinkage="auto"), Xq, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
Z = LDA(solver="eigen", shrinkage="auto", n_components=2).fit(Xq, y).transform(Xq)
figA, ax = plt.subplots(figsize=(4.2, 3.8))
for g in ["cryo-EM", "X-ray", "NMR", "AlphaFold2", "Boltz2", "AlphaFold3"]:
    m = y == g; col, mk = fd.ORIGIN_STYLE[g]; expg = g in ("X-ray", "cryo-EM", "NMR")
    ax.scatter(Z[m, 0], Z[m, 1], s=(16 if not expg else 11), marker=mk, c=col, alpha=(0.5 if expg else 0.85),
               edgecolors="k" if not expg else "none", linewidths=0.25 if not expg else 0, label=g)
H, L = ax.get_legend_handles_labels(); order = [L.index(g) for g in labels]
ax.legend([H[i] for i in order], [L[i] for i in order], markerscale=1.4, framealpha=.95, loc="best")
ax.set_xlabel("LDA 1"); ax.set_ylabel("LDA 2"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"method of origin from geometry (cross-validated {100*cv:.0f}%)")
fs.savepanel(figA, O, "Fig2a_LDA_projection")
rec("fig2_lda_cv_pct", 100 * cv, "LDA 5-fold cross-validated accuracy, method of origin", "%")

# b: random-forest out-of-bag confusion
rf = RandomForestClassifier(500, oob_score=True, random_state=0, n_jobs=-1).fit(Xq, y)
oob = rf.classes_[np.argmax(rf.oob_decision_function_, axis=1)]
cm = confusion_matrix(y, oob, labels=labels); cmn = 100 * cm / cm.sum(1, keepdims=True)
figB, ax2 = plt.subplots(figsize=(4.4, 3.9))
im = ax2.imshow(cmn, cmap="Greens", vmin=0, vmax=100)
ax2.set_xticks(range(6)); ax2.set_yticks(range(6))
ax2.set_xticklabels(labels, rotation=35, ha="right"); ax2.set_yticklabels(labels)
for i in range(6):
    for j in range(6):
        ax2.text(j, i, f"{cmn[i,j]:.0f}", ha="center", va="center", color="white" if cmn[i, j] > 55 else "black")
ax2.set_xlabel("predicted method"); ax2.set_ylabel("true method")
ax2.set_title(f"out-of-bag classification ({100*rf.oob_score_:.0f}%)")
figB.colorbar(im, ax=ax2, fraction=0.046, label="% of row")
fs.savepanel(figB, O, "Fig2b_confusion_matrix")
rec("fig2_rf_oob_pct", 100 * rf.oob_score_, "random-forest out-of-bag accuracy, method of origin", "%")
ix = {g: i for i, g in enumerate(labels)}; EXP = ["X-ray", "cryo-EM", "NMR"]
for g in ["AlphaFold2", "AlphaFold3", "Boltz2"]:
    rec(f"fig2_pct_{g}_called_experimental", float(cmn[ix[g], [ix[e] for e in EXP]].sum()),
        f"% of {g} models classified as an experimental method (X-ray, cryo-EM or NMR)", "%")
rec("fig2_pct_xray_called_cryoem", float(cmn[ix["X-ray"], ix["cryo-EM"]]), "% of X-ray classified as cryo-EM", "%")
rec("fig2_pct_cryoem_called_xray", float(cmn[ix["cryo-EM"], ix["X-ray"]]), "% of cryo-EM classified as X-ray", "%")
rec("fig2_pct_af3_called_af2", float(cmn[ix["AlphaFold3"], ix["AlphaFold2"]]), "% of AlphaFold3 classified as AlphaFold2", "%")
rec("fig2_pct_af2_called_af3", float(cmn[ix["AlphaFold2"], ix["AlphaFold3"]]), "% of AlphaFold2 classified as AlphaFold3", "%")
pd.DataFrame(cmn, index=labels, columns=labels).to_csv(os.path.join(O, "Fig2b_confusion_matrix.tsv"), sep="\t")

# c-f: coordinate-only over-regularisation strip (no descriptors, no model): copy + re-read numbers
G = os.path.join(AN, "over_regularisation_geometry")
shutil.copy2(os.path.join(G, "Fig_overregularisation_geometry.pdf"), os.path.join(O, "Fig2cf_overregularisation.pdf"))
shutil.copy2(os.path.join(G, "Fig_overregularisation_geometry.png"), os.path.join(O, "Fig2cf_overregularisation.png"))
off = pd.read_csv(os.path.join(G, "cohort_mean_offset.tsv"), sep="\t")
po = off.pivot(index="param", columns="cohort", values="mean_offset_vs_1A")
angs = ["angle_CA_C_N", "angle_CA_C_O", "angle_C_N_CA", "angle_N_CA_C", "angle_N_CA_CB"]
for c in ["AF3", "Boltz2"]:
    rec(f"fig2f_mean_abs_angle_offset_{c}", float(po.loc[angs, c].abs().mean()), f"{c} mean |offset| of 5 backbone angles vs <1 A X-ray", "deg")
rec("fig2f_max_angle_offset_AF3", float(po.loc[angs, "AF3"].max()), "AF3 largest backbone-angle offset vs <1 A X-ray", "deg")
rec("fig2f_max_angle_offset_AF3_param", str(po.loc[angs, "AF3"].idxmax()), "backbone angle with the largest AF3 offset")
sd = pd.read_csv(os.path.join(G, "cohort_within_type_SD.tsv"), sep="\t")
ps = sd.pivot(index="param", columns="cohort", values="within_type_SD"); ratio = ps.div(ps["<1A X-ray"], axis=0)
for c in [c for c in ratio.columns if c != "<1A X-ray"]:
    rec(f"fig2e_sd_ratio_min_{c}", float(ratio[c].min()), f"{c}: smallest within-type SD ratio to <1 A X-ray")
    rec(f"fig2e_sd_ratio_max_{c}", float(ratio[c].max()), f"{c}: largest within-type SD ratio to <1 A X-ray")
print(f"[ms_fig2] LDA {100*cv:.1f}%  RF OOB {100*rf.oob_score_:.1f}%  features {len(feats)}  -> {O}")
