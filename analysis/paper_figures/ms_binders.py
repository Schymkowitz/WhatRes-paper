#!/usr/bin/env python3.14
"""Supplementary Figure 7 and every reported number on de novo binders (all values to the number registry).

Data: Overath et al. 2025 meta-analysis (bioRxiv 10.1101/2025.08.14.670059; Zenodo 10.5281/zenodo.15722219,
CC BY 4.0), fetched by binders/fetch_binder_dataset.py; WhatRes scores (binders/score_binders.py) and FoldX
AnalyseComplex energies on the raw complexes (binders/foldx_complexes.py) in $WHATRES_BINDER_DATA
(default ~/data/binder_meta_2025).

Unit of analysis: one design campaign against one target. 'Testable' campaigns have >= 5 binders and >= 5
non-binders. good model = realism >= 20 and apparent resolution <= 3.0 A (WhatRes on the complex the energy was
computed on); for the good-versus-poor comparison a campaign needs >= 3 binders and >= 3 non-binders on each side.
p values: labels (or good/poor) shuffled within campaign, fixed seeds.
a  within-campaign AUROC of AF3 ipSAE, force-field energies and WhatRes scores (13 campaigns)
b  energy AUROC on good versus poor AF3 complex models, per campaign (FoldX and Rosetta)
c  sensitivity of b to the realism cut-off"""
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from ms_common import fs, outdir, recorder
import matplotlib.pyplot as plt

rec = recorder(__file__); O = outdir("Supplementary")
B = os.environ.get("WHATRES_BINDER_DATA", os.path.expanduser("~/data/binder_meta_2025"))
RMIN, AMAX = 20, 3.0

raw = pd.read_csv(f"{B}/raw/final_dataset.csv", low_memory=False)
n_rows = len(raw); lab = raw[raw.binder.notna()]; dup = lab.binder_id.duplicated(keep=False)
rec("bind_n_rows", n_rows, "rows in the binder meta-analysis table"); rec("bind_n_labelled", int((~dup).sum()), "designs with an outcome label (unique binder_id)")
sc = pd.read_csv(f"{B}/whatres/scores.tsv", sep="\t")
ds = raw[["binder_id", "af3_ipSAE_min", "input_rosetta_interface_dG", "af3_rosetta_interface_dG"]].drop_duplicates("binder_id")
W = sc.pivot_table(index=["sid", "binder_id", "source", "target_id", "binder"], columns="set", values=["realism", "apparent_resolution"]).reset_index()
W.columns = ["_".join([c for c in col if c]) for col in W.columns]
fxa = pd.read_csv(f"{B}/foldx/foldx_af3_complex.tsv", sep="\t")[["sid", "Interaction Energy"]].rename(columns={"Interaction Energy": "foldx_af3"})
fxd = pd.read_csv(f"{B}/foldx/foldx_design_complex.tsv", sep="\t")[["sid", "Interaction Energy"]].rename(columns={"Interaction Energy": "foldx_design"})
d = W.merge(ds, on="binder_id").merge(fxa, on="sid", how="left").merge(fxd, on="sid", how="left")
d["y"] = d.binder.astype(str).str.lower().eq("true").astype(int); d["g"] = d.source + " | " + d.target_id
rec("bind_n_designs", len(d), "labelled designs scored by WhatRes (4 structure sets each)"); rec("bind_n_binders", int(d.y.sum()), "experimental binders")
rec("bind_n_campaigns", d.g.nunique(), "design campaigns (source x target)"); rec("bind_n_targets", d.target_id.nunique(), "targets")
rec("bind_foldx_failed", int(d.foldx_design.isna().sum() + d.foldx_af3.isna().sum()), "complexes FoldX AnalyseComplex could not analyse")

def auc(y, x):
    x = np.asarray(x, float); ok = np.isfinite(x); y, x = y[ok], x[ok]
    return mannwhitneyu(x[y == 1], x[y == 0]).statistic / (y.sum() * (1 - y).sum()) if y.sum() >= 3 and (1 - y).sum() >= 3 else np.nan

# ---------- a: within-campaign AUROC ----------
T = d.groupby("g").filter(lambda x: x.y.sum() >= 5 and (1 - x.y).sum() >= 5)
camps = sorted(T.g.unique())
rec("bind_n_testable", len(camps), "campaigns with >= 5 binders and >= 5 non-binders"); rec("bind_n_testable_designs", len(T), "designs in testable campaigns")
rec("bind_n_testable_binders", int(T.y.sum()), "binders in testable campaigns")
PRED = {"AF3 ipSAE": T.af3_ipSAE_min, "Rosetta, AF3 model": -T.af3_rosetta_interface_dG, "FoldX, AF3 model": -T.foldx_af3,
        "Rosetta, design model": -T.input_rosetta_interface_dG, "FoldX, design model": -T.foldx_design,
        "realism, design model": T.realism_design_complex, "realism, AF3 model": T.realism_af3_complex,
        "app. res., design model": T.apparent_resolution_design_complex, "app. res., AF3 model": T.apparent_resolution_af3_complex}
KEY = {"AF3 ipSAE": "ipsae", "Rosetta, AF3 model": "rosetta_af3", "FoldX, AF3 model": "foldx_af3", "Rosetta, design model": "rosetta_design",
       "FoldX, design model": "foldx_design", "realism, design model": "realism_design", "realism, AF3 model": "realism_af3",
       "app. res., design model": "appres_design", "app. res., AF3 model": "appres_af3"}
A = {k: [auc(T[T.g == g].y.values, v[T.g == g].values) for g in camps] for k, v in PRED.items()}
rng = np.random.default_rng(0)
for k, v in PRED.items():
    m = np.nanmean(A[k]); rec(f"bind_auroc_{KEY[k]}", m, f"mean within-campaign AUROC, {k}")
    if k.startswith(("realism", "app. res.")):
        ys = [T[T.g == g].y.values for g in camps]; xs = [v[T.g == g].values for g in camps]
        null = [np.nanmean([auc(rng.permutation(y), x) for y, x in zip(ys, xs)]) for _ in range(2000)]
        rec(f"bind_auroc_{KEY[k]}_p", (np.sum(np.abs(np.array(null) - 0.5) >= abs(m - 0.5)) + 1) / 2001, f"within-campaign permutation p, {k}")
    rec(f"bind_auroc_{KEY[k]}_n_above", int(np.sum(np.array(A[k]) > 0.5)), f"testable campaigns with AUROC > 0.5, {k}")

# added value over ipSAE: within-campaign 5-fold CV AUROC
def cv_auc(cols):
    out = []
    for g in camps:
        x = T[T.g == g]; X = x[cols].values.astype(float); y = x.y.values; pr = np.zeros(len(x))
        for a, b in StratifiedKFold(min(5, int(y.sum())), shuffle=True, random_state=0).split(X, y):
            mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
            pr[b] = LogisticRegression(max_iter=5000).fit((X[a] - mu) / sd, y[a]).predict_proba((X[b] - mu) / sd)[:, 1]
        out.append(roc_auc_score(y, pr))
    return np.array(out)
c0 = cv_auc(["af3_ipSAE_min"]); c1 = cv_auc(["af3_ipSAE_min", "realism_design_binder", "apparent_resolution_design_binder"])
rec("bind_cv_ipsae", c0.mean(), "within-campaign CV AUROC, ipSAE alone"); rec("bind_cv_ipsae_whatres", c1.mean(), "within-campaign CV AUROC, ipSAE + realism + apparent resolution (design model)")
rec("bind_cv_n_improved", int(np.sum(c1 > c0 + 1e-9)), "campaigns where adding WhatRes to ipSAE improved the CV AUROC")

# ---------- b, c: energy on good versus poor models ----------
def good_vs_poor(e_col, sign, rmin, amax, nperm=1000, seed=1, record=None):
    x0 = d.assign(e=sign * d[e_col], good=((d.realism_af3_complex >= rmin) & (d.apparent_resolution_af3_complex <= amax)).astype(int))
    rows = []
    for g, x in x0.groupby("g"):
        gg, pp = x[x.good == 1], x[x.good == 0]
        if min(gg.y.sum(), (1 - gg.y).sum(), pp.y.sum(), (1 - pp.y).sum()) >= 3:
            rows.append((g, len(gg), int(gg.y.sum()), auc(gg.y.values, gg.e.values), len(pp), int(pp.y.sum()), auc(pp.y.values, pp.e.values)))
    R = pd.DataFrame(rows, columns=["campaign", "n_good", "bind_good", "auc_good", "n_poor", "bind_poor", "auc_poor"])
    if R.empty: return R, np.nan, np.nan
    obs = float(np.mean(R.auc_good - R.auc_poor)); r = np.random.default_rng(seed); sub = [x0[x0.g == g] for g in R.campaign]; null = []
    for _ in range(nperm):
        dd = []
        for x in sub:
            l = r.permutation(x.good.values); dd.append(auc(x.y.values[l == 1], x.e.values[l == 1]) - auc(x.y.values[l == 0], x.e.values[l == 0]))
        null.append(np.nanmean(dd))
    p = (np.sum(np.abs(np.array(null)) >= abs(obs)) + 1) / (nperm + 1)
    return R, obs, p
RF, dF, pF = good_vs_poor("foldx_af3", -1, RMIN, AMAX); RR, dR, pR = good_vs_poor("af3_rosetta_interface_dG", -1, RMIN, AMAX)
for k, R, dv, pv in (("foldx", RF, dF, pF), ("rosetta", RR, dR, pR)):
    rec(f"bind_gp_{k}_n_campaigns", len(R), f"campaigns testable for good vs poor, {k}"); rec(f"bind_gp_{k}_n_better", int((R.auc_good > R.auc_poor).sum()), f"campaigns where the {k} energy classifies better on good models")
    rec(f"bind_gp_{k}_diff", dv, f"mean AUROC difference good - poor models, {k} energy on AF3 complexes"); rec(f"bind_gp_{k}_p", pv, f"permutation p (good/poor shuffled within campaign), {k}")
    rec(f"bind_gp_{k}_bind_good_min", int(R.bind_good.min()), f"fewest binders in a good-model group, {k}"); rec(f"bind_gp_{k}_bind_good_max_small", int(R.bind_good.nsmallest(len(R) - 1).max()), f"most binders in a good-model group outside the largest campaign, {k}")
loo = [float(np.mean([a - b for c, a, b in zip(RF.campaign, RF.auc_good, RF.auc_poor) if c != g])) for g in RF.campaign]
rec("bind_gp_foldx_loo_min", min(loo), "leave-one-campaign-out mean difference, minimum (FoldX)"); rec("bind_gp_foldx_loo_max", max(loo), "leave-one-campaign-out mean difference, maximum (FoldX)")
for c, a, b in zip(RF.campaign, RF.auc_good, RF.auc_poor):
    t = c.split(" | ")[1]; rec(f"bind_gp_foldx_{t}_good", a, f"FoldX AUROC on good models, {c}"); rec(f"bind_gp_foldx_{t}_poor", b, f"FoldX AUROC on poor models, {c}")
SENS = []
for rmin in (10, 15, 20, 25, 30):
    R, dv, pv = good_vs_poor("foldx_af3", -1, rmin, AMAX); SENS.append((rmin, len(R), dv, pv))   # same permutations as the main test
    rec(f"bind_sens_foldx_r{rmin}_diff", dv, f"FoldX good - poor AUROC difference at realism >= {rmin}"); rec(f"bind_sens_foldx_r{rmin}_n", len(R), f"testable campaigns at realism >= {rmin}")
    rec(f"bind_sens_foldx_r{rmin}_p", pv, f"permutation p at realism >= {rmin}")

# ---------- figure ----------
fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 0.9]})
ax = axs[0]; order = list(PRED); xs = np.arange(len(order))
COL = {"AF3 ipSAE": "#4d4d4d", "Rosetta": "#1b7837", "FoldX": "#762a83", "realism": "#2166ac", "app. res.": "#b2182b"}
for i, k in enumerate(order):
    col = next(v for p, v in COL.items() if k.startswith(p)); v = np.array(A[k], float)
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=12, color=col, alpha=0.6, edgecolors="none")
    ax.hlines(np.nanmean(v), i - 0.3, i + 0.3, color="k", lw=1.6)
ax.axhline(0.5, color="0.5", ls=":", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(order, rotation=40, ha="right"); ax.set_ylim(0, 1.02)
ax.set_ylabel("AUROC within campaign"); ax.set_title(f"binder vs non-binder ({len(camps)} campaigns)")
ax = axs[1]
for R, col, mk, lab_ in ((RF, COL["FoldX"], "o", "FoldX"), (RR, COL["Rosetta"], "s", "Rosetta")):
    for j, r in enumerate(R.itertuples()):
        ax.plot([0, 1], [r.auc_poor, r.auc_good], color=col, alpha=0.55, lw=1); ax.scatter([0, 1], [r.auc_poor, r.auc_good], color=col, marker=mk, s=16, label=lab_ if j == 0 else None, zorder=3)
ax.axhline(0.5, color="0.5", ls=":", lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(["poor models", "good models"]); ax.set_xlim(-0.3, 1.3); ax.set_ylim(0.2, 1.05)
ax.set_ylabel("AUROC of interaction energy"); ax.set_title("energy on AF3 complexes"); ax.legend(frameon=False, loc="lower right")
ax = axs[2]; S = pd.DataFrame(SENS, columns=["rmin", "n", "diff", "p"])
ax.plot(S.rmin, S["diff"], "o-", color=COL["FoldX"]); ax.axhline(0, color="0.5", lw=0.8)
for r in S.itertuples(): ax.annotate(f"n={r.n}", (r.rmin, r.diff), textcoords="offset points", xytext=(9, -3), ha="left", fontsize=7)
ax.set_ylim(min(-0.02, S["diff"].min() - 0.02), S["diff"].max() * 1.2); ax.set_xlim(7, 33)
ax.set_xlabel("realism cut-off for 'good'"); ax.set_ylabel("AUROC difference, good - poor"); ax.set_title("cut-off sensitivity (FoldX)")
fig.tight_layout(); fs.panel_letters(fig, axs); fs.savepanel(fig, O, "SuppFig7_binders")
print(f"[ms_binders] {len(d)} designs, {d.y.sum()} binders, {len(camps)} testable campaigns | ipSAE {np.nanmean(A['AF3 ipSAE']):.2f}, "
      f"realism {np.nanmean(A['realism, design model']):.2f}/{np.nanmean(A['realism, AF3 model']):.2f} | FoldX good-poor {dF:+.3f} p {pF:.3f} ({(RF.auc_good > RF.auc_poor).sum()}/{len(RF)}), "
      f"Rosetta {dR:+.3f} p {pR:.3f} | sensitivity " + " ".join(f"r{r.rmin}:{r.diff:+.2f}(n{r.n})" for r in S.itertuples()))
