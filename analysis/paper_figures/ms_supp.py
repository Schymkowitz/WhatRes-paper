#!/usr/bin/env python3.14
"""Manuscript supplementary figures (all values honest: out-of-fold for training structures).
S1 SuppFig1_realism_vs_resolution   realism vs deposited resolution, X-ray + cryo-EM
S2 SuppFig2_ai_vs_experimental      realism vs apparent resolution; matched 1.8-2.6 A band by method
S3 SuppFig3_shoulder_helix_sheet    made by ms_supp3.py
S4 SuppFig4a_threshold_sweep        grouped-CV MAE vs selection threshold (selection inside folds)
   SuppFig4b_random_control         selected descriptors vs random draws of the same size
S5 SuppFig5_era_before_after_1995   refinement era at fixed 2.0-2.5 A
S6 SuppFig6_size_reliability        realism vs length; subsampling precision; long-side-chain counts
"""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, sys, shutil, glob, numpy as np, pandas as pd, joblib
from ms_common import fs, fd, whatres, outdir, recorder, AN, DATA
import whatres_train as WT
from whatres.features import aggregate_quality_df
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr, mannwhitneyu, levene
from sklearn.metrics import mean_absolute_error

rec = recorder(__file__); O = outdir("Supplementary"); ST = fd.ORIGIN_STYLE
D, _ = fd.origin_dataset(); D["k"] = fd._key(D)
D["appres"] = fd.honest_appres(D); D["realism"] = fd.honest_realism(D)
er = fd._exact_res(); D["res"] = D["k"].map(dict(zip(er.k, er.exact_resolution)))
EXP = D.origin.isin(["X-ray", "cryo-EM"])
def violin(ax, i, v, col, w0=0.4):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3: return
    k = gaussian_kde(v); g = np.linspace(v.min(), v.max(), 120); w = k(g); w = w / w.max() * w0
    ax.fill_betweenx(g, i - w, i + w, facecolor=col, edgecolor="k", lw=0.5, alpha=0.9); ax.hlines(np.median(v), i - w0, i + w0, color="k", lw=1.4)

# ---------- S1 ----------
E = D[EXP & D.res.notna()]; r1 = spearmanr(E.res, E.realism)[0]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for g in ["X-ray", "cryo-EM"]:
    s = E[E.origin == g]; ax.scatter(s.res, s.realism, s=8, c=ST[g][0], alpha=0.45, edgecolors="none", label=f"{g} (n={len(s)})")
bins = np.arange(0.5, 5.01, 0.5); mid = (bins[:-1] + bins[1:]) / 2; med = [E[(E.res >= a) & (E.res < b)].realism.median() for a, b in zip(bins[:-1], bins[1:])]
ok = [np.isfinite(m) and ((E.res >= a) & (E.res < b)).sum() >= 10 for m, a, b in zip(med, bins[:-1], bins[1:])]
ax.plot(mid[ok], np.array(med)[ok], "k-o", ms=4, lw=1.6, label="binned median")
ax.set_xlabel("deposited resolution (Å)"); ax.set_ylabel("realism score"); ax.set_ylim(0, 102); ax.legend(fontsize=7, loc="lower left")
ax.set_title(f"Realism is independent of resolution for experimental structures\n(Spearman {r1:+.2f} with deposited resolution)", fontsize=8.5)
fs.savepanel(fig, O, "SuppFig1_realism_vs_resolution")
rec("supp1_spearman_realism_deposited", r1, "Spearman realism vs deposited resolution, X-ray + cryo-EM")

# ---------- S2 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9)); OG = ["X-ray", "cryo-EM", "NMR", "AlphaFold2", "Boltz2", "AlphaFold3"]
for g in ["X-ray", "cryo-EM", "NMR", "AlphaFold2", "Boltz2", "AlphaFold3"]:
    s = D[D.origin == g]; axs[0].scatter(s.appres, s.realism, s=8, c=ST[g][0], alpha=0.5, edgecolors="none", label=g)
axs[0].set_xlabel("apparent resolution (Å)"); axs[0].set_ylabel("realism score"); axs[0].set_ylim(0, 102); axs[0].legend(fontsize=6.5, ncol=2, loc="lower right")
axs[0].set_title("At any apparent resolution, predicted models\nscore far below experimental structures", fontsize=8.5)
B = D[(D.appres >= 1.8) & (D.appres <= 2.6)]
for i, g in enumerate(OG):
    v = B[B.origin == g].realism.values; violin(axs[1], i, v, ST[g][0])
    rec(f"supp2_band_median_{g}", float(np.median(v)), f"median realism, {g}, apparent resolution 1.8-2.6 A"); rec(f"supp2_band_n_{g}", len(v), f"n {g} in the 1.8-2.6 A band")
rec("supp2_band_median_experimental", float(B[B.origin.isin(["X-ray", "cryo-EM"])].realism.median()), "median realism, X-ray + cryo-EM, 1.8-2.6 A band")
axs[1].set_xticks(range(6)); axs[1].set_xticklabels(OG, rotation=20, fontsize=7.5); axs[1].set_ylabel("realism score"); axs[1].set_ylim(0, 102)
axs[1].set_title("Matched apparent resolution (1.8–2.6 Å):\nexperimental high, predictors low and method-dependent", fontsize=8.5)
fig.tight_layout(); fs.panel_letters(fig, axs); fs.savepanel(fig, O, "SuppFig2_ai_vs_experimental")

# ---------- S3: made by ms_supp3.py (coordinate-only N-Ca-C by phi/psi basin) ----------

# ---------- S4: selection robustness (selection inside folds, same as the deployed design) ----------
from sklearn.model_selection import GroupKFold
tr, y, grp, cand = WT.appres_training(); gkf = list(GroupKFold(5).split(tr, y, grp))
def cv_select(th):
    p = np.full(len(y), np.nan)
    for a, b in gkf:
        s, _, _ = WT.select(tr, y, cand, a, th)
        if not s: return np.nan
        p[b] = WT.predict(WT.make_appres().fit(WT.Xof(tr.iloc[a], s), y[a]), WT.Xof(tr.iloc[b], s))
    return mean_absolute_error(y, p)
def cv_fixed(cols):
    p = np.full(len(y), np.nan)
    for a, b in gkf: p[b] = WT.predict(WT.make_appres().fit(WT.Xof(tr.iloc[a], cols), y[a]), WT.Xof(tr.iloc[b], cols))
    return mean_absolute_error(y, p)
THR = [0.1, 0.2, 0.3, 0.4, 0.5]; sw = []
for t in THR:
    n_all = len(WT.select(tr, y, cand, np.arange(len(y)), t)[0]); m = cv_select(t); sw.append((t, n_all, m))
    rec(f"supp4a_mae_thr{t}", m, f"grouped-CV MAE, selection threshold {t} (inside folds)", "A"); rec(f"supp4a_n_thr{t}", n_all, f"descriptors selected at threshold {t} (all data)")
S = np.array(sw, float); fig, a1 = plt.subplots(figsize=(4.3, 3.6)); a2 = a1.twinx()
a1.plot(S[:, 0], S[:, 2], "o-", color="#2166ac"); a1.set_ylabel("grouped-CV MAE (Å)", color="#2166ac")
a2.plot(S[:, 0], S[:, 1], "s--", color="#999999"); a2.set_ylabel("# descriptors kept", color="#777777")
a1.axvline(0.3, color="crimson", ls=":", lw=1.3); a1.set_xlabel("|Spearman with resolution| threshold"); a1.set_title("threshold robustness")
fs.savepanel(fig, O, "SuppFig4a_threshold_sweep")
SF = whatres.SCALE_FEATURES; _, geom, _ = WT.select(tr, y, cand, np.arange(len(y)))
sel_mae = cv_select(0.3); rng = np.random.default_rng(0); rnd = [cv_fixed(list(rng.choice(geom, len(SF), replace=False))) for _ in range(20)]
fig, ax = plt.subplots(figsize=(4.3, 3.6))
ax.hist(rnd, bins=10, color="#cccccc", edgecolor="k", label=f"random {len(SF)} descriptors (×20)")
ax.axvline(sel_mae, color="#2166ac", lw=2.2, label=f"resolution-selected (MAE {sel_mae:.2f})")
ax.set_xlabel("grouped-CV MAE (Å)"); ax.set_ylabel("count"); ax.legend(fontsize=7); ax.set_title("random-subset control")
fs.savepanel(fig, O, "SuppFig4b_random_control")
rec("supp4b_selected_mae", sel_mae, "grouped-CV MAE, resolution-selected descriptors (selection inside folds)", "A")
rec("supp4b_random_n", len(SF), "random draw size"); rec("supp4b_random_min", float(min(rnd)), "best random draw MAE", "A"); rec("supp4b_random_max", float(max(rnd)), "worst random draw MAE", "A")
rec("supp4b_random_median", float(np.median(rnd)), "median random draw MAE", "A"); rec("supp4b_selected_beats", int(sum(r > sel_mae for r in rnd)), "random draws the selected set beats (of 20)")

# ---------- S5: era ----------
SP = os.path.join(AN, "era_before_after_1995")
fm = pd.read_csv(os.path.join(SP, "era_fm.tsv"), sep="\t"); fm["id"] = fm["id"].astype(str).str.lower()
yr = pd.read_csv(os.path.join(SP, "era_years.tsv"), sep="\t"); yr["id"] = yr["id"].astype(str).str.lower()
fm = fm.merge(yr, on="id", how="left"); fm["appres"] = fd.honest_appres(fm); fm["realism"] = fd.honest_realism(fm)
rec("supp5_n_in_training", int(fm.id.isin(set(tr["id"])).sum()), "era structures that are also apparent-resolution training structures")
fm["era"] = np.where(fm.year < 1995, "pre-1995", "1995+"); a_, b_ = fm[fm.era == "pre-1995"], fm[fm.era == "1995+"]
M = joblib.load(os.path.join(AN, "models", "over_regularisation_model.joblib")); PAN, CO = M["panel"], M["appres_resid_coef"]
R = pd.DataFrame({c: pd.to_numeric(fm[c], errors="coerce").values - whatres.resid_expected(CO[c], fm["appres"].values) for c in PAN}).fillna(0.0)
fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.1)); PRE, POST = "#6a51a3", "#2ca02c"
for ax, col, lab in [(axs[0], "appres", "apparent resolution (Å)"), (axs[1], "realism", "realism score (0–100)")]:
    for i, (g, c) in enumerate([(a_, PRE), (b_, POST)]):
        violin(ax, i, g[col], c, 0.36); ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(g)), g[col], s=9, c="k", alpha=0.35, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"pre-1995\n(n={len(a_)})", f"1995+\n(n={len(b_)})"]); ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.3)
pa, pr = mannwhitneyu(a_.appres, b_.appres)[1], mannwhitneyu(a_.realism, b_.realism)[1]
la, lr = levene(a_.appres, b_.appres)[1], levene(a_.realism, b_.realism)[1]; ry = spearmanr(fm.year, fm.realism)[0]
axs[0].set_title(f"apparent resolution\nmedian {a_.appres.median():.2f} vs {b_.appres.median():.2f} Å, p={pa:.2f}", fontsize=9)
axs[1].set_title(f"realism\nmedian {a_.realism.median():.1f} vs {b_.realism.median():.1f}, p={pr:.2g}; SD {a_.realism.std():.1f} vs {b_.realism.std():.1f}", fontsize=9)
rows = []
for c in PAN:
    try: p = mannwhitneyu(R.loc[fm.era == "pre-1995", c], R.loc[fm.era == "1995+", c])[1]
    except Exception: p = np.nan
    rows.append((c, float(R.loc[fm.era == "pre-1995", c].median() - R.loc[fm.era == "1995+", c].median()), p))
dd = pd.DataFrame(rows, columns=["desc", "eff", "p"]).sort_values("p").head(10)[::-1]
axs[2].barh(range(len(dd)), dd.eff, color="#b2182b"); axs[2].set_yticks(range(len(dd)))
axs[2].set_yticklabels([c.replace("qc_", "").replace("_mean_score", "·mean").replace("_sd_pctile", "·spread").replace("_frac_extreme", "·outliers") for c in dd.desc], fontsize=7.5)
axs[2].axvline(0, color="k", lw=0.6); axs[2].set_xlabel("pre-1995 minus 1995+\n(resolution-corrected descriptor median)"); axs[2].set_title("descriptors carrying the era effect", fontsize=9)
fig.suptitle(f"Refinement era at fixed resolution (2.0–2.5 Å X-ray, deposited {int(fm.year.min())}–{int(fm.year.max())})", fontsize=10.5, y=1.02)
fig.tight_layout(); fs.panel_letters(fig, axs); fs.savepanel(fig, O, "SuppFig5_era_before_after_1995")
dd.to_csv(os.path.join(O, "SuppFig5_era_descriptor_effects.tsv"), sep="\t", index=False)
for k_, v_, d_ in [("supp5_n_pre", len(a_), "pre-1995 structures"), ("supp5_n_post", len(b_), "1995+ structures"),
                   ("supp5_year_min", int(fm.year.min()), "earliest deposition year"), ("supp5_year_max", int(fm.year.max()), "latest deposition year"),
                   ("supp5_appres_median_pre", a_.appres.median(), "median apparent resolution pre-1995"), ("supp5_appres_median_post", b_.appres.median(), "median apparent resolution 1995+"),
                   ("supp5_appres_p", pa, "Mann-Whitney p, apparent resolution"), ("supp5_appres_levene_p", la, "Levene p, apparent resolution"),
                   ("supp5_realism_median_pre", a_.realism.median(), "median realism pre-1995"), ("supp5_realism_median_post", b_.realism.median(), "median realism 1995+"),
                   ("supp5_realism_p", pr, "Mann-Whitney p, realism"), ("supp5_realism_sd_pre", a_.realism.std(), "realism SD pre-1995"),
                   ("supp5_realism_sd_post", b_.realism.std(), "realism SD 1995+"), ("supp5_realism_levene_p", lr, "Levene p, realism"),
                   ("supp5_spearman_realism_year", ry, "Spearman realism vs deposition year")]:
    rec(k_, v_, d_)
rec("supp5_top_effects", ";".join(dd.desc[::-1].head(5)), "descriptors with the strongest era effect")

# ---------- S6: size ----------
fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.5))
for g in ["X-ray", "cryo-EM", "AlphaFold2", "AlphaFold3", "Boltz2"]:
    s = D[D.origin == g]; axs[0].scatter(pd.to_numeric(s.qc_n_residues, errors="coerce"), s.realism, s=6, c=ST[g][0], alpha=0.5, edgecolors="none", label=g)
axs[0].set_xscale("log"); axs[0].set_xlabel("protein length (residues, log)"); axs[0].set_ylabel("realism score"); axs[0].set_ylim(0, 102); axs[0].legend(fontsize=6.5, ncol=2)
axs[0].set_title("realism vs length (benchmark)", fontsize=9)
# (b) subsampling within ten large X-ray structures, scored by the fold models that never saw them
IDS = ["22wz", "22xh", "22zw", "24io", "8weh", "9c6a", "9nyu", "9vv4", "9yz5", "9z0y"]; NS = [30, 50, 100, 150, 200, 300, 400, 500]; NREP = 100
resp = os.path.join(DATA, "size_reliability_residues.tsv")
if not os.path.exists(resp):
    rep = os.path.join(DATA, "bench_quality_report.tsv.gz")
    keep = []
    for ch in pd.read_csv(rep, sep="\t", chunksize=200000, low_memory=False):
        keep.append(ch[ch["pdb"].astype(str).str.lower().str.replace(r'\.(pdb|ent)$', '', regex=True).isin(IDS)])
    pd.concat(keep).to_csv(resp, sep="\t", index=False)
RS = pd.read_csv(resp, sep="\t", low_memory=False); RS["k"] = RS["pdb"].astype(str).str.lower().str.replace(r'\.(pdb|ent)$', '', regex=True)
FMx = pd.read_csv(os.path.join(AN, "feature_matrix.tsv"), sep="\t"); FMx["id"] = FMx["id"].str.lower()
oa = pd.read_csv(os.path.join(AN, "models", "apparent_resolution_oof.tsv"), sep="\t").set_index("id")
orl = pd.read_csv(os.path.join(AN, "models", "realism_oof.tsv"), sep="\t").set_index("id")
ALLr = pd.read_csv(os.path.join(AN, "feature_matrix.tsv"), sep="\t"); ALLr["id"] = ALLr["id"].str.lower()
ALLr = ALLr[ALLr.id.isin(orl.index)].copy(); AIr = pd.read_csv(os.path.join(AN, "ai_feature_matrix.tsv"), sep="\t"); AIr = AIr[AIr.id.isin(orl.index)].copy()
ALLr = pd.concat([ALLr, AIr], ignore_index=True); ALLr["cls"] = ALLr.id.map(orl.cls); ALLr["appres"] = ALLr.id.map(orl.appres); ALLr["fold"] = ALLr.id.map(orl.fold)
PANr = joblib.load(os.path.join(AN, "models", "over_regularisation_model.joblib"))["panel"]
rng = np.random.default_rng(0); res_rows = []
for pid in IDS:
    fa, frl = int(oa.loc[pid, "fold"]), int(orl.loc[pid, "fold"])
    m_ap, s_ap = WT.fit_appres(tr, y, cand, np.where(oa.loc[tr["id"], "fold"].values != fa)[0])
    clf_f, CO_f = WT.fit_realism(ALLr[ALLr.fold != frl], PANr)
    base = FMx[FMx.id == pid].iloc[0]; res_ = RS[RS.k == pid]; n_res = len(res_)
    def score(sub):
        q = aggregate_quality_df(sub.drop(columns=["k"])).iloc[0]; row = base.copy()
        for c, v in q.items():
            if c in row.index and c != "id": row[c] = v
        df = pd.DataFrame([row]); df["appres"] = WT.predict(m_ap, WT.Xof(df, s_ap))
        return 100 * (1 - clf_f.predict_proba(WT.resid(df, CO_f, PANr))[:, 1][0])
    full = score(res_)
    for N in NS:
        if N > n_res: continue
        v = [score(res_.iloc[rng.choice(n_res, N, replace=False)]) for _ in range(NREP)]
        res_rows.append({"pdb": pid, "n_res": n_res, "N": N, "full": full, "mean": float(np.mean(v)), "sd": float(np.std(v))})
SS = pd.DataFrame(res_rows); SS.to_csv(os.path.join(O, "SuppFig6_subsample_realism.tsv"), sep="\t", index=False)
for pid, g in SS.groupby("pdb"): axs[1].plot(g.N, g.sd, color="#bbbbbb", lw=1)
mS = SS.groupby("N").sd.mean(); axs[1].plot(mS.index, mS.values, "o-", color="#d62728", lw=2, label="mean over 10 structures")
axs[1].plot(mS.index, mS.loc[100] * np.sqrt(100 / mS.index.values), "k--", lw=1, label="~ 1/√N guide")
axs[1].set_xlabel("residues summarised (subset size)"); axs[1].set_ylabel("realism score SD (sampling only)"); axs[1].legend(fontsize=7)
axs[1].set_title(f"precision improves with size\n(±{mS.loc[50]:.0f} at N=50, ±{mS.loc[400]:.0f} at N=400)", fontsize=9)
for N in NS: rec(f"supp6_sd_N{N}", float(mS.loc[N]), f"mean realism SD over subsets of {N} residues (10 structures)")
rec("supp6_nres_min", int(SS.n_res.min()), "smallest of the ten structures (residues)"); rec("supp6_nres_max", int(SS.n_res.max()), "largest of the ten structures (residues)")
rec("supp6_max_abs_bias", float((SS["mean"] - SS["full"]).abs().max()), "largest |mean subset realism - full realism|")
# (c) long side chains: composition of the benchmark experimental structures
bench = pd.read_csv(os.path.join(DATA, "bench_quality_report.tsv.gz"), sep="\t", usecols=["pdb", "aa"])
f3 = bench.aa.isin(list("RKQEM")).mean(); f4 = bench.aa.isin(list("RK")).mean(); Ln = np.array([30, 50, 100, 150, 200, 300, 400, 500])
axs[2].plot(Ln, Ln, "k-", lw=1, label="backbone (all residues)")
axs[2].plot(Ln, f3 * Ln, "o-", color="#d62728", label=f"χ3-bearing (~{100*f3:.0f}%: R,K,Q,E,M)"); axs[2].plot(Ln, f4 * Ln, "s-", color="#e08214", label=f"χ4-bearing (~{100*f4:.0f}%: R,K)")
axs[2].axhline(10, color="0.4", ls=":", lw=0.8); axs[2].set_xlabel("protein length (residues)"); axs[2].set_ylabel("usable observations for the descriptor"); axs[2].legend(fontsize=7)
axs[2].set_title("long side chains are the limiting sample\n(χ4 only in Arg/Lys)", fontsize=9)
rec("supp6_frac_chi3", f3, "fraction of residues with chi3 (R,K,Q,E,M), benchmark"); rec("supp6_frac_chi4", f4, "fraction of residues with chi4 (R,K), benchmark")
fig.tight_layout(); fs.panel_letters(fig, axs); fs.savepanel(fig, O, "SuppFig6_size_reliability")
print(f"[ms_supp] S1 rho {r1:+.3f} | S4 sel {sel_mae:.3f} random {min(rnd):.3f}-{max(rnd):.3f} | era realism {a_.realism.median():.1f} vs {b_.realism.median():.1f} | S6 SD50 {mS.loc[50]:.1f} SD400 {mS.loc[400]:.1f} -> {O}")
