#!/usr/bin/env python3.14
"""Manuscript Figure 3: apparent resolution.
a Fig3a_scaling_descriptors_ranked   resolution-scaling descriptors ranked by |Spearman with resolution|
b Fig3b_descriptors_vs_resolution    three scaling descriptors vs deposited resolution (monotone fit),
                                     AlphaFold3/Boltz2 medians overlaid at their apparent resolution
c Fig3c_training_correlation         grouped-CV (MMseqs2 clusters) out-of-fold vs deposited resolution
d Fig3d_independent_set              independent set, no shared sequence cluster with training
e Fig3e_by_method                    apparent resolution by method (deployed model, as in the paper)
f Fig3f_pLDDT_vs_apparent_resolution predictor confidence vs apparent resolution, all predicted models
Also writes data/appres_training_oof.tsv (used for the MolProbity comparison in ms_misc.py).
"""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, numpy as np, pandas as pd
from ms_common import fs, fd, whatres, outdir, recorder, AN, DATA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, spearmanr
from scipy.optimize import curve_fit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

rec = recorder(__file__); O = outdir("Figure3")
tr, y, grp, pool, rho_res = fd.appres_training()
SF = whatres.SCALE_FEATURES
XRAY_C = fd.ORIGIN_STYLE["X-ray"][0]; EM_C = fd.ORIGIN_STYLE["cryo-EM"][0]
mx = tr["category"].values == "xray"
slope = lambda a, b: float(np.polyfit(a, b, 1)[0])
rep = fd.seq_cluster_map()

# ---- c: training grouped-CV out-of-fold ----
_o = pd.read_csv(os.path.join(AN, "models", "apparent_resolution_oof.tsv"), sep="\t")   # canonical, from build_apparent_resolution.py
_m = dict(zip(_o["id"].astype(str).str.lower(), _o["oof"]))
oof = np.array([_m[i] for i in tr["id"].str.lower()])   # KeyError = training cohort mismatch: hard error
mae, r2, rho = mean_absolute_error(y, oof), r2_score(y, oof), spearmanr(y, oof)[0]
figC, ax = plt.subplots(figsize=(4.2, 4.0))
ax.scatter(y[mx], oof[mx], s=10, alpha=0.5, c=XRAY_C, label=f"X-ray (n={int(mx.sum())})")
ax.scatter(y[~mx], oof[~mx], s=10, alpha=0.5, c=EM_C, label=f"cryo-EM (n={int((~mx).sum())})")
lim = [0.5, max(5.0, float(y.max()) + 0.2)]; ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("deposited resolution (Å)"); ax.set_ylabel("apparent resolution (Å), grouped-CV out-of-fold")
ax.set_title(f"training set (grouped-CV out-of-fold)\nMAE {mae:.2f} Å · R² {r2:.2f} · Spearman {rho:.2f}")
ax.legend(loc="upper left"); ax.set_aspect("equal")
fs.savepanel(figC, O, "Fig3c_training_correlation")
rec("fig3_n_train", len(tr), "training structures (X-ray + cryo-EM with deposited resolution)")
rec("fig3_n_train_xray", int(mx.sum()), "training X-ray structures"); rec("fig3_n_train_cryoem", int((~mx).sum()), "training cryo-EM structures")
rec("fig3_train_mae", mae, "training grouped-CV (MMseqs2) MAE", "A"); rec("fig3_train_r2", r2, "training grouped-CV R2")
rec("fig3_train_spearman", rho, "training grouped-CV Spearman"); rec("fig3_train_slope", slope(y, oof), "training OOF slope of apparent vs deposited")
for lab, m in [("xray", mx), ("cryoem", ~mx)]:
    rec(f"fig3_train_mae_{lab}", mean_absolute_error(y[m], oof[m]), f"training grouped-CV MAE, {lab} only", "A")
    rec(f"fig3_train_r2_{lab}", r2_score(y[m], oof[m]), f"training grouped-CV R2, {lab} only")
rec("fig3_train_n_no_cluster", int(sum(1 for i in tr["id"].str.upper() if i not in rep)), "training structures without an MMseqs2 cluster (singletons)")
pd.DataFrame({"id": tr["id"], "category": tr["category"], "res": y, "oof": oof}).to_csv(os.path.join(DATA, "appres_training_oof.tsv"), sep="\t", index=False)

# ---- d: independent set ----
vids = pd.read_csv(os.path.join(AN, "dms_hero", "val_ids.tsv"), sep="\t", header=None)
vf = pd.read_csv(os.path.join(AN, "dms_hero", "val_feature_matrix.tsv"), sep="\t"); vf["k"] = vf["id"].astype(str).str.lower()
vr = pd.read_csv(os.path.join(AN, "dms_hero", "val_resolution.tsv"), sep="\t"); vr["k"] = vr["id"].astype(str).str.lower()
ext = vf.merge(vr[["k", "exact_resolution", "method"]], on="k", how="inner")
n_feat = len(ext)
tc = set(grp); in_tr = ext.k.str.upper().map(lambda i: rep.get(i, i)).isin(tc)
ext = ext[~in_tr].copy(); n_removed = int(in_tr.sum())
ext["appres"] = fd.honest_appres(ext); ext["res"] = pd.to_numeric(ext.exact_resolution, errors="coerce")
ext = ext[ext.res.notna() & ext.method.str.contains("X-RAY|ELECTRON", case=False, na=False)].reset_index(drop=True)
exr = ext.method.str.contains("X-RAY", case=False, na=False).values
emae, er2, erho = mean_absolute_error(ext.res, ext.appres), r2_score(ext.res, ext.appres), spearmanr(ext.res, ext.appres)[0]
figD, ax = plt.subplots(figsize=(4.2, 4.0))
ax.scatter(ext.res[exr], ext.appres[exr], s=10, alpha=0.5, c=XRAY_C, label=f"X-ray (n={int(exr.sum())})")
ax.scatter(ext.res[~exr], ext.appres[~exr], s=10, alpha=0.5, c=EM_C, label=f"cryo-EM (n={int((~exr).sum())})")
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("deposited resolution (Å)"); ax.set_ylabel("apparent resolution (Å)")
ax.set_title(f"independent set (n={len(ext)}, no training overlap)\nMAE {emae:.2f} Å · R² {er2:.2f} · Spearman {erho:.2f}")
ax.legend(loc="upper left"); ax.set_aspect("equal")
fs.savepanel(figD, O, "Fig3d_independent_set")
# per-candidate status of the independent set (Supplementary Table 2, Validation_IDs sheet)
_feat = set(vf.merge(vr[["k"]], on="k", how="inner").k.str.upper()); _used = set(ext.k.str.upper())
_vr = vr.assign(K=vr.k.str.upper()).drop_duplicates("K").set_index("K")
_rows = []
for _pid, _meth in zip(vids[0].astype(str).str.upper(), vids[1]):
    _cl = rep.get(_pid, _pid)
    if _pid not in _feat: _stt = "excluded: no feature row"
    elif _cl in tc: _stt = "excluded: MMseqs2 cluster shared with training"
    elif _pid in _used: _stt = "used in Fig 3d" + ("" if _pid in rep else " (singleton: no cluster, independence unverified)")
    else: _stt = "excluded: method or resolution"
    _rows.append([_pid, _meth, _vr.exact_resolution.get(_pid), "yes" if _pid in _feat else "no", _cl,
                  "yes" if _stt.startswith("used") else "no", _stt])
_VS = pd.DataFrame(_rows, columns=["PDB ID", "Method", "Deposited resolution (Å)", "Has feature row", "MMseqs2 cluster", "Used in Fig 3d", "Status / exclusion reason"])
assert (_VS["Used in Fig 3d"] == "yes").sum() == len(ext), "validation status disagrees with the Fig. 3d set"
_VS.to_csv(os.path.join(DATA, "validation_status.tsv"), sep="\t", index=False)
rec("val_n_candidates", len(vids), "independent-set candidates (val_ids.tsv)")
rec("val_n_with_descriptors", n_feat, "candidates with computed descriptors and a deposited resolution")
rec("val_n_lacking_descriptors", len(vids) - n_feat, "candidates without computed descriptors")
rec("val_n_removed_shared_cluster", n_removed, "candidates removed for sharing an MMseqs2 cluster with training")
rec("fig3_ext_n", len(ext), "independent set size"); rec("fig3_ext_n_xray", int(exr.sum()), "independent set X-ray")
rec("fig3_ext_n_cryoem", int((~exr).sum()), "independent set cryo-EM")
rec("fig3_ext_res_min", float(ext.res.min()), "independent set lowest deposited resolution", "A")
rec("fig3_ext_res_max", float(ext.res.max()), "independent set highest deposited resolution", "A")
rec("fig3_ext_mae", emae, "independent set MAE", "A"); rec("fig3_ext_r2", er2, "independent set R2")
rec("fig3_ext_spearman", erho, "independent set Spearman"); rec("fig3_ext_slope", slope(ext.res, ext.appres), "independent set slope")
for lab, m in [("xray", exr), ("cryoem", ~exr)]:
    rec(f"fig3_ext_mae_{lab}", mean_absolute_error(ext.res[m], ext.appres[m]), f"independent set MAE, {lab}", "A")
    rec(f"fig3_ext_spearman_{lab}", spearmanr(ext.res[m], ext.appres[m])[0], f"independent set Spearman, {lab}")
nocl = ~ext.k.str.upper().isin(set(rep))
rec("fig3_ext_n_no_cluster", int(nocl.sum()), "independent-set structures without an MMseqs2 cluster")
rec("fig3_ext_mae_excl_no_cluster", mean_absolute_error(ext.res[~nocl], ext.appres[~nocl]), "independent set MAE excluding unclustered", "A")

# ---- a: ranked descriptors ----
def fam(c):
    if c.startswith("dist_"): return "atom-type distance"
    if c.startswith("surf_"): return "surface/cavity"
    if c.startswith("ws_clash") or "clashscore" in c: return "clashscore"
    for k, n in [("bond_", "bond length"), ("angle_", "bond angle"), ("cb_deviation", "Cβ deviation"),
                 ("ca_chirality", "Cα chirality"), ("rotamer", "rotamer outlier"), ("chi", "side-chain rotamer"),
                 ("cablam", "CaBLAM"), ("peptide_planarity", "peptide planarity"), ("omega", "omega"),
                 ("rama", "Ramachandran"), ("phi", "phi"), ("psi", "psi"), ("cis", "cis-peptide")]:
        if k in c: return n
    return "other geometry"
FAMCOL = {"atom-type distance": "#4c72b0", "bond length": "#dd8452", "rotamer outlier": "#c44e52",
          "Cα chirality": "#8172b3", "Ramachandran": "#937860", "bond angle": "#da8bc3", "clashscore": "#55a868",
          "psi": "#64b5cd", "other geometry": "#8c8c8c"}
rho_all = {c: rho_res.get(c, spearmanr(pd.to_numeric(tr[c], errors="coerce"), y, nan_policy="omit")[0]) for c in SF}
rank = sorted(SF, key=lambda c: abs(rho_all[c]), reverse=True); top = rank[:15][::-1]
nice = lambda c: (c.replace("qc_", "").replace("_mean_score", " (mean)").replace("_sd_pctile", " spread")
                  .replace("_frac_extreme", " outliers").replace("_mean_pctile", "").replace("dist_", "")
                  .replace("_mid", " dist").replace("_short", " dist(short)"))
figA, ax = plt.subplots(figsize=(4.6, 4.2))
ax.barh(range(len(top)), [abs(rho_all[c]) for c in top], color=[FAMCOL.get(fam(c), "#8c8c8c") for c in top])
ax.set_yticks(range(len(top))); ax.set_yticklabels([nice(c) for c in top], fontsize=6.5)
ax.set_xlabel("|Spearman with resolution|"); ax.set_title(f"resolution-scaling descriptors (top 15 of {len(SF)})")
fp = [f for f in FAMCOL if any(fam(c) == f for c in top)]
ax.legend(handles=[Patch(color=FAMCOL[f], label=f) for f in fp], fontsize=6, loc="lower right")
fs.savepanel(figA, O, "Fig3a_scaling_descriptors_ranked")
rec("n_appres_features", len(SF), "resolution-scaling descriptors in the deployed apparent-resolution model")
fams = pd.Series([fam(c) for c in SF]).value_counts()
for f, n in fams.items(): rec(f"appres_family_n_{f.replace(' ', '_')}", int(n), f"apparent-resolution descriptors in family '{f}'")
pd.DataFrame({"descriptor": rank, "family": [fam(c) for c in rank], "spearman_with_resolution": [rho_all[c] for c in rank]}) \
  .to_csv(os.path.join(O, "Fig3a_scaling_descriptors_ranked.tsv"), sep="\t", index=False)

# ---- origin dataset with apparent resolution (deployed model) ----
D, _ = fd.origin_dataset(); D["appres"] = fd.honest_appres(D)
er = fd._exact_res(); D["k"] = D["id"].astype(str).str.lower().str.replace(r'\.(pdb|ent)$', '', regex=True)
D["res"] = D["k"].map(dict(zip(er.k, er.exact_resolution)))
ai = D[D.origin.isin(["AlphaFold3", "Boltz2"])]

# ---- b: descriptors vs deposited resolution, monotone fit ----
def expsat(x, a, b, k): return a + b * np.exp(-k * x)
SHOW = [("qc_rotamer_outlier_mean_pctile", "rotamer-outlier level"), ("qc_bond_C_O_sd_pctile", "C=O bond-length spread"),
        ("dist_Calif-Calif_mid", "aliphatic–aliphatic distance")]
figB, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
for ax, (c, lab) in zip(axs, SHOW):
    xr = tr["res"].values; v = pd.to_numeric(tr[c], errors="coerce").values; ok = np.isfinite(v)
    ax.scatter(xr, v, s=6, alpha=0.25, color="#999999", label="experimental")
    xs = np.linspace(xr.min(), xr.max(), 80)
    try:
        b0 = (np.mean(v[ok][xr[ok] < np.percentile(xr, 25)]) - np.mean(v[ok][xr[ok] > np.percentile(xr, 75)])) or 1.0
        p, _ = curve_fit(expsat, xr[ok], v[ok], p0=[np.median(v[ok]), b0, 1.0], bounds=([-1e9, -1e9, 0.01], [1e9, 1e9, 10]), maxfev=20000)
        ax.plot(xs, expsat(xs, *p), color="k", lw=1.6)
    except Exception:
        ax.plot(xs, np.polyval(np.polyfit(xr[ok], v[ok], 1), xs), color="k", lw=1.6)
    for g in ["AlphaFold3", "Boltz2"]:
        s = ai[ai.origin == g]; col, mk = fd.ORIGIN_STYLE[g]
        ax.errorbar(s["appres"].median(), pd.to_numeric(s[c], errors="coerce").median(), fmt=mk, color=col, ms=7, mec="k", mew=0.4, label=g)
    ax.set_xlabel("resolution (Å)"); ax.set_title(lab, fontsize=8.5)
    rec(f"fig3b_spearman_{c}", spearmanr(v[ok], xr[ok])[0], f"Spearman of {lab} with deposited resolution (training)")
axs[0].set_ylabel("descriptor value"); axs[2].legend(fontsize=6.5, loc="best")
figB.tight_layout(); fs.savepanel(figB, O, "Fig3b_descriptors_vs_resolution")

# ---- e: apparent resolution by method (deployed model, as in the paper) ----
xr_ = D[D.origin == "X-ray"]
groups = [("X-ray ≤2Å", xr_[xr_.res <= 2.0]["appres"].dropna().values), ("X-ray >2.5Å", xr_[xr_.res > 2.5]["appres"].dropna().values),
          ("cryo-EM", D[D.origin == "cryo-EM"]["appres"].dropna().values), ("NMR", D[D.origin == "NMR"]["appres"].dropna().values),
          ("AF3", D[D.origin == "AlphaFold3"]["appres"].dropna().values), ("Boltz2", D[D.origin == "Boltz2"]["appres"].dropna().values)]
figE, ax = plt.subplots(figsize=(5.2, 3.6))
for i, (lab, v) in enumerate(groups):
    k = gaussian_kde(v); gr = np.linspace(v.min(), v.max(), 120); w = k(gr); w = w / w.max() * 0.38
    ax.fill_betweenx(gr, i - w, i + w, facecolor="#6baed6" if "X-ray" in lab or lab == "cryo-EM" else ("#bbbbbb" if lab == "NMR" else "#e08214"),
                     edgecolor="k", lw=0.5, alpha=0.9)
    ax.hlines(np.median(v), i - 0.38, i + 0.38, color="k", lw=1.4)
    rec(f"fig3e_median_{lab}", float(np.median(v)), f"median apparent resolution, {lab} (deployed model)", "A")
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], rotation=20, fontsize=7.5)
ax.set_ylabel("apparent resolution (Å)"); ax.set_title("apparent resolution by method")
fs.savepanel(figE, O, "Fig3e_by_method")
xa = ai["appres"].values
for g in ["AlphaFold3", "Boltz2"]:
    v = ai[ai.origin == g]["appres"].values
    rec(f"appres_mean_{g}", float(v.mean()), f"mean apparent resolution, {g}", "A"); rec(f"appres_median_{g}", float(np.median(v)), f"median apparent resolution, {g}", "A")
rec("appres_ai_min", float(xa.min()), "lowest apparent resolution, AF3 + Boltz2", "A"); rec("appres_ai_max", float(xa.max()), "highest apparent resolution, AF3 + Boltz2", "A")
rec("appres_ai_p05", float(np.percentile(xa, 5)), "5th percentile apparent resolution, AF3 + Boltz2", "A")
rec("appres_ai_p95", float(np.percentile(xa, 95)), "95th percentile apparent resolution, AF3 + Boltz2", "A")
# the paper's "~1.2 A for the highest-resolution X-ray to ~3.9 A for the lowest-resolution cryo-EM": deposited-resolution extremes
xs_ = xr_.dropna(subset=["res"]); em_ = D[(D.origin == "cryo-EM")].dropna(subset=["res"])
rec("fig3e_median_xray_res_le_1.5", float(xs_[xs_.res <= 1.5]["appres"].median()), "median apparent resolution (out-of-fold), X-ray deposited <= 1.5 A (the paper's '~1.2 A')", "A")
rec("fig3e_median_cryoem_res_ge_3.5", float(em_[em_.res >= 3.5]["appres"].median()), "median apparent resolution (out-of-fold), cryo-EM deposited >= 3.5 A (the paper's '~3.9 A')", "A")

# ---- f: pLDDT vs apparent resolution, all predicted models ----
P = D[D.origin.isin(["AlphaFold2", "AlphaFold3", "Boltz2"])].copy(); P["plddt"] = pd.to_numeric(P["bf_ca_mean"], errors="coerce")
P = P.dropna(subset=["plddt"]); r_all = spearmanr(P.plddt, P.appres)[0]
figF, ax = plt.subplots(figsize=(4.0, 4.0))
for g in ["AlphaFold2", "AlphaFold3", "Boltz2"]:
    s = P[P.origin == g]; col, mk = fd.ORIGIN_STYLE[g]
    ax.scatter(s.plddt, s.appres, s=10, marker=mk, c=col, alpha=0.55, edgecolors="none", label=g)
    rec(f"fig3f_spearman_plddt_appres_{g}", spearmanr(s.plddt, s.appres)[0], f"Spearman pLDDT vs apparent resolution, {g}")
ax.set_xlabel("mean pLDDT"); ax.set_ylabel("apparent resolution (Å)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
ax.set_title(f"confidence vs apparent resolution\n(Spearman {r_all:.2f})", fontsize=9)
fs.savepanel(figF, O, "Fig3f_pLDDT_vs_apparent_resolution")
rec("fig3f_spearman_plddt_appres", r_all, "Spearman pLDDT vs apparent resolution, all predicted models")
rec("fig3f_n_models", len(P), "predicted models with pLDDT")
print(f"[ms_fig3] train MAE {mae:.3f} R2 {r2:.3f} rho {rho:.3f} | ext n={len(ext)} MAE {emae:.3f} R2 {er2:.3f} rho {erho:.3f} | -> {O}")
