#!/usr/bin/env python3.14
"""Manuscript Figure 4: the realism axis.
a Fig4a_resolution_invariant   chi3, chi4, psi surprise (qc_*) and the raw mean backbone C=O length
                               (from coordinates, gemmi) by cohort
b Fig4b_residualisation_fit    four descriptors vs apparent resolution with the deployed expected-value fit
c Fig4c_top_contributors       signed logistic weights of the deployed realism classifier
d Fig4d_realism_by_method      realism by method of origin
e Fig4e_orthogonality          realism vs apparent resolution
f Fig4f_pLDDT_vs_realism       pLDDT vs realism and vs apparent resolution, AF3/Boltz2
g Fig4g_7dg7_overlay           molecular overlay (image, unchanged)
h Fig4h_7dg7_rotamers          AF3 vs experimental chi3/chi4 of Lys/Arg (coordinates), numbers in the title computed
"""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, shutil, numpy as np, pandas as pd, gemmi, joblib
from ms_common import fs, fd, whatres, outdir, recorder, realism, AN, DATA, MS
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr

rec = recorder(__file__); O = outdir("Figure4"); ST = fd.ORIGIN_STYLE
H = os.path.dirname(AN); BENCH = os.path.join(H, "benchmark")
D, _ = fd.origin_dataset(); D["appres"] = fd.honest_appres(D); D["realism"] = fd.honest_realism(D)   # no memorised values
er = fd._exact_res(); D["k"] = D["id"].astype(str).str.lower().str.replace(r'\.(pdb|ent)$', '', regex=True)
D["res"] = D["k"].map(dict(zip(er.k, er.exact_resolution)))
EXP = D.origin.isin(["X-ray", "cryo-EM"])

# ---------- raw backbone C=O length per structure, from coordinates ----------
AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
paths = {}
for lst in ["benchmark_existing.txt", "ai_list.txt"]:
    for p in open(os.path.join(AN, "refbuild", lst)).read().split():
        paths[os.path.splitext(os.path.basename(p))[0].lower()] = os.path.join(BENCH, p)
co_p = os.path.join(DATA, "co_bondlength.tsv")
co = pd.read_csv(co_p, sep="\t") if os.path.exists(co_p) else pd.DataFrame(columns=["k", "co_mean", "co_sd", "n"])
todo = [k for k in D.k if k not in set(co.k) and k in paths]
rows = []
for k in todo:
    st = gemmi.read_structure(paths[k]); d = []
    for ch in st[0]:
        for r in ch:
            if r.name not in AA: continue
            c, o = r.find_atom("C", "*"), r.find_atom("O", "*")
            if c and o: d.append(c.pos.dist(o.pos))
    if d: rows.append({"k": k, "co_mean": float(np.mean(d)), "co_sd": float(np.std(d)), "n": len(d)})
if rows:
    co = pd.concat([co, pd.DataFrame(rows)], ignore_index=True); co.to_csv(co_p, sep="\t", index=False)
D = D.merge(co[["k", "co_mean", "co_sd"]], on="k", how="left")
rec("fig4a_n_without_co", int(D.co_mean.isna().sum()), "structures without a computed C=O length (no file found)")

# ---------- a: resolution-invariant descriptors ----------
GR = [("X-ray ≤2Å", (D.origin == "X-ray") & (D.res <= 2.0), "#2e8540"), ("X-ray >2.5Å", (D.origin == "X-ray") & (D.res > 2.5), "#8cc97f"),
      ("cryo-EM", D.origin == "cryo-EM", ST["cryo-EM"][0]), ("AF3", D.origin == "AlphaFold3", ST["AlphaFold3"][0]),
      ("Boltz2", D.origin == "Boltz2", ST["Boltz2"][0])]
PA = [("qc_chi3_mean_score", "χ3 rotamer", "mean per-residue surprise (−log$_2$p)"), ("qc_chi4_mean_score", "χ4 rotamer", "mean per-residue surprise (−log$_2$p)"),
      ("qc_psi_mean_score", "ψ backbone", "mean per-residue surprise (−log$_2$p)"), ("co_mean", "C=O bond length", "mean C=O length (Å)")]
figA, axs = plt.subplots(1, 4, figsize=(15.5, 3.9))
for ax, (c, ttl, yl) in zip(axs, PA):
    for i, (lab, m, col) in enumerate(GR):
        v = pd.to_numeric(D.loc[m, c], errors="coerce").dropna().values
        if len(v) < 3: continue
        k = gaussian_kde(v); g = np.linspace(v.min(), v.max(), 150); w = k(g); w = w / w.max() * 0.38
        ax.fill_betweenx(g, i - w, i + w, facecolor=col, edgecolor="k", lw=0.4, alpha=0.95)
        ax.hlines(np.median(v), i - 0.38, i + 0.38, color="k", lw=1.4)
    if c == "co_mean": ax.axhline(np.median(pd.to_numeric(D.loc[EXP, c], errors="coerce").dropna()), color="k", ls=":", lw=0.9)
    ax.set_xticks(range(len(GR))); ax.set_xticklabels([g[0] for g in GR], rotation=20, fontsize=7.5)
    ax.set_title(ttl, fontsize=9); ax.set_ylabel(yl); ax.grid(axis="y", alpha=0.3)
figA.tight_layout(); fs.savepanel(figA, O, "Fig4a_resolution_invariant")
def cohen(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)); return (a.mean() - b.mean()) / s
AI = D.origin.isin(["AlphaFold3", "Boltz2"])
for c, nm in [("qc_chi3_mean_score", "chi3"), ("qc_chi4_mean_score", "chi4"), ("qc_psi_mean_score", "psi")]:
    rec(f"fig4a_median_{nm}_experimental", float(D.loc[EXP, c].median()), f"median {nm} surprise, X-ray + cryo-EM")
    for g in ["AlphaFold3", "Boltz2"]: rec(f"fig4a_median_{nm}_{g}", float(D.loc[D.origin == g, c].median()), f"median {nm} surprise, {g}")
    rec(f"fig4a_cohend_{nm}_AI_vs_exp", cohen(D.loc[AI, c], D.loc[EXP, c]), f"Cohen's d (AF3+Boltz2 minus X-ray+cryo-EM), {nm} surprise")
    rec(f"fig4a_cohend_{nm}_AF3_vs_exp", cohen(D.loc[D.origin == "AlphaFold3", c], D.loc[EXP, c]), f"Cohen's d (AF3 minus X-ray+cryo-EM), {nm} surprise")
for lab, m, _c in GR:
    rec(f"fig4a_median_co_{lab}", float(D.loc[m, "co_mean"].median()), f"median per-structure mean C=O length, {lab} (coordinates)", "A")
rec("fig4a_median_co_sd_AF3", float(D.loc[D.origin == "AlphaFold3", "co_sd"].median()), "median within-structure SD of C=O length, AF3", "A")
rec("fig4a_median_co_sd_xray_le2", float(D.loc[(D.origin == "X-ray") & (D.res <= 2.0), "co_sd"].median()), "median within-structure SD of C=O length, X-ray <= 2 A", "A")

# ---------- b: residualisation fit ----------
M = joblib.load(os.path.join(AN, "models", "over_regularisation_model.joblib")); PAN, CO, CLF = M["panel"], M["appres_resid_coef"], M["clf"]
expD = D[EXP].copy(); aiD = D[AI].copy()
SHOW = [("qc_ca_chirality_mean_score", "Cα-chirality surprise"), ("qc_bond_C_O_sd_pctile", "C=O bond-length spread"),
        ("qc_chi3_mean_score", "χ3 rotamer surprise"), ("qc_chi4_mean_score", "χ4 rotamer surprise")]
figB, axs = plt.subplots(1, 4, figsize=(15.2, 3.4))
for ax, (c, lab) in zip(axs, SHOW):
    xe = expD["appres"].values; ye = pd.to_numeric(expD[c], errors="coerce").values
    r = spearmanr(xe, ye, nan_policy="omit")[0]
    kind = f"resolution-dependent, ρ={r:.2f}" if abs(r) > 0.3 else "resolution-invariant"
    ax.scatter(xe, ye, s=6, alpha=0.25, color="#9aa0a6", label="experimental")
    xs = np.linspace(np.nanmin(xe), np.nanmax(xe), 100); ax.plot(xs, whatres.resid_expected(CO[c], xs), color="k", lw=2, label="expected-value fit")
    for g in ["AlphaFold3", "Boltz2"]:
        s = aiD[aiD.origin == g]; col, mk = ST[g]
        ax.scatter(s["appres"], pd.to_numeric(s[c], errors="coerce"), s=14, marker=mk, color=col, edgecolors="k", lw=0.3, alpha=0.8, label=g)
    ax.set_xlabel("apparent resolution (Å)"); ax.set_title(f"{lab}  ({kind})", fontsize=8); ax.grid(alpha=0.3)
    rec(f"fig4b_spearman_appres_{c}", r, f"Spearman of {lab} with apparent resolution (experimental)")
axs[0].set_ylabel("descriptor value"); axs[0].legend(fontsize=6.5, loc="best")
figB.tight_layout(); fs.savepanel(figB, O, "Fig4b_residualisation_fit")

# ---------- c: top contributors ----------
nice = lambda c: (c.replace("qc_", "").replace("_mean_score", " (mean)").replace("_sd_pctile", " (spread)").replace("_frac_extreme", " (outliers)")
                  .replace("ca_chirality", "Cα-chirality").replace("cb_deviation", "Cβ-dev").replace("bond_C_O", "C=O bond").replace("rotamer_outlier", "rotamer-out"))
coef = CLF.named_steps["logisticregression"].coef_[0]; order = np.argsort(-np.abs(coef))[:15][::-1]
figC, ax = plt.subplots(figsize=(4.8, 4.4))
ax.barh(range(len(order)), [coef[i] for i in order], color=["#d62728" if coef[i] > 0 else "#1f77b4" for i in order])
ax.set_yticks(range(len(order))); ax.set_yticklabels([nice(PAN[i]) for i in order], fontsize=6.8)
ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("logistic weight on standardised residual")
ax.set_title("top realism contributors\n(red = model-like ↓realism, blue = experimental-like)", fontsize=8.5)
fs.savepanel(figC, O, "Fig4c_top_contributors")
top5 = [PAN[i] for i in np.argsort(-np.abs(coef))[:5]]
rec("fig4c_top5", ";".join(top5), "five strongest realism contributors (|logistic weight|)")

# ---------- d: realism by method ----------
og = ["X-ray", "cryo-EM", "NMR", "AlphaFold2", "Boltz2", "AlphaFold3"]
figD, ax = plt.subplots(figsize=(5.4, 3.8))
for i, g in enumerate(og):
    v = D[D.origin == g]["realism"].dropna().values
    k = gaussian_kde(v); gr = np.linspace(max(0, v.min()), min(100, v.max()), 120); w = k(gr); w = w / w.max() * 0.4
    ax.fill_betweenx(gr, i - w, i + w, facecolor=ST[g][0], edgecolor="k", lw=0.5, alpha=0.9)
    ax.hlines(np.median(v), i - 0.4, i + 0.4, color="k", lw=1.4)
    rec(f"realism_median_{g}", float(np.median(v)), f"median realism, {g}")
    rec(f"realism_q1_{g}", float(np.percentile(v, 25)), f"realism 25th percentile, {g}"); rec(f"realism_q3_{g}", float(np.percentile(v, 75)), f"realism 75th percentile, {g}")
ax.set_xticks(range(len(og))); ax.set_xticklabels(og, rotation=20, fontsize=7.5)
ax.set_ylabel("realism score (0–100)"); ax.set_title("realism by method of origin"); ax.set_ylim(0, 102)
fs.savepanel(figD, O, "Fig4d_realism_by_method")

# ---------- e: orthogonality ----------
figE, ax = plt.subplots(figsize=(4.6, 4.0))
for g in ["X-ray", "cryo-EM", "NMR", "AlphaFold2", "AlphaFold3", "Boltz2"]:
    s = D[D.origin == g]; col, mk = ST[g]
    ax.scatter(s["appres"], s["realism"], s=12, marker=mk, c=col, alpha=0.55, edgecolors="none", label=g)
rho = spearmanr(D.loc[EXP, "appres"], D.loc[EXP, "realism"], nan_policy="omit")[0]
ax.set_xlabel("apparent resolution (Å)"); ax.set_ylabel("realism score (0–100)")
ax.set_title(f"realism vs apparent resolution\n(experimental Spearman {rho:.2f})", fontsize=9)
ax.legend(fontsize=6.3, loc="lower left", ncol=2); ax.set_ylim(0, 102)
fs.savepanel(figE, O, "Fig4e_orthogonality")
rec("fig4e_spearman_realism_appres_exp", rho, "Spearman realism vs apparent resolution, X-ray + cryo-EM")

# ---------- f: pLDDT vs realism / apparent resolution ----------
A_ = D[AI].copy(); A_["plddt"] = pd.to_numeric(A_["bf_ca_mean"], errors="coerce")
figF, axs = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True)
for ax, (yc, ylab) in zip(axs, [("realism", "realism score (0–100)"), ("appres", "apparent resolution (Å)")]):
    for g in ["AlphaFold3", "Boltz2"]:
        s = A_[A_.origin == g]; col, mk = ST[g]; r = spearmanr(s["plddt"], s[yc], nan_policy="omit")[0]
        ax.scatter(s["plddt"], s[yc], s=14, marker=mk, c=col, alpha=0.7, edgecolors="k", lw=0.3, label=f"{g} (ρ={r:+.2f})")
        rec(f"fig4f_spearman_plddt_{yc}_{g}", r, f"Spearman pLDDT vs {yc}, {g}")
    ax.set_xlabel("mean pLDDT"); ax.set_ylabel(ylab); ax.legend(fontsize=6.8); ax.grid(alpha=0.3)
axs[0].set_title("confidence vs realism (weak)", fontsize=9); axs[1].set_title("confidence vs apparent resolution (strong)", fontsize=9)
figF.tight_layout(); fs.savepanel(figF, O, "Fig4f_pLDDT_vs_realism")

# ---------- g: overlay image (coordinates only, unchanged) ----------
shutil.copy2(os.path.join(MS, "Figures", "Figure4", "Fig4g_7dg7_overlay.png"), os.path.join(O, "Fig4g_7dg7_overlay.png"))

# ---------- h: 7dg7 rotamers ----------
MODEL = os.path.join(BENCH, "ai", "af3_7dg7.pdb"); EXPP = os.path.join(H, "pdbs", "7dg7.pdb")
def dih(p):
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]; b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
CHI = {"LYS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"]],
       "ARG": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"]]}
def rot(path):
    st = gemmi.read_structure(path); out = {}
    for c in st[0]:
        for r in c:
            if r.name in CHI:
                cs = []
                for ats in CHI[r.name]:
                    try: cs.append(dih([np.array([r[a][0].pos.x, r[a][0].pos.y, r[a][0].pos.z]) for a in ats]))
                    except Exception: cs.append(np.nan)
                out[(r.name, r.seqid.num)] = cs
    return out
mod, exp_ = rot(MODEL), rot(EXPP); common = sorted(set(mod) & set(exp_), key=lambda x: x[1])
wrong = set()
for k in common:
    for ci in (2, 3):
        x, y_ = exp_[k][ci], mod[k][ci]
        if np.isfinite(x) and np.isfinite(y_):
            dd = abs(x - y_) % 360; dd = min(dd, 360 - dd)
            if dd > 40: wrong.add(k)
st = gemmi.read_structure(MODEL); plddt = float(np.mean([r["CA"][0].b_iso for c in st[0] for r in c if r.find_atom("CA", "*")]))
row = D[D.k == "af3_7dg7"].iloc[0]; r7, a7 = float(row.realism), float(row.appres)
figH, axs = plt.subplots(1, 2, figsize=(8.6, 4.3)); AA1 = {"ARG": "R", "LYS": "K"}
for ax, ci, lab in [(axs[0], 2, "χ3"), (axs[1], 3, "χ4")]:
    for k in common:
        x, y_ = exp_[k][ci], mod[k][ci]
        if not (np.isfinite(x) and np.isfinite(y_)): continue
        dd = abs(x - y_) % 360; dd = min(dd, 360 - dd); w_ = dd > 40
        ax.scatter(x, y_, s=42, c="#d62728" if w_ else "#2ca02c", edgecolors="k", lw=0.4, zorder=3)
        if w_: ax.annotate(f"{AA1.get(k[0], k[0][0])}{k[1]}", (x, y_), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.plot([-180, 180], [-180, 180], "k--", lw=0.8, alpha=0.6); ax.set_xlim(-185, 185); ax.set_ylim(-185, 185); ax.set_aspect("equal")
    ax.set_xlabel(f"experimental {lab} (°)"); ax.set_ylabel(f"AF3 model {lab} (°)"); ax.set_title(f"{lab}, Lys/Arg", fontsize=9); ax.grid(alpha=0.3)
axs[0].scatter([], [], c="#d62728", label="differs from deposited (>40°)"); axs[0].scatter([], [], c="#2ca02c", label="matches deposited"); axs[0].legend(fontsize=7, loc="lower right")
figH.suptitle(f"7dg7: AlphaFold3 (pLDDT {plddt:.1f}, realism {r7:.1f}) differs from the deposited conformation in {len(wrong)} of {len(common)} long side chains", fontsize=9)
figH.tight_layout(); fs.savepanel(figH, O, "Fig4h_7dg7_rotamers")
rec("case_7dg7_plddt", plddt, "AF3 7dg7 mean CA pLDDT"); rec("case_7dg7_realism", r7, "AF3 7dg7 realism"); rec("case_7dg7_appres", a7, "AF3 7dg7 apparent resolution", "A")
rec("case_7dg7_n_wrong", len(wrong), "AF3 7dg7 Lys/Arg side chains with chi3 or chi4 >40 deg from deposited")
rec("case_7dg7_n_longsc", len(common), "Lys/Arg side chains compared, 7dg7")
rec("case_7dg7_wrong_list", ",".join(f"{AA1[k[0]]}{k[1]}" for k in sorted(wrong, key=lambda x: x[1])), "the differing side chains")
print(f"[ms_fig4] realism medians " + " ".join(f"{g} {np.median(D[D.origin==g].realism):.1f}" for g in og) + f" | rho {rho:.3f} | 7dg7 realism {r7:.1f} appres {a7:.2f} -> {O}")
