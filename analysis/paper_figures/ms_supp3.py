#!/usr/bin/env python3.14
"""Supplementary Figure 3: the N-Ca-C backbone angle (tau) by backbone conformation.
Coordinate-only: per-residue geometry from over_regularisation_geometry/geom_residues.tsv.gz
(extract_geometry.py; no restraints, no energy function, no WhatRes normalisation).
Same residue filter as the N-Ca-C panel of Figure 2 (make_geom_fig.py): Gly and Pro excluded,
100 <= tau <= 122 deg. Conformation is assigned from phi/psi (core basins):
  alpha-helical  -100 < phi < -30  and  -80 < psi < -10
  beta           -180 <= phi < -90  and  90 < psi <= 180
  other          everything else with phi and psi defined
(a) tau density in the <1 A X-ray reference, split by basin
(b) helix minus sheet mean tau per cohort (error bar = 95% CI, residue level)
(c) mean tau offset from the <1 A reference, per basin and cohort
"""
import os, numpy as np, pandas as pd
from scipy.stats import gaussian_kde
from ms_common import fs, outdir, recorder, AN
import matplotlib.pyplot as plt

rec = recorder(__file__); O = outdir("Supplementary")
COH = ["<1A X-ray", ">2.5A X-ray", "cryo-EM", "AF3", "Boltz2"]
COL = {"<1A X-ray": "#000000", ">2.5A X-ray": "#2ca02c", "cryo-EM": "#1f77b4", "AF3": "#d62728", "Boltz2": "#e08214"}
LAB = {"<1A X-ray": "X-ray <1 Å", ">2.5A X-ray": "X-ray >2.5 Å", "cryo-EM": "cryo-EM", "AF3": "AlphaFold3", "Boltz2": "Boltz2"}
BCOL = {"helix": "#7b3294", "sheet": "#008837", "other": "#999999"}   # not the cohort colours of b/c

d = pd.read_csv(os.path.join(AN, "over_regularisation_geometry", "geom_residues.tsv.gz"), sep="\t",
                usecols=["cohort", "resn", "angle_N_CA_C", "phi", "psi"])
n = {"all": len(d)}
d = d[~d.resn.isin({"GLY", "PRO"})]; n["no_gly_pro"] = len(d)
d = d.dropna(subset=["angle_N_CA_C", "phi", "psi"]); n["phi_psi_tau_defined"] = len(d)
d = d[(d.angle_N_CA_C >= 100) & (d.angle_N_CA_C <= 122)]; n["tau_100_122"] = len(d)
print("[supp3] residues:", " -> ".join(f"{k} {v}" for k, v in n.items()))
missing = set(COH) - set(d.cohort)
if missing: raise SystemExit(f"STOP: cohorts absent {missing}")
f, p = d.phi, d.psi
d["basin"] = np.where((f > -100) & (f < -30) & (p > -80) & (p < -10), "helix",
             np.where((f < -90) & (p > 90), "sheet", "other"))
t = d.groupby(["cohort", "basin"]).angle_N_CA_C.agg(["mean", "var", "count"])
ref = t.loc["<1A X-ray"]

gap, ci = {}, {}
for c in COH:
    h, s = t.loc[(c, "helix")], t.loc[(c, "sheet")]
    gap[c] = h["mean"] - s["mean"]; ci[c] = 1.96 * np.sqrt(h["var"] / h["count"] + s["var"] / s["count"])
    rec(f"supp3_gap_{c}", gap[c], f"helix minus sheet mean N-CA-C angle, {c}", "deg")
    for b in ("helix", "sheet"):
        rec(f"supp3_n_{b}_{c}", int(t.loc[(c, b), "count"]), f"{b} residues, {c}")
        if c != "<1A X-ray":
            rec(f"supp3_offset_{b}_{c}", t.loc[(c, b), "mean"] - ref.loc[b, "mean"], f"mean N-CA-C offset vs <1 A, {b}, {c}", "deg")
for b in ("helix", "sheet", "other"):
    rec(f"supp3_ref_mean_{b}", ref.loc[b, "mean"], f"<1 A X-ray mean N-CA-C angle, {b}", "deg")
for k, v in n.items(): rec(f"supp3_nres_{k}", v, f"residues after filter step {k}")

fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.3), gridspec_kw={"width_ratios": [1.2, 1, 1]})
# (a) <1 A reference: density split by basin (components weighted by their share, so they sum to the total)
ax = axs[0]; r = d[d.cohort == "<1A X-ray"]; g = np.linspace(102, 120, 300)
ax.plot(g, gaussian_kde(r.angle_N_CA_C)(g), color="k", lw=1.8, label=f"all (n={len(r):,})")
for b in ("helix", "sheet", "other"):
    v = r[r.basin == b].angle_N_CA_C
    ax.plot(g, gaussian_kde(v)(g) * len(v) / len(r), color=BCOL[b], lw=1.3, label=f"{b} ({v.mean():.1f}°)")
ax.set_xlabel("N-Cα-C angle (°)"); ax.set_ylabel("density"); ax.legend(fontsize=7, frameon=False)
ax.set_title("X-ray <1 Å: the two modes are helix and sheet")
# (b) helix - sheet gap per cohort
ax = axs[1]; x = np.arange(len(COH))
ax.bar(x, [gap[c] for c in COH], yerr=[ci[c] for c in COH], color=[COL[c] for c in COH], capsize=2, error_kw={"lw": 0.8})
ax.axhline(gap["<1A X-ray"], color="k", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels([LAB[c] for c in COH], rotation=25, ha="right")
ax.set_ylabel("helix minus sheet N-Cα-C (°)"); ax.set_title("the helix-sheet gap is compressed")
# (c) offset from the <1 A mean, per basin
ax = axs[2]; oth = COH[1:]; w = 0.2; xb = np.arange(2)
for i, c in enumerate(oth):
    ax.bar(xb + (i - 1.5) * w, [t.loc[(c, b), "mean"] - ref.loc[b, "mean"] for b in ("helix", "sheet")], w, color=COL[c], label=LAB[c])
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xb); ax.set_xticklabels(["helix", "sheet"])
ax.set_ylabel("mean N-Cα-C offset vs <1 Å (°)"); ax.legend(fontsize=7, frameon=False); ax.set_title("offset from the <1 Å reference")
fig.tight_layout(); fs.panel_letters(fig, axs); fs.savepanel(fig, O, "SuppFig3_shoulder_helix_sheet")
print("[supp3] <1A helix %.2f sheet %.2f other %.2f | gap " % tuple(ref.loc[b, "mean"] for b in ("helix", "sheet", "other"))
      + " ".join(f"{c} {gap[c]:.2f}" for c in COH)
      + " | AF3 offset helix %.2f sheet %.2f | Boltz2 helix %.2f sheet %.2f" % tuple(
          t.loc[(c, b), "mean"] - ref.loc[b, "mean"] for c in ("AF3", "Boltz2") for b in ("helix", "sheet")))
