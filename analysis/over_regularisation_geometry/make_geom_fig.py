#!/usr/bin/env python3.14
"""Javi-style over-regularisation figure: ultra-high-res (<1A) geometry is broadest; low-res X-ray,
cryo-EM, AF3 and Boltz2 are all tighter (homogenised), and AF3 is mean-shifted off the <1A average
while Boltz2 is not."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
OUT=os.path.dirname(os.path.abspath(__file__))
df=pd.read_csv(os.path.join(OUT,"geom_residues.tsv.gz"),sep="\t")
sd=pd.read_csv(os.path.join(OUT,"cohort_within_type_SD.tsv"),sep="\t")
off=pd.read_csv(os.path.join(OUT,"cohort_mean_offset.tsv"),sep="\t")
COH=["<1A X-ray",">2.5A X-ray","cryo-EM","AF3","Boltz2"]
COL={"<1A X-ray":"#000000",">2.5A X-ray":"#2ca02c","cryo-EM":"#1f77b4","AF3":"#d62728","Boltz2":"#e08214"}
LAB={"<1A X-ray":"X-ray <1 Å (reference)",">2.5A X-ray":"X-ray >2.5 Å","cryo-EM":"cryo-EM","AF3":"AlphaFold3","Boltz2":"Boltz2"}
GLYPRO={"GLY","PRO"}
RNG={"angle_N_CA_C":(100,122),"bond_C_O":(1.16,1.33)}
def centered(param,span):
    lo,hi=RNG[param]
    d=df[["cohort","resn",param]].dropna()
    d=d[(d[param]>=lo)&(d[param]<=hi)]
    if param=="angle_N_CA_C": d=d[~d.resn.isin(GLYPRO)]
    ref=d[d.cohort=="<1A X-ray"].groupby("resn")[param].mean()
    d=d[d.resn.isin(ref.index)].copy()
    d["c"]=d[param]-d.resn.map(ref)   # centre each type at its <1A mean
    return d[(d.c>-span)&(d.c<span)]

fig,axs=plt.subplots(1,4,figsize=(16.5,4.1))
# ---- (a) N-CA-C angle, centred ----
ax=axs[0]; d=centered("angle_N_CA_C",8); grid=np.linspace(-8,8,200)
for c in COH:
    v=d[d.cohort==c]["c"].values
    if len(v)<50: continue
    k=gaussian_kde(v); ax.plot(grid,k(grid),color=COL[c],lw=2.2 if c=="<1A X-ray" else 1.6,
        ls="-" if c!="<1A X-ray" else "-",label=LAB[c],alpha=0.9)
ax.axvline(0,color="0.6",lw=0.8,ls=":")
ax.set_xlabel("N-Cα-C angle, deviation from <1 Å mean (°)"); ax.set_ylabel("density")
ax.set_title("backbone bond angle (N-Cα-C)\nbroadest at <1 Å; AF3 shifted right",fontsize=9)
ax.legend(fontsize=6.6)
# ---- (b) C=O bond length, centred ----
ax=axs[1]; d=centered("bond_C_O",0.05); grid=np.linspace(-0.05,0.05,200)
for c in COH:
    v=d[d.cohort==c]["c"].values
    if len(v)<50: continue
    k=gaussian_kde(v); ax.plot(grid,k(grid),color=COL[c],lw=2.2 if c=="<1A X-ray" else 1.6,label=LAB[c],alpha=0.9)
ax.axvline(0,color="0.6",lw=0.8,ls=":")
ax.set_xlabel("C=O bond length, deviation from <1 Å mean (Å)"); ax.set_ylabel("density")
ax.set_title("backbone bond length (C=O)\nall regularised cohorts tighter than <1 Å",fontsize=9)
ax.legend(fontsize=6.6)
# ---- (c) within-type SD ratio to <1A ----
ax=axs[2]
piv=sd.pivot(index="param",columns="cohort",values="within_type_SD")[COH]
ratio=piv.div(piv["<1A X-ray"],axis=0)
params=list(ratio.index); x=np.arange(len(params)); w=0.17
show=[c for c in COH if c!="<1A X-ray"]
for i,c in enumerate(show):
    ax.bar(x+(i-1.5)*w,ratio[c],w,color=COL[c],label=LAB[c])
ax.axhline(1.0,color="k",lw=0.8,ls="--",label="<1 Å reference")
nice={"angle_CA_C_N":"∠Cα-C-N","angle_CA_C_O":"∠Cα-C-O","angle_C_N_CA":"∠C-N-Cα","angle_N_CA_C":"∠N-Cα-C","angle_N_CA_CB":"∠N-Cα-Cβ","bond_CA_C":"Cα-C","bond_CA_CB":"Cα-Cβ","bond_C_N":"C-N","bond_C_O":"C=O","bond_N_CA":"N-Cα"}
ax.set_xticks(x); ax.set_xticklabels([nice.get(p,p) for p in params],rotation=55,ha="right",fontsize=6.5)
ax.set_ylabel("within-type SD ÷ <1 Å SD"); ax.set_ylim(0,1.15)
ax.set_title("geometric spread relative to <1 Å\n(every cohort is tighter)",fontsize=9); ax.legend(fontsize=6.2)
# ---- (d) mean offset vs <1A for the backbone angles: AF3 vs Boltz2 ----
ax=axs[3]
po=off.pivot(index="param",columns="cohort",values="mean_offset_vs_1A")[COH]
angs=["angle_CA_C_N","angle_CA_C_O","angle_C_N_CA","angle_N_CA_C","angle_N_CA_CB"]
po=po.loc[angs]; x=np.arange(len(angs))
for i,c in enumerate(show):
    ax.bar(x+(i-1.5)*w,po[c],w,color=COL[c],label=LAB[c])
ax.axhline(0,color="k",lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([nice[p] for p in angs],rotation=55,ha="right",fontsize=6.5)
ax.set_ylabel("mean angle offset vs <1 Å (°)")
mo=po.abs().mean()
ax.set_title(f"bias from the <1 Å average\nmean |offset|: AF3 {mo['AF3']:.2f}° vs Boltz2 {mo['Boltz2']:.2f}°",fontsize=9)
ax.legend(fontsize=6.2)
# suptitle removed: this strip is now panels c-f of the merged Figure 2; the description lives in the figure legend
for ax,lab in zip(axs,["c","d","e","f"]):   # panels c-f of the merged Figure 2 (method-of-origin + over-regularisation)
    ax.text(-0.16,1.04,lab,transform=ax.transAxes,fontsize=14,fontweight="bold",va="bottom",ha="left")
fig.tight_layout()
for ext in ["pdf","png","svg"]: fig.savefig(os.path.join(OUT,f"Fig_overregularisation_geometry.{ext}"),dpi=150,bbox_inches="tight")
print("wrote Fig_overregularisation_geometry.*")
