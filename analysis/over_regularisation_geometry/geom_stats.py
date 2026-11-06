#!/usr/bin/env python3.14
"""Cohort geometry statistics: within-residue-type SD (homogenisation) and mean-offset vs <1A reference."""
import os, numpy as np, pandas as pd
OUT=os.path.dirname(os.path.abspath(__file__))
df=pd.read_csv(os.path.join(OUT,"geom_residues.tsv.gz"),sep="\t")
COH=["<1A X-ray",">2.5A X-ray","cryo-EM","AF3","Boltz2"]
# physical ranges (drop gross outliers / bad atoms); ideal (Engh-Huber) for reference
PAR={ # param: (lo, hi, ideal, unit)
 "bond_N_CA":(1.35,1.55,1.459,"Å"), "bond_CA_C":(1.42,1.63,1.525,"Å"),
 "bond_C_O":(1.16,1.33,1.229,"Å"),  "bond_CA_CB":(1.45,1.63,1.530,"Å"),
 "bond_C_N":(1.26,1.42,1.336,"Å"),
 "angle_N_CA_C":(100,122,111.0,"°"),"angle_CA_C_N":(108,128,116.2,"°"),
 "angle_C_N_CA":(112,132,121.7,"°"),"angle_CA_C_O":(112,130,120.8,"°"),
 "angle_N_CA_CB":(100,120,110.6,"°"),
}
GLYPRO={"GLY","PRO"}
def clean(p):
    lo,hi,_,_=PAR[p]
    d=df[["cohort","resn",p]].dropna()
    d=d[(d[p]>=lo)&(d[p]<=hi)]
    if p.startswith("angle_N_CA_C") or p=="angle_N_CA_C":
        d=d[~d.resn.isin(GLYPRO)]
    return d
# within-type pooled SD + mean-offset vs <1A, per cohort
sd_rows=[]; off_rows=[]
for p in PAR:
    d=clean(p)
    ref_mean=d[d.cohort=="<1A X-ray"].groupby("resn")[p].mean()  # per-type <1A means
    for c in COH:
        g=d[d.cohort==c]
        # within-type SD pooled: subtract each type's mean (in THIS cohort), pool, std
        cent=g[p]-g.groupby("resn")[p].transform("mean")
        sd=cent.std()
        # mean offset vs <1A: per type (cohort type-mean - <1A type-mean), weighted by count
        tm=g.groupby("resn")[p].mean(); n=g.groupby("resn")[p].count()
        common=tm.index.intersection(ref_mean.index)
        off=np.average((tm[common]-ref_mean[common]), weights=n[common]) if len(common) else np.nan
        sd_rows.append(dict(param=p,cohort=c,within_type_SD=sd,n=len(g)))
        off_rows.append(dict(param=p,cohort=c,mean_offset_vs_1A=off,unit=PAR[p][3]))
sd=pd.DataFrame(sd_rows); off=pd.DataFrame(off_rows)
sd.to_csv(os.path.join(OUT,"cohort_within_type_SD.tsv"),sep="\t",index=False)
off.to_csv(os.path.join(OUT,"cohort_mean_offset.tsv"),sep="\t",index=False)

pd.set_option("display.width",200,"display.float_format",lambda x:f"{x:.4f}")
print("=== within-type SD by cohort (lower = more homogenised) ===")
piv=sd.pivot(index="param",columns="cohort",values="within_type_SD")[COH]
print(piv.to_string())
print("\n=== SD ratio to <1A reference (1.0 = as variable as ultrahigh-res; <1 = tighter) ===")
print((piv.div(piv["<1A X-ray"],axis=0)).to_string())
print("\n=== |mean offset| vs <1A, AF3 vs Boltz2 (bias away from ultrahigh-res average) ===")
po=off.pivot(index="param",columns="cohort",values="mean_offset_vs_1A")[COH]
print(po.to_string())
print("\nMean |offset| across params:")
print(po.abs().mean().to_string())
