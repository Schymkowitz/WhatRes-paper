#!/usr/bin/env python3.14
"""Raw backbone geometry across cohorts (ultra-high-res <1A reference vs regularised cohorts).
Computes per-residue backbone bond lengths, bond angles, Cbeta deviation, omega, phi/psi
directly from coordinates (no restraints, no energy function, no WhatRes normalisation).
Cohorts: <1A X-ray (reference), >2.5A X-ray, cryo-EM, AF3, Boltz2.
"""
import os, sys, glob, numpy as np, pandas as pd, gemmi
HRP=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # HighResPDB
BM=os.path.join(HRP,"benchmark"); PDBS=os.path.join(BM,"pdbs"); AI=os.path.join(BM,"ai")
OUT=os.environ.get("GEOM_OUT") or os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT,exist_ok=True)

def ids_from(listfile):
    out=[]
    for ln in open(os.path.join(BM,listfile)):
        ln=ln.strip()
        if not ln: continue
        out.append(os.path.basename(ln).replace(".pdb","").lower())
    return out

COHORTS={
 "<1A X-ray":        [os.path.join(PDBS,i+".pdb") for i in ids_from("list_xray_0.5-1.0A.txt")],
 ">2.5A X-ray":      [os.path.join(PDBS,i+".pdb") for i in ids_from("list_xray_2.5-3.0A.txt")+ids_from("list_xray_3.0-3.5A.txt")],
 "cryo-EM":          [os.path.join(PDBS,i+".pdb") for i in ids_from("list_cryoem_2.0-2.5A.txt")+ids_from("list_cryoem_2.5-3.0A.txt")],
 "AF3":              sorted(glob.glob(os.path.join(AI,"af3_*.pdb"))),
 "Boltz2":           sorted(glob.glob(os.path.join(AI,"boltz2_*.pdb"))),
}
AA=set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
def vec(a): return np.array([a.pos.x,a.pos.y,a.pos.z])
def dist(a,b): return float(np.linalg.norm(vec(a)-vec(b)))
def angle(a,b,c):
    u=vec(a)-vec(b); v=vec(c)-vec(b)
    cos=np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos,-1,1))))
def dih(p1,p2,p3,p4):
    b0=vec(p1)-vec(p2); b1=vec(p3)-vec(p2); b2=vec(p4)-vec(p3)
    b1n=b1/np.linalg.norm(b1)
    v=b0-np.dot(b0,b1n)*b1n; w=b2-np.dot(b2,b1n)*b1n
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1n,v),w),np.dot(v,w))))
def cb_dev(N,CA,C,CB):
    # ideal CB from N,CA,C (standard tetrahedral construction)
    n=vec(N)-vec(CA); c=vec(C)-vec(CA)
    n/=np.linalg.norm(n); c/=np.linalg.norm(c)
    b=np.cross(c,n); b/=np.linalg.norm(b)
    bis=(n+c); bis/=np.linalg.norm(bis)
    # ideal: CB along -bisector tilted out of plane; use canonical angles
    ideal=vec(CA)+1.532*(-0.5667*bis+0.8238*b)  # approx tetrahedral geometry
    return float(np.linalg.norm(vec(CB)-ideal))

def getatom(res,name):
    # gemmi reports no-altloc atoms as '\x00'; accept those plus the primary 'A' conformer
    for a in res:
        if a.name==name and a.altloc in ("\x00","","A"): return a
    return None

rows=[]; used=[]
for coh,files in COHORTS.items():
    nstruct=0; nmiss=0; nfail=0
    for f in files:
        if not os.path.exists(f): nmiss+=1; continue          # listed in the design but no file (logged below)
        try: st=gemmi.read_structure(f)
        except Exception: nfail+=1; continue
        if len(st)==0: nfail+=1; continue
        m=st[0]; nstruct+=1; used.append((os.path.splitext(os.path.basename(f))[0], coh))
        for ch in m:
            resl=[r for r in ch if r.name in AA]
            for idx,r in enumerate(resl):
                N=getatom(r,"N");CA=getatom(r,"CA");C=getatom(r,"C");O=getatom(r,"O");CB=getatom(r,"CB")
                if not(N and CA and C): continue
                d=dict(cohort=coh,resn=r.name)
                try: d["bond_N_CA"]=dist(N,CA)
                except: pass
                try: d["bond_CA_C"]=dist(CA,C)
                except: pass
                if O is not None:
                    try: d["bond_C_O"]=dist(C,O)
                    except: pass
                    try: d["angle_CA_C_O"]=angle(CA,C,O)
                    except: pass
                if CB is not None:
                    try: d["bond_CA_CB"]=dist(CA,CB)
                    except: pass
                    try: d["angle_N_CA_CB"]=angle(N,CA,CB)
                    except: pass
                    try: d["angle_CB_CA_C"]=angle(CB,CA,C)
                    except: pass
                try: d["angle_N_CA_C"]=angle(N,CA,C)
                except: pass
                # peptide to next residue
                if idx+1<len(resl):
                    r2=resl[idx+1]; N2=getatom(r2,"N");CA2=getatom(r2,"CA")
                    if N2 is not None:
                        try: d["bond_C_N"]=dist(C,N2)
                        except: pass
                        try: d["angle_CA_C_N"]=angle(CA,C,N2)
                        except: pass
                        if CA2 is not None:
                            try: d["angle_C_N_CA"]=angle(C,N2,CA2)
                            except: pass
                            try: d["omega"]=dih(CA,C,N2,CA2)
                            except: pass
                # phi/psi for binning
                if idx>0:
                    r0=resl[idx-1]; C0=getatom(r0,"C")
                    if C0 is not None:
                        try: d["phi"]=dih(C0,N,CA,C)
                        except: pass
                if idx+1<len(resl):
                    r2=resl[idx+1]; N2=getatom(r2,"N")
                    if N2 is not None:
                        try: d["psi"]=dih(N,CA,C,N2)
                        except: pass
                rows.append(d)
    print(f"{coh}: {len(files)} listed, {nstruct} read, {nmiss} without a file, {nfail} unreadable")
df=pd.DataFrame(rows)
df.to_parquet(os.path.join(OUT,"geom_residues.parquet")) if False else df.to_csv(os.path.join(OUT,"geom_residues.tsv.gz"),sep="\t",index=False,compression="gzip")
pd.DataFrame(used,columns=["id","cohort"]).to_csv(os.path.join(OUT,"geom_structures.tsv"),sep="\t",index=False)
print("total residues:",len(df))
print(df.cohort.value_counts())
print("wrote geom_residues.tsv.gz")
