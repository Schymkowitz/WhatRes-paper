#!/usr/bin/env python3
"""build_param_table.py — supplementary table of all WhatRes parameters with provenance.
Columns separate IMPLEMENTATION (what actually computed it: our whatres-features C++ or Python)
from CONCEPT/ORIGIN (the literature the idea comes from).
Output: native_pipeline/tables/WhatRes Suppl Table 1 Parameters.xlsx  (sheets: Parameters, Legend)"""
import os, re, pandas as pd
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cols=[c for c in pd.read_csv(os.path.join(HERE,"feature_matrix.tsv"),sep="\t",nrows=1).columns if c!="id"]

# implementation engines
CPP_BG ="whatres-features C++ (own; BondGeometry.h)"
CPP_CL ="whatres-features C++ (own; Clashscore.h)"
CPP_RD ="whatres-features C++ (own; StructureReferenceDB.h)"
CPP_SURF="whatres-features C++ (own; pdb_surface_analysis: SurfacePatcher.h / CavityDetector.h)"
CPP_DIST="whatres-features C++ (own; PDB_dist) + Python histogram summary"
CPP_QC ="whatres-features C++ (own; pdb_quality_check) + Python aggregation"
PY     ="Python (build_feature_matrix.py)"
CPP_DIH="whatres-features C++ (own; Peptide::SetDihedrals in the PDB reader)"

CMD_QC="pdb_quality_check"; CMD_SURF="pdb_surface_analysis"; CMD_DIST="PDB_dist"; CMD_PY="build_feature_matrix.py"

# base per-residue metric → (family, description, command, implementation, concept/origin)
M={
 "bond_CA_C":("Covalent geometry — bond length","Cα–C backbone bond length",CMD_QC,CPP_BG,"ideal covalent geometry (Engh & Huber 1991)"),
 "bond_C_O":("Covalent geometry — bond length","C=O carbonyl bond length",CMD_QC,CPP_BG,"ideal covalent geometry (Engh & Huber 1991)"),
 "bond_C_N":("Covalent geometry — bond length","C–N peptide bond length",CMD_QC,CPP_BG,"ideal covalent geometry (Engh & Huber 1991)"),
 "bond_N_CA":("Covalent geometry — bond length","N–Cα backbone bond length",CMD_QC,CPP_BG,"ideal covalent geometry (Engh & Huber 1991)"),
 "bond_CA_CB":("Covalent geometry — bond length","Cα–Cβ bond length",CMD_QC,CPP_BG,"ideal covalent geometry (Engh & Huber 1991)"),
 "angle_N_CA_C":("Covalent geometry — bond angle","N–Cα–C backbone bond angle (τ)",CMD_QC,CPP_BG,"ideal covalent geometry (Engh & Huber 1991)"),
 "angle_CA_C_N":("Covalent geometry — bond angle","Cα–C–N backbone bond angle",CMD_QC,CPP_BG,"ideal covalent geometry (Engh & Huber 1991)"),
 "angle_C_N_CA":("Covalent geometry — bond angle","C–N–Cα backbone bond angle",CMD_QC,CPP_BG,"ideal covalent geometry (Engh & Huber 1991)"),
 "phi":("Backbone dihedral","Backbone φ torsion",CMD_QC,CPP_DIH,"Ramachandran et al. 1963"),
 "psi":("Backbone dihedral","Backbone ψ torsion",CMD_QC,CPP_DIH,"Ramachandran et al. 1963"),
 "omega":("Backbone dihedral","Backbone ω (peptide) torsion",CMD_QC,CPP_DIH,"peptide-bond geometry"),
 "rama2D":("Backbone dihedral","Ramachandran 2D (φ,ψ) probability density",CMD_QC,CPP_RD,"Ramachandran et al. 1963; reference density this work"),
 "chi1":("Side-chain dihedral","Side-chain χ1 torsion",CMD_QC,CPP_BG,"side-chain rotamers"),
 "chi2":("Side-chain dihedral","Side-chain χ2 torsion",CMD_QC,CPP_BG,"side-chain rotamers"),
 "chi3":("Side-chain dihedral","Side-chain χ3 torsion (long side chains)",CMD_QC,CPP_BG,"side-chain rotamers"),
 "chi4":("Side-chain dihedral","Side-chain χ4 torsion (Arg/Lys)",CMD_QC,CPP_BG,"side-chain rotamers"),
 "rotamer_outlier":("Side-chain dihedral","χ1 distance to nearest staggered well (rotamericity proxy)",CMD_QC,CPP_BG,"this work (cf. Dunbrack rotamer library)"),
 "ca_chirality":("Chirality & planarity","Cα improper dihedral (N–Cα–C–Cβ); L/D chirality",CMD_QC,CPP_BG,"Cα chirality (improper dihedral)"),
 "cb_deviation":("Chirality & planarity","Cβ deviation from ideal position (Å)",CMD_QC,CPP_BG,"Lovell et al. 2003"),
 "peptide_planarity":("Chirality & planarity","Peptide-bond planarity (|ω| deviation from 0/180°)",CMD_QC,CPP_BG,"peptide planarity"),
 "cis_peptide":("Chirality & planarity","cis-peptide flag (|ω| < 30°)",CMD_QC,CPP_BG,"cis-peptide detection (cf. Williams 2018)"),
 "cablam_mu_in":("Backbone conformation (CaBLAM)","CaBLAM μ_in Cα virtual dihedral",CMD_QC,CPP_BG,"CaBLAM (Prisant et al. 2020)"),
 "cablam_mu_out":("Backbone conformation (CaBLAM)","CaBLAM μ_out Cα virtual dihedral",CMD_QC,CPP_BG,"CaBLAM (Prisant et al. 2020)"),
 "cablam_nu":("Backbone conformation (CaBLAM)","CaBLAM ν carbonyl-O virtual dihedral (peptide orientation)",CMD_QC,CPP_BG,"CaBLAM (Prisant et al. 2020)"),
 "cablam_ca_virtual":("Backbone conformation (CaBLAM)","CaBLAM Cα virtual angle",CMD_QC,CPP_BG,"CaBLAM (Prisant et al. 2020)"),
 "clashscore":("Clashes & packing","MolProbity-style all-atom clashscore (overlaps >0.4 Å per 1000 atoms)",CMD_QC,CPP_CL,"MolProbity clashscore (Word et al. 1999; Chen et al. 2010) — re-implemented here"),
}

AGG={"mean_pctile":"mean per-residue percentile vs high-resolution reference (50 = reference median)",
     "sd_pctile":"SD of per-residue percentile across residues",
     "frac_extreme":"fraction of residues with extreme percentile (<5 or >95)",
     "mean_score":"mean per-residue surprise = −log2(two-tailed tail probability)"}
QCSPECIAL={  # name -> (family, description, aggregation, command, implementation, concept)
 "qc_n_residues":("Quality summary","Number of scored residues","count",CMD_QC,CPP_QC,"this work"),
 "qc_residue_score_mean":("Quality summary","Mean per-residue combined surprise score","mean",CMD_QC,CPP_QC,"this work (surprise scoring)"),
 "qc_residue_score_sd":("Quality summary","SD of per-residue combined surprise score","sd",CMD_QC,CPP_QC,"this work"),
 "qc_worst_score_mean":("Quality summary","Mean worst-metric surprise per residue","mean",CMD_QC,CPP_QC,"this work"),
 "qc_n_outliers_mean":("Quality summary","Mean number of metric outliers per residue","mean",CMD_QC,CPP_QC,"this work"),
 "qc_outlier_rate":("Quality summary","Fraction of residues with ≥1 outlier metric","fraction",CMD_QC,CPP_QC,"this work"),
 "qc_frac_H":("Quality summary","Fraction of residues in α-helix","fraction",CMD_QC,"whatres-features C++ (own; secondary-structure assignment)","secondary structure"),
 "qc_frac_E":("Quality summary","Fraction of residues in β-strand","fraction",CMD_QC,"whatres-features C++ (own; secondary-structure assignment)","secondary structure"),
 "qc_frac_C":("Quality summary","Fraction of residues in coil","fraction",CMD_QC,"whatres-features C++ (own; secondary-structure assignment)","secondary structure"),
 "qc_facc_mean":("Quality summary","Mean fractional solvent accessibility","mean",CMD_QC,CPP_SURF,"solvent accessibility"),
 "qc_facc_buried_frac":("Quality summary","Fraction of buried residues","fraction",CMD_QC,CPP_SURF,"solvent accessibility"),
}
SURF={
 "surf_n_chains":"Number of protein chains","surf_n_res_sum":"Total residue count (surface analysis)",
 "surf_n_surf_sum":"Number of surface-exposed residues","surf_n_hphob_patches_sum":"Number of hydrophobic surface patches",
 "surf_n_pos_clusters_sum":"Number of positive charge clusters","surf_n_neg_clusters_sum":"Number of negative charge clusters",
 "surf_hphob1_size_max":"Largest hydrophobic patch size","surf_hphob1_kd_sum_max":"Largest hydrophobic patch Kyte–Doolittle sum",
 "surf_hphob1_rg_max":"Largest hydrophobic patch radius of gyration","surf_hphob1_aromatic_max":"Aromatic count in largest hydrophobic patch",
 "surf_hphob1_entropy_max":"Composition (Shannon) entropy of largest hydrophobic patch","surf_pos1_size_max":"Largest positive cluster size",
 "surf_pos1_charge_max":"Largest positive cluster net charge","surf_pos1_rg_max":"Largest positive cluster radius of gyration",
 "surf_neg1_size_max":"Largest negative cluster size","surf_neg1_charge_max":"Largest negative cluster net charge",
 "surf_neg1_rg_max":"Largest negative cluster radius of gyration","surf_frac_surf_mean":"Mean fraction of residues exposed",
 "surf_mean_burial_mean":"Mean residue burial (Cβ-neighbour count)","surf_net_charge_mean":"Mean surface net charge",
 "surf_frac_surf_hphob_mean":"Mean fraction of hydrophobic surface","surf_hphob1_mean_burial_mean":"Mean burial of largest hydrophobic patch",
 "surf_pos1_mean_burial_mean":"Mean burial of largest positive cluster","surf_neg1_mean_burial_mean":"Mean burial of largest negative cluster",
 "surf_n_cavities":"Number of internal cavities","surf_total_cavity_vol":"Total internal cavity volume",
 "surf_largest_cavity_vol":"Largest cavity volume","surf_protein_volume":"Protein volume","surf_void_fraction":"Void (packing-defect) fraction",
 "surf_n_iface_pairs":"Number of inter-chain interface residue pairs","surf_iface_void_vol_mean":"Mean interface void volume",
 "surf_packing_density_mean":"Mean packing density","surf_void_index_mean":"Mean void index",
}
ATOM={"C":"backbone carbonyl carbon","CA":"α-carbon","Calif":"aliphatic side-chain carbon","Carom":"aromatic side-chain carbon",
 "N":"backbone amide nitrogen","NE":"Arg NE nitrogen","NHARG":"Arg guanidinium nitrogen (NH1/NH2)","NLYS":"Lys NZ nitrogen",
 "NQN":"Asn/Gln side-chain amide nitrogen","Narom":"aromatic-ring nitrogen (His/Trp)","O":"backbone carbonyl oxygen",
 "ODE":"Asp/Glu carboxylate oxygen","OH":"hydroxyl oxygen (Ser/Thr/Tyr)","OQN":"Asn/Gln side-chain carbonyl oxygen",
 "OXT":"C-terminal carboxylate oxygen","SCYS":"Cys sulfur","SMET":"Met sulfur"}
DSTAT={"short":"contacts at short range (<4 Å)","mid":"contacts at mid range (4–6 Å)","percontact":"distance distribution normalised per contact"}
BF={"bf_ca_mean":"Mean Cα B-factor / pLDDT","bf_ca_sd":"SD of Cα B-factor / pLDDT","bf_ca_min":"Min Cα B-factor / pLDDT",
 "bf_ca_max":"Max Cα B-factor / pLDDT","bf_ca_p10":"10th-percentile Cα B-factor / pLDDT","bf_ca_p90":"90th-percentile Cα B-factor / pLDDT",
 "bf_all_mean":"Mean all-atom B-factor / pLDDT","bf_all_sd":"SD all-atom B-factor / pLDDT",
 "bf_looks_plddt":"Flag: B-factor column looks like pLDDT (0–100 range)","bf_ca_frac_lt70":"Fraction of residues with Cα confidence <70"}

COLS=["Parameter","Family","Base feature","Aggregation","Computed by (command)","Implementation (own C++ / Python)","Concept / literature origin","Description"]
rows=[]
for c in cols:
    if c.startswith("dist_"):
        body=c[5:]; stat=body.rsplit("_",1)[1]; pair=body.rsplit("_",1)[0]; a,b=(pair.split("-",1)+[""])[:2]
        rows.append([c,"Atom-type pairwise distances",pair,stat,CMD_DIST,CPP_DIST,"this work (atom-type distance distributions)",
                     f"{ATOM.get(a,a)} – {ATOM.get(b,b)} pairwise distance: {DSTAT.get(stat,stat)}"])
    elif c.startswith("surf_"):
        rows.append([c,"Surface & cavities",c,"per-structure",CMD_SURF,CPP_SURF,"this work (packing/patch/cavity analysis; cf. WHATIF)",SURF.get(c,"surface/cavity descriptor")])
    elif c.startswith("bf_"):
        rows.append([c,"B-factor / model confidence",c,"per-structure",CMD_PY,PY,"PDB B-factor column (pLDDT for predicted models)",BF.get(c,"B-factor statistic")])
    elif c in QCSPECIAL:
        fam,desc,agg,cmd,impl,con=QCSPECIAL[c]; rows.append([c,fam,c,agg,cmd,impl,con,desc])
    elif c.startswith("qc_"):
        agg=next((k for k in AGG if c.endswith(k)),""); m=re.sub(r"_(mean_pctile|sd_pctile|frac_extreme|mean_score)$","",c[3:])
        if m in M:
            fam,bdesc,cmd,impl,con=M[m]; rows.append([c,fam,m,agg,cmd,impl,con,f"{bdesc} — {AGG.get(agg,agg)}"])
        else:
            rows.append([c,"(uncategorised)",m,agg,CMD_QC,CPP_QC,"this work",AGG.get(agg,agg)])
    else:
        rows.append([c,"(other)",c,"","","","",""])
T=pd.DataFrame(rows,columns=COLS)

leg=[("IMPLEMENTATION ENGINES — what actually computed each parameter","")]
leg+=[("whatres-features C++ (own)","our own C++ code: BondGeometry.h (geometry/dihedrals/chirality/CaBLAM), Clashscore.h (clashscore), "
       "SurfacePatcher.h/CavityDetector.h (surface), PDB_dist (distances), StructureReferenceDB.h (reference scoring), "
       "Peptide::SetDihedrals (backbone dihedrals) and NativeQCInputs.h (side-chain burial that assigns each residue its reference cell: occupancy of the side chain by neighbouring heavy atoms, Gaussian-weighted atomic volumes)"),
     ("Python","build_feature_matrix.py — B-factor parsing and per-structure aggregation of whatres-features per-residue output"),
     ("","")]
leg+=[("CONCEPT vs IMPLEMENTATION","'Concept / literature origin' = where the idea comes from (e.g. MolProbity, CaBLAM); "
       "'Implementation' = what we actually ran. E.g. clashscore: concept = MolProbity, implementation = our own Clashscore.h."),("","")]
leg+=[("ATOM-TYPE CODES (dist_ features)","")]+[(k,v) for k,v in ATOM.items()]+[("","")]
leg+=[("DISTANCE-HISTOGRAM SUMMARIES","")]+[(k,v) for k,v in DSTAT.items()]+[("","")]
leg+=[("qc_ AGGREGATIONS (per-structure, computed in Python over whatres-features per-residue output)","")]+[(k,v) for k,v in AGG.items()]
L=pd.DataFrame(leg,columns=["Code","Meaning"])

out=os.path.join(HERE,"native_pipeline","tables","WhatRes Suppl Table 1 Parameters.xlsx"); os.makedirs(os.path.dirname(out),exist_ok=True)
with pd.ExcelWriter(out,engine="openpyxl") as xw:
    T.to_excel(xw,sheet_name="Parameters",index=False); L.to_excel(xw,sheet_name="Legend",index=False)
    for sh,df in [("Parameters",T),("Legend",L)]:
        ws=xw.sheets[sh]
        for i,col in enumerate(df.columns,1):
            ws.column_dimensions[chr(64+i)].width=min(75,max(14,int(df[col].astype(str).str.len().quantile(0.9))+2))
print(f"wrote {out}: {len(T)} parameters")
print("\nImplementation breakdown:")
print(T["Implementation (own C++ / Python)"].apply(lambda s:s.split(' (')[0]).value_counts().to_string())
