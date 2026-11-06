# Genuine MolProbity-score baseline

Computed with the official MolProbity programs/data, not a reimplementation:
- clashscore: Word's `probe` (v2.26) on `reduce`-built hydrogens, MolProbity definition (overlap <= -0.4 A, H-bonds excluded)
- rotamer-outlier %: official Top8000 rotamer contour grids (rlabduke reference_data); outlier = pct < 0.3%
- Ramachandran-favored %: official Top8000 Ramachandran contour grids; favored >= 2%
- MolProbity score: Chen et al. 2010 formula = 0.42574 ln(1+clash) + 0.32996 ln(1+max(0,rota_out-1)) + 0.24979 ln(1+max(0,rama_notfav-2)) + 0.5

Validation vs phenix.molprobity on Medin Morphology-2: cs 8.83/9.69, rota 0/0 %, rama_fav 82/80 %, MPscore 2.18/2.25.
Pipeline: mp_genuine_pipeline.py. Full MolProbity suite installed in conda env `mp` (cctbx-base 2026.9) if the single-command run is ever needed.
