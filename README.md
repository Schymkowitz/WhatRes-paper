# WhatRes-paper

Code that trains the WhatRes models and reproduces every number, figure and table of the paper
"Realism and resolution: how AI-predicted protein structures differ from experimental structures".
The tool itself (feature program, scorer, command line) is in
[WhatRes](https://github.com/Schymkowitz/WhatRes); this repository uses it and holds no copy of it.

## Setup

```bash
git clone https://github.com/Schymkowitz/WhatRes.git            # the tool
make -C WhatRes/whatres-features && pip install -e WhatRes
git clone https://github.com/Schymkowitz/WhatRes-paper.git && cd WhatRes-paper
pip install -r requirements.txt
# data: unpack the Zenodo deposit (https://doi.org/10.5281/zenodo.24014748) into this folder; it adds the structures
# (benchmark/, pdbs/, structures/) and the large tables under analysis/
export WHATRES_FEATURES=$PWD/../WhatRes/whatres-features/whatres-features
```

External programs: MMseqs2 (`mmseqs`), MolProbity `reduce` and `probe` (with `$REDUCE_HET_DICT`).

## Run order

| step | command | makes |
|---|---|---|
| 1 | `analysis/data_prep/run_all.sh` | checks the cohort lists against their criteria and re-makes every derived input (resolutions, top models, sequence clusters, MolProbity baseline, backbone geometry), comparing each with the deposited file; `--install` writes them |
| 2 | `analysis/native_pipeline/rebuild_all.sh` and `rebuild_era.sh` | the feature tables, from the structures; then both models are retrained |
| 3 | `analysis/native_pipeline/regenerate_all.sh` | every number (`native_pipeline/numbers.tsv`), figure and table |
| 4 | `analysis/binders/fetch_binder_dataset.py`, `score_binders.py`, `foldx_complexes.py`, then step 3 again | the de novo binder analysis (Supplementary Fig. 7): fetches the Overath et al. 2025 dataset (Zenodo 10.5281/zenodo.15722219, CC BY 4.0, 2.4 GB), scores it with WhatRes and computes FoldX AnalyseComplex energies (FoldX, academic licence from foldxsuite.crg.eu); without it step 3 skips Supplementary Fig. 7 and says so |

Every reported number is written by the script that computes it to `analysis/native_pipeline/numbers.tsv`;
the copy in this repository is the reference for checking a re-run.

Selections made once (the 2,000-structure reference set, the benchmark design, the 120 prediction
targets, the 100 era structures and the validation set) are records: the lists are in this repository
and `analysis/data_prep/check_cohorts.py` verifies each against the criteria it was drawn by.
Figure 1 and the structure overlay of Fig. 4g are drawn by hand.

## Layout

| folder | |
|---|---|
| `analysis/code/` | model training (`whatres_train.py`: the single training implementation) |
| `analysis/data_prep/` | step 1 |
| `analysis/native_pipeline/` | steps 2 and 3, the number registry |
| `analysis/paper_figures/` | one script per figure group and table |
| `analysis/over_regularisation_geometry/` | coordinate-only backbone geometry (Fig. 2c-f, Supp. Fig. 3) |
| `analysis/molprobity_baseline/` | the MolProbity-score baseline |
| `analysis/refbuild/`, `dms_hero/`, `era_before_after_1995/`, `seq_clusters/`, `prediction_set/` | cohort lists |
| `benchmark/` | benchmark design and the predictors' confidence summaries |

## Licence

BSD 3-Clause (`LICENSE`). The Top8000 reference grids used by the MolProbity baseline are fetched
from [rlabduke/reference_data](https://github.com/rlabduke/reference_data) (CC BY 4.0).
