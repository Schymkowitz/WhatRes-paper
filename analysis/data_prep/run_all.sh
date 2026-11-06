#!/usr/bin/env bash
# Data preparation: every derived input of the WhatRes analysis, re-made from the structures and the
# deposited cohort lists and checked against the files in use. Runs before native_pipeline/rebuild_all.sh.
#
#   ./run_all.sh            re-make into a staging folder and CHECK (default; changes nothing)
#   ./run_all.sh --install  re-make, check, and on success copy the results into place
#
# Needs: python3.14 with gemmi and pandas; mmseqs ($MMSEQS); reduce and probe ($REDUCE, $PROBE);
# the predictor confidence summaries ($WHATRES_PREDICTIONS). Space-free staging: $WHATRES_SCRATCH.
#
#   step                     makes                                          check
#   check_cohorts.py         (nothing)                                      cohort lists meet their stated criteria
#   validation_set.py        val_resolution, val_pdbs/, val_list            identical rows / entries
#   cohorts.py               benchmark_existing, exact_resolution,          identical to the files in use
#                            ai_topmodel_meta, ai_list, era_years
#   sequences.py             sequence clusters (MMseqs2)                    identical; FASTA re-extraction listed
#   molprobity_benchmark.py  molprobity_train_genuine.tsv (+ failures)      same structures, values within 1e-9
#   extract_geometry.py      geom_residues.tsv.gz, geom_structures.tsv      identical
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; A="$(dirname "$D")"; PY="${PYTHON:-python3.14}"
INSTALL=0; [ "${1:-}" = "--install" ] && INSTALL=1
OUT="${WHATRES_SCRATCH:-$(mktemp -d /tmp/whatres_dataprep.XXXXXX)}"; mkdir -p "$OUT"
case "$OUT" in *" "*) echo "WHATRES_SCRATCH must be a space-free path"; exit 1;; esac
fail=0; step() { echo "== $1"; shift; "$@" || { echo "   FAILED"; fail=1; }; }

step "cohort criteria"     "$PY" "$D/check_cohorts.py"
step "validation set"    "$PY" "$D/validation_set.py" --out "$OUT/validation" --check
step "cohort tables"       "$PY" "$D/cohorts.py" --out "$OUT/cohorts" --check
step "sequence clusters"   "$PY" "$D/sequences.py" --out "$OUT/seq" --check
"$A/molprobity_baseline/fetch_top8000.sh" >/dev/null || fail=1
step "MolProbity baseline" bash -c "cd '$OUT' && '$PY' '$D/molprobity_benchmark.py' --out '$OUT/molprobity' --check"
step "backbone geometry"   bash -c "mkdir -p '$OUT/geom' && cd '$OUT/geom' && GEOM_OUT='$OUT/geom' '$PY' '$A/over_regularisation_geometry/extract_geometry.py' | grep -E 'listed|total' \
  && '$PY' -c \"import pandas as pd,sys; a=pd.read_csv('$A/over_regularisation_geometry/geom_residues.tsv.gz',sep='\t'); b=pd.read_csv('$OUT/geom/geom_residues.tsv.gz',sep='\t'); ok=a.equals(b); print('  [check] geom_residues.tsv.gz', 'identical' if ok else 'DIFFERS'); sys.exit(0 if ok else 1)\""

[ $fail -eq 0 ] || { echo "data preparation: checks FAILED (staging $OUT); nothing installed"; exit 1; }
if [ $INSTALL -eq 1 ]; then
  cp "$OUT/cohorts/benchmark_existing.txt" "$OUT/cohorts/ai_list.txt" "$A/refbuild/"
  cp "$OUT/cohorts/exact_resolution.tsv" "$OUT/cohorts/ai_topmodel_meta.tsv" "$A/"
  cp "$OUT/cohorts/era_years.tsv" "$A/era_before_after_1995/"
  cp "$OUT/validation/val_resolution.tsv" "$OUT/validation/val_list.txt" "$A/dms_hero/"
  cp "$OUT/seq/mmseqs_clusters_id30_cov80.tsv" "$A/seq_clusters/"
  cp "$OUT/molprobity/molprobity_train_genuine.tsv" "$OUT/molprobity/molprobity_failed.tsv" "$A/molprobity_baseline/"
  cp "$OUT/geom/geom_residues.tsv.gz" "$OUT/geom/geom_structures.tsv" "$A/over_regularisation_geometry/"
  echo "data preparation: all checks passed; installed"
else
  echo "data preparation: all checks passed (staging $OUT; run with --install to copy into place)"
fi
