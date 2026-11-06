#!/usr/bin/env bash
# Regenerate every WhatRes manuscript number, figure and table from the feature tables.
# Order matters: apparent resolution -> realism (uses its out-of-fold values) -> figures -> composites.
# Prerequisite: rebuild_all.sh + rebuild_era.sh + strip_legacy_columns.py (the feature tables).
set -euo pipefail
A="$(cd "$(dirname "$0")/.." && pwd)"; P="$A/native_pipeline"; cd "$A"; PY="${PYTHON:-python3.14}"
mkdir -p "$P/logs"; rm -f "$P/numbers.tsv"
run() { echo "== $1"; "$PY" "$1" > "$P/logs/$(basename "$1" .py).log" 2>&1 || { echo "FAILED: $1"; tail -20 "$P/logs/$(basename "$1" .py).log"; exit 1; }; grep -E '^\[|removed' "$P/logs/$(basename "$1" .py).log" | tail -8; }
run code/build_apparent_resolution.py
run code/build_realism.py
run paper_figures/ms_fig2.py
run paper_figures/ms_fig3.py
run paper_figures/ms_fig4.py
run paper_figures/ms_misc.py
run paper_figures/ms_tables.py
run paper_figures/ms_table_s2.py
run paper_figures/ms_supp.py
run paper_figures/ms_supp3.py
run paper_figures/ms_extra.py
run paper_figures/compose_figures.py
echo "== done $(date '+%F %T')"
