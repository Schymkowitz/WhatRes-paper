#!/usr/bin/env bash
# Era analysis (Supplementary Fig. 5): feature table of the 100 era structures.
#   WHATRES_FEATURES=/path/to/whatres-features ./rebuild_era.sh
#   structures: structures/era_before_after_1995/  (MANIFEST.tsv = provenance + sha256; fetched if absent)
#   ids/years : analysis/era_before_after_1995/era_years.tsv
#   output    : analysis/era_before_after_1995/era_fm.tsv
set -uo pipefail
P="$(cd "$(dirname "$0")" && pwd)"; A="$(dirname "$P")"; H="$(dirname "$A")"; PY="${PYTHON:-python3.14}"
: "${WHATRES_FEATURES:?set WHATRES_FEATURES to the whatres-features binary}"
SCR="${WHATRES_SCRATCH:-$(mktemp -d /tmp/whatres_era.XXXXXX)}"              # must be space-free
case "$SCR" in *" "*) echo "WHATRES_SCRATCH must be a space-free path"; exit 1;; esac
S="$H/structures/era_before_after_1995"; mkdir -p "$P/logs" "$SCR"

# 1. structures: idempotent (files with a matching sha256 are never fetched again)
"$PY" "$A/code/fetch_structures.py" --ids "$A/era_before_after_1995/era_years.tsv" --out "$S" --root "$H" \
  || { echo "[era] fetch incomplete: STOP"; exit 1; }

# 2. feature table
OPT="$SCR/Options_features.txt"
sed "s|<qc_reference_db>.*;|<qc_reference_db>$A/refbuild/native49_structure_reference.bin;|" "$A/Options_features_native.txt" > "$OPT"
rm -rf "$SCR/era"
"$PY" -m whatres.features --bin "$WHATRES_FEATURES" --list "$S/list.txt" --root "$H" \
  --opt "$OPT" --out "$A/era_before_after_1995/era_fm.tsv" \
  --scratch "$SCR/era" --workdir "$SCR/era_work" > "$P/logs/era.log" 2>&1 || { echo "[era] feature build failed"; exit 1; }

# 3. completeness: all era ids present
"$PY" - "$A" <<'PY' || exit 1
import sys, pandas as pd
A = sys.argv[1]
want = set(pd.read_csv(f"{A}/era_before_after_1995/era_years.tsv", sep="\t").id.astype(str).str.lower())
got = set(pd.read_csv(f"{A}/era_before_after_1995/era_fm.tsv", sep="\t", usecols=["id"]).id.astype(str).str.lower())
print(f"[era] ids {len(want)} | in table {len(got & want)} | missing {sorted(want - got)}")
sys.exit(1 if want - got else 0)
PY
echo "[era] done $(date '+%F %T')"
