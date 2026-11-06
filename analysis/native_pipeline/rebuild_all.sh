#!/usr/bin/env bash
# Rebuild every WhatRes feature table from the structures, then retrain both models.
#   WHATRES_FEATURES=/path/to/whatres-features ./rebuild_all.sh   (see the whatres-features repository)
# Structures: the structures_*.zip files of the Zenodo deposit, unpacked into the project root.
#   1. options with this checkout's reference DB
#   2. the four feature tables (in parallel)
#   3. completeness check: every structure in each list has a row (nothing dropped silently)
#   4. retrain apparent resolution, then realism
set -uo pipefail
P="$(cd "$(dirname "$0")" && pwd)"; A="$(dirname "$P")"; H="$(dirname "$A")"; PY="${PYTHON:-python3.14}"
: "${WHATRES_FEATURES:?set WHATRES_FEATURES to the whatres-features binary}"
SCR="${WHATRES_SCRATCH:-$(mktemp -d /tmp/whatres_rebuild.XXXXXX)}"          # must be space-free
case "$SCR" in *" "*) echo "WHATRES_SCRATCH must be a space-free path"; exit 1;; esac
mkdir -p "$P/logs" "$SCR"; cd "$A"
{ echo "rebuild $(date '+%F %T')"; echo "whatres-features $WHATRES_FEATURES sha1 $(shasum "$WHATRES_FEATURES" | cut -c1-40)"; } > "$P/logs/recipe.log"

# 1. options
OPT="$SCR/Options_features.txt"
sed "s|<qc_reference_db>.*;|<qc_reference_db>$A/refbuild/native49_structure_reference.bin;|" "$A/Options_features_native.txt" > "$OPT"

# 2. feature tables: name | list | output
pick_root() {   # the root under which EVERY list entry exists
  for r in "$H" "$H/benchmark" "$A"; do
    ok=1; while read -r p; do [ -z "$p" ] && continue; [ -s "$r/$p" ] || { ok=0; break; }; done < "$1"
    [ $ok = 1 ] && { echo "$r"; return; }
  done; echo NONE; }
TABLES=("bench|$A/refbuild/benchmark_existing.txt|$A/feature_matrix.tsv"
        "ai|$A/refbuild/ai_list.txt|$A/ai_feature_matrix.tsv"
        "aiall|$A/refbuild/ai_all_list.txt|$A/ai_all_feature_matrix.tsv"
        "val|$A/dms_hero/val_list.txt|$A/dms_hero/val_feature_matrix.tsv")
for t in "${TABLES[@]}"; do
  IFS='|' read -r name list out <<< "$t"; root=$(pick_root "$list")
  echo "[$name] list $(wc -l < "$list") root $root" | tee -a "$P/logs/recipe.log"
  [ "$root" = NONE ] && { echo "[$name] some list entries are missing on disk: STOP"; exit 1; }
  rm -rf "$SCR/$name"
  "$PY" -m whatres.features --bin "$WHATRES_FEATURES" --list "$list" --root "$root" \
     --opt "$OPT" --out "$out" --scratch "$SCR/$name" --workdir "$SCR/${name}_work" \
     > "$P/logs/$name.log" 2>&1 &
done
wait; echo "[tables] all builds finished"
# keep the benchmark per-residue quality report (read by Supplementary Fig. 6)
Q=$(ls "$SCR"/bench/qc/*quality_report.tsv 2>/dev/null | head -1)
[ -n "$Q" ] || { echo "[bench] per-residue quality report not found: STOP"; exit 1; }
mkdir -p "$P/data"; gzip -nc "$Q" > "$P/data/bench_quality_report.tsv.gz.tmp" && mv "$P/data/bench_quality_report.tsv.gz.tmp" "$P/data/bench_quality_report.tsv.gz"
echo "[bench] per-residue quality report kept: $(( $(wc -l < "$Q") - 1 )) residues"

# 3. completeness: every list entry must have a row
"$PY" - "$A" <<'PY' || exit 1
import os, sys, pandas as pd
A = sys.argv[1]; bad = 0
for lst, tab in [("refbuild/benchmark_existing.txt", "feature_matrix.tsv"), ("refbuild/ai_list.txt", "ai_feature_matrix.tsv"),
                 ("refbuild/ai_all_list.txt", "ai_all_feature_matrix.tsv"), ("dms_hero/val_list.txt", "dms_hero/val_feature_matrix.tsv")]:
    want = {os.path.splitext(os.path.basename(l.strip()))[0].lower() for l in open(f"{A}/{lst}") if l.strip()}
    got = set(pd.read_csv(f"{A}/{tab}", sep="\t", usecols=["id"]).id.astype(str).str.lower())
    miss = sorted(want - got); bad += len(miss)
    print(f"[check] {tab}: {len(want)} structures listed, {len(got & want)} in the table, missing {len(miss)} {miss[:8]}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || { echo "[check] structures missing: STOP before retraining"; exit 1; }

# 4. retrain (apparent resolution first: build_realism imports whatres, which loads that model)
"$PY" "$A/code/build_apparent_resolution.py" > "$P/logs/train_appres.log" 2>&1 || { echo "apparent-resolution training failed"; exit 1; }
"$PY" "$A/code/build_realism.py"             > "$P/logs/train_realism.log" 2>&1 || { echo "realism training failed"; exit 1; }
echo "[done] $(date '+%F %T')"
