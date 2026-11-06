"""numbers_registry.py: the single table of every number the WhatRes manuscript reports.

Every script that computes a reported number calls record(); nothing in the manuscript is typed by
hand. The table is analysis/native_pipeline/numbers.tsv (one row per key, last write wins):
    key, value, unit, description, source, updated
The manuscript is reconciled against this table (native_pipeline/reconcile_numbers.py).
"""
import csv, datetime, os

AN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(AN, "native_pipeline", "numbers.tsv")
COLS = ["key", "value", "unit", "description", "source", "updated"]

def _load():
    if not os.path.exists(PATH): return {}
    with open(PATH) as f: return {r["key"]: r for r in csv.DictReader(f, delimiter="\t")}

def record(key, value, description, source, unit=""):
    """Store one reported number. value may be int/float/str; floats are stored at full precision."""
    rows = _load()
    try:
        import numpy as np
        if isinstance(value, (bool, np.bool_)): value = bool(value)
        elif isinstance(value, np.integer): value = int(value)
        elif isinstance(value, (float, np.floating)): value = repr(float(value))
    except ImportError:
        if isinstance(value, float): value = repr(value)
    rows[key] = {"key": key, "value": str(value), "unit": unit, "description": description,
                 "source": os.path.relpath(source, AN) if os.path.isabs(source) else source,
                 "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t"); w.writeheader()
        for k in sorted(rows): w.writerow(rows[k])
    return value

def get(key):
    r = _load().get(key)
    return None if r is None else r["value"]
