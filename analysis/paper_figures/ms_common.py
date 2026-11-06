"""Shared setup for the manuscript figure producers (ms_*.py).
Every ms_ script writes its panels, named exactly as in the manuscript, to
analysis/native_pipeline/figures/<Figure>/ and records every number it reports via rec()."""
import os as _os; _os.environ.setdefault("WHATRES_MODELS_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models"))
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(AN, "code"))
import figstyle as fs          # noqa: E402  (configures matplotlib for editable vector output)
import figdata as fd           # noqa: E402
import numbers_registry as NR  # noqa: E402
import whatres                 # noqa: E402

OUT = os.path.join(AN, "native_pipeline", "figures")
DATA = os.path.join(AN, "native_pipeline", "data")
os.makedirs(DATA, exist_ok=True)
# Manuscript folder (Figure 1, the Fig. 4g overlay and two supplementary tables are read from it):
# $WHATRES_MANUSCRIPT_DIR, else the path in the untracked file ms_paths_local.txt next to this file,
# else <project root>/manuscript (the layout of the Zenodo deposit).
_LOCAL = os.path.join(HERE, "ms_paths_local.txt")
MS = (os.environ.get("WHATRES_MANUSCRIPT_DIR")
      or (open(_LOCAL).read().strip() if os.path.exists(_LOCAL) else "")
      or os.path.join(os.path.dirname(AN), "manuscript"))

def outdir(fig):
    d = os.path.join(OUT, fig); os.makedirs(d, exist_ok=True); return d

def recorder(script):
    src = os.path.relpath(os.path.abspath(script), AN)
    def rec(key, value, desc, unit=""):
        NR.record(key, value, desc, src, unit)
        return value
    return rec

def realism(df):
    """Deployed realism (0-100) for any feature table."""
    return whatres.realism_score(whatres.realism_load(df))
