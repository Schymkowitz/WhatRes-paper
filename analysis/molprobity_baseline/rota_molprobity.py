"""MolProbity Top8000 rotamer percentile per residue (reference implementation for checking).
Grids: Agadir_wrapper2026/data/top8000_rotamer (rlabduke/reference_data, CC BY 4.0); bounds, bins and
wrapping read from each file's header; sparse (missing bins = 0); n-linear interpolation."""
import os, re, sys, itertools, numpy as np
from collections import defaultdict
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top8000", "rotamer")   # fetch_top8000.sh
FILE = {"ARG": "arg", "ASN": "asn", "ASP": "asp", "CYS": "cys", "GLN": "gln", "GLU": "glu", "HIS": "his", "ILE": "ile",
        "LEU": "leu", "LYS": "lys", "MET": "met", "PHE": "phetyr", "TYR": "phetyr", "PRO": "pro", "SER": "ser",
        "THR": "thr", "TRP": "trp", "VAL": "val"}
CHI = {"SER": ["N CA CB OG"], "THR": ["N CA CB OG1"], "CYS": ["N CA CB SG"], "VAL": ["N CA CB CG1"],
       "ILE": ["N CA CB CG1", "CA CB CG1 CD1"], "LEU": ["N CA CB CG", "CA CB CG CD1"],
       "PHE": ["N CA CB CG", "CA CB CG CD1"], "TYR": ["N CA CB CG", "CA CB CG CD1"], "TRP": ["N CA CB CG", "CA CB CG CD1"],
       "HIS": ["N CA CB CG", "CA CB CG ND1"], "ASN": ["N CA CB CG", "CA CB CG OD1"], "ASP": ["N CA CB CG", "CA CB CG OD1"],
       "GLN": ["N CA CB CG", "CA CB CG CD", "CB CG CD OE1"], "GLU": ["N CA CB CG", "CA CB CG CD", "CB CG CD OE1"],
       "MET": ["N CA CB CG", "CA CB CG SD", "CB CG SD CE"], "PRO": ["N CA CB CG"],
       "LYS": ["N CA CB CG", "CA CB CG CD", "CB CG CD CE", "CG CD CE NZ"],
       "ARG": ["N CA CB CG", "CA CB CG CD", "CB CG CD NE", "CG CD NE CZ"]}
_g = {}
def grid(name):
    if name in _g: return _g[name]
    dims, vals = [], {}
    for l in open(os.path.join(D, f"rota8000-{name}.data")):
        m = re.match(r"#\s+x\d+:\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)", l)
        if m: dims.append((float(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(4) == "true")); continue
        if l.startswith("#") or not l.strip(): continue
        f = l.split(); x = [float(v) for v in f[:-1]]
        idx = tuple(int(round((x[k] - dims[k][0]) / ((dims[k][1] - dims[k][0]) / dims[k][2]) - 0.5)) for k in range(len(dims)))
        vals[idx] = float(f[-1])
    _g[name] = (dims, vals); return _g[name]
def value(name, chis):
    dims, vals = grid(name)
    pos, base = [], []
    for k, (lo, hi, n, wrap) in enumerate(dims):
        span = hi - lo; c = chis[k]
        c = (c - lo) % span + lo                       # into [lo, hi) (folds symmetric 0-180 dims)
        f = (c - lo) / (span / n) - 0.5                # continuous bin index (bin centres at integers)
        i0 = int(np.floor(f)); pos.append(f - i0); base.append(i0)
    tot = 0.0
    for corner in itertools.product((0, 1), repeat=len(dims)):
        w = 1.0; idx = []
        for k, b in enumerate(corner):
            w *= pos[k] if b else 1 - pos[k]
            n = dims[k][2]; idx.append((base[k] + b) % n)
        tot += w * vals.get(tuple(idx), 0.0)
    return tot
def dih(p0, p1, p2, p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2; b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
def model(path):
    R = defaultdict(dict); nm = {}
    for l in open(path):
        if l.startswith("ATOM"):
            k = (l[21], int(l[22:26])); R[k][l[12:16].strip()] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]); nm[k] = l[17:20]
    out = []
    for k, a in sorted(R.items()):
        r3 = nm[k]
        if r3 not in CHI: continue
        try: chis = [dih(*[a[x] for x in q.split()]) for q in CHI[r3]]
        except KeyError: continue
        out.append((k[0], k[1], r3, chis, value(FILE[r3], chis)))
    return out
if __name__ == "__main__":
    for c, r, r3, chis, v in model(sys.argv[1])[:10]: print(c, r, r3, np.round(chis, 1), round(v, 5))
