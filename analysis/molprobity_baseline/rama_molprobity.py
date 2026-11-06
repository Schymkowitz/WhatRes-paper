"""MolProbity Top8000 Ramachandran classification (Favored / Allowed / OUTLIER).
Grids: top8000_rama/*.data (rlabduke/reference_data), 2-degree bins centred at odd values,
bilinear interpolation with wrapping.  Thresholds: favored >= 0.02; outlier < 0.0005 (general)
or < 0.001 (other classes); checked against wwPDB XML classifications before use."""
import os, numpy as np
from collections import defaultdict
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top8000", "rama")   # fetch_top8000.sh
FILES = {"general": "rama8000-general-noGPIVpreP.data", "gly": "rama8000-gly-sym.data",
         "transpro": "rama8000-transpro.data", "cispro": "rama8000-cispro.data",
         "prepro": "rama8000-prepro-noGP.data", "ileval": "rama8000-ileval-nopreP.data"}
OUTLIER = {"general": 0.0005}
_grid = {}
def grid(cls):
    if cls not in _grid:
        g = np.zeros((180, 180))
        for l in open(os.path.join(D, FILES[cls])):
            if l.startswith("#") or not l.strip(): continue
            x, y, v = l.split()
            g[int((float(x) + 179) / 2), int((float(y) + 179) / 2)] = float(v)
        _grid[cls] = g
    return _grid[cls]
def value(cls, phi, psi):
    g = grid(cls)
    fx, fy = (phi + 179.0) / 2.0, (psi + 179.0) / 2.0
    i0, j0 = int(np.floor(fx)), int(np.floor(fy)); dx, dy = fx - i0, fy - j0
    v = lambda i, j: g[i % 180, j % 180]
    return ((1 - dx) * (1 - dy) * v(i0, j0) + dx * (1 - dy) * v(i0 + 1, j0) +
            (1 - dx) * dy * v(i0, j0 + 1) + dx * dy * v(i0 + 1, j0 + 1))
def rclass(res3, next3, omega_prev=None):
    if res3 == "GLY": return "gly"
    if res3 == "PRO": return "cispro" if omega_prev is not None and abs(omega_prev) < 30 else "transpro"
    if next3 == "PRO": return "prepro"
    if res3 in ("ILE", "VAL"): return "ileval"
    return "general"
def classify(cls, phi, psi):
    v = value(cls, phi, psi)
    if v >= 0.02: return "Favored", v
    if v < OUTLIER.get(cls, 0.001): return "OUTLIER", v
    return "Allowed", v
def dih(p0, p1, p2, p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2; b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))
def model(path):
    """yield (chain, resnum, res3, phi, psi, class, verdict, value) for a PDB file"""
    R = defaultdict(dict); name = {}
    for l in open(path):
        if l.startswith("ATOM"):
            k = (l[21], int(l[22:26])); R[k][l[12:16].strip()] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            name[k] = l[17:20]
    out = []
    for (c, r) in sorted(R):
        p, n = R.get((c, r - 1)), R.get((c, r + 1)); a = R[(c, r)]
        # as MolProbity: evaluate only residues with C(i-1), N/CA/C(i) and N(i+1) present (incomplete neighbours are skipped)
        if not p or not n or not all(x in a for x in ("N", "CA", "C")) or "C" not in p or "N" not in n: continue
        if np.linalg.norm(p["C"] - a["N"]) > 2.0 or np.linalg.norm(a["C"] - n["N"]) > 2.0: continue
        phi, psi = dih(p["C"], a["N"], a["CA"], a["C"]), dih(a["N"], a["CA"], a["C"], n["N"])
        omega_prev = dih(p["CA"], p["C"], a["N"], a["CA"]) if "CA" in p else None   # cis/trans Pro needs CA(i-1)
        cls = rclass(name[(c, r)], name.get((c, r + 1)), omega_prev)
        verdict, v = classify(cls, phi, psi)
        out.append((c, r, name[(c, r)], phi, psi, cls, verdict, v))
    return out
if __name__ == "__main__":
    import sys
    from collections import Counter
    res = model(sys.argv[1])
    print(sys.argv[1], dict(Counter(x[6] for x in res)), f"outliers {100*sum(x[6]=='OUTLIER' for x in res)/len(res):.2f}%")
