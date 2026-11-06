"""Genuine MolProbity score from the official reference implementations/data:
  clashscore = Word's `probe` on reduce-built hydrogens (MolProbity definition, H-bonds excluded)
  rota_out%  = official Top8000 rotamer contour grids (outlier = pct < 0.3%)   [rlabduke reference_data]
  rama_fav%  = official Top8000 Ramachandran contour grids (favored >= 2%)       [rlabduke reference_data]
  MPscore    = Chen et al. 2010 formula
Validated vs phenix.molprobity on Medin: cs 8.83/9.69, rota 0/0, rama_fav 80/80, MP ~2.2/2.25."""
import sys, os, math, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rota_ / rama_molprobity.py
import rota_molprobity as RO, rama_molprobity as RA
import shutil
REDUCE=os.environ.get("REDUCE") or shutil.which("reduce") or "reduce"; PROBE=os.environ.get("PROBE") or shutil.which("probe") or "probe"   # MolProbity reduce and probe
# reduce reads its het dictionary from $REDUCE_HET_DICT (reduce_wwPDB_het_dict.txt, shipped with reduce)
def clashscore(path):
    with tempfile.TemporaryDirectory() as td:
        h=os.path.join(td,"h.pdb")
        with open(h,"w") as fh: subprocess.run([REDUCE,"-quiet","-build",path],stdout=fh,stderr=subprocess.DEVNULL)
        na=sum(1 for l in open(h) if l.startswith(("ATOM","HETATM")))
        out=subprocess.run([PROBE,"-q","-u","-CON","-mc","-het","-once","-self","all",h],capture_output=True,text=True).stdout
    worst={}
    for line in out.splitlines():
        p=line.split(":")
        if len(p)<8: continue
        try: g=float(p[6])
        except ValueError: continue
        k=frozenset((tuple(p[3].split()[:4]),tuple(p[4].split()[:4])))
        if k not in worst or g<worst[k][0]: worst[k]=(g,p[2].strip())
    cl=[k for k,(g,t) in worst.items() if g<=-0.4 and t!="hb"]
    return (len(cl)*1000.0/na if na else float("nan")), na
def mpscore(path):
    cs,na=clashscore(path)
    rot=RO.model(path); nro=sum(1 for *_ ,v in rot if v<0.003); roo=100.0*nro/max(len(rot),1)
    ram=RA.model(path); nfav=sum(1 for x in ram if x[6]=="Favored"); raf=100.0*nfav/max(len(ram),1)
    mp=(0.42574*math.log1p(cs)+0.32996*math.log1p(max(0.0,roo-1.0))
        +0.24979*math.log1p(max(0.0,(100.0-raf)-2.0))+0.5)
    return dict(clashscore=cs,n_atoms=na,rota_out=roo,rama_fav=raf,mpscore=mp,n_rot=len(rot),n_ram=len(ram))
if __name__=="__main__":
    for p in sys.argv[1:]:
        r=mpscore(p)
        print(f"{os.path.basename(p):36s} cs={r['clashscore']:.2f} rota_out={r['rota_out']:.2f}% "
              f"rama_fav={r['rama_fav']:.2f}% MP={r['mpscore']:.2f}  (nrot={r['n_rot']} nram={r['n_ram']})")
