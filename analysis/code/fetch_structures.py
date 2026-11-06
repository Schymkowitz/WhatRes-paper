#!/usr/bin/env python3.14
"""fetch_structures.py: idempotent, checksummed fetch of RCSB entries into a cohort folder.

usage: fetch_structures.py --ids <file> --out <cohort dir> [--root <HighResPDB>]

  --ids   text/TSV file; first column = PDB id (a header line 'id' is skipped)
  --out   cohort folder, e.g. HighResPDB/structures/<cohort>/

Writes, inside --out:
  <id>.pdb        asymmetric unit, lowercase id (PDB format; entries without a
                  PDB-format file are fetched as mmCIF and converted with gemmi)
  MANIFEST.tsv    one row per entry: id, file, source_url, format, bytes, sha256,
                  fetched_utc, method, resolution, deposit_date, release_date
  list.txt        file paths relative to --root, ready for build_feature_matrix.py
A file already present whose sha256 matches MANIFEST.tsv is kept and never fetched
again. Failures are listed and the script exits non-zero.
"""
import argparse, csv, datetime, hashlib, json, os, subprocess, sys, tempfile

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def curl(url, dest):
    r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", dest, url])
    return r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1000

def metadata(ids):
    out = {}
    for i in range(0, len(ids), 100):
        q = '{entries(entry_ids:%s){rcsb_id rcsb_entry_info{resolution_combined experimental_method} ' \
            'rcsb_accession_info{deposit_date initial_release_date}}}' % json.dumps([x.upper() for x in ids[i:i + 100]])
        r = subprocess.run(["curl", "-sf", "-X", "POST", "https://data.rcsb.org/graphql",
                            "-H", "Content-Type: application/json", "-d", json.dumps({"query": q})],
                           capture_output=True, text=True)
        if r.returncode != 0: continue
        for e in json.loads(r.stdout)["data"]["entries"] or []:
            if not e: continue
            info, acc = e.get("rcsb_entry_info") or {}, e.get("rcsb_accession_info") or {}
            res = info.get("resolution_combined")
            out[e["rcsb_id"].lower()] = {"method": info.get("experimental_method", ""),
                                         "resolution": res[0] if res else "",
                                         "deposit_date": (acc.get("deposit_date") or "")[:10],
                                         "release_date": (acc.get("initial_release_date") or "")[:10]}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    a = ap.parse_args()
    ids = []
    for ln in open(a.ids):
        t = ln.strip().split("\t")[0].strip().lower()
        if t and t != "id": ids.append(t)
    ids = list(dict.fromkeys(ids))
    os.makedirs(a.out, exist_ok=True)
    man_p = os.path.join(a.out, "MANIFEST.tsv")
    old = {r["id"]: r for r in csv.DictReader(open(man_p), delimiter="\t")} if os.path.exists(man_p) else {}
    meta = metadata(ids)
    rows, failed, fetched, kept = [], [], 0, 0
    for i in ids:
        dest = os.path.join(a.out, f"{i}.pdb")
        if i in old and os.path.exists(dest) and sha256(dest) == old[i]["sha256"]:
            rows.append(old[i]); kept += 1; continue
        url, fmt = f"https://files.rcsb.org/download/{i.upper()}.pdb", "pdb"
        if not curl(url, dest):
            url, fmt = f"https://files.rcsb.org/download/{i.upper()}.cif", "cif->pdb"
            with tempfile.TemporaryDirectory() as td:
                cif = os.path.join(td, f"{i}.cif")
                ok = curl(url, cif)
                if ok:
                    import gemmi
                    st = gemmi.read_structure(cif); st.setup_entities(); st.write_pdb(dest)
            if not (os.path.exists(dest) and os.path.getsize(dest) > 1000):
                failed.append(i); continue
        fetched += 1
        m = meta.get(i, {})
        rows.append({"id": i, "file": os.path.basename(dest), "source_url": url, "format": fmt,
                     "bytes": os.path.getsize(dest), "sha256": sha256(dest),
                     "fetched_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "method": m.get("method", ""), "resolution": m.get("resolution", ""),
                     "deposit_date": m.get("deposit_date", ""), "release_date": m.get("release_date", "")})
    cols = ["id", "file", "source_url", "format", "bytes", "sha256", "fetched_utc",
            "method", "resolution", "deposit_date", "release_date"]
    with open(man_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(rows)
    rel = os.path.relpath(a.out, a.root)
    with open(os.path.join(a.out, "list.txt"), "w") as f:
        for r in rows: f.write(f"{rel}/{r['file']}\n")
    print(f"[fetch] {len(ids)} ids: fetched {fetched}, already present {kept}, failed {len(failed)} {failed}")
    print(f"[fetch] manifest {man_p}; list {os.path.join(a.out, 'list.txt')}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
