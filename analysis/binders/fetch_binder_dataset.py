#!/usr/bin/env python3.14
"""Fetch the de novo binder meta-analysis dataset (Overath et al. 2025, bioRxiv 10.1101/2025.08.14.670059;
data: Zenodo 10.5281/zenodo.15722219, CC BY 4.0). Idempotent: a file whose md5 matches the Zenodo record is
never fetched again. Archives are unpacked next to themselves (once).

usage: fetch_binder_dataset.py [--out DIR] [--files f1 f2 ...]     (default DIR: ~/data/binder_meta_2025)"""
import argparse, hashlib, json, os, subprocess, sys, tarfile, time

REC = "15722219"
DEFAULT = ["final_dataset.csv", "input_pdbs.tar.zst", "AF3_outputs.tar.zstd"]

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.expanduser("~/data/binder_meta_2025"))
    ap.add_argument("--files", nargs="*", default=DEFAULT); a = ap.parse_args()
    raw = os.path.join(a.out, "raw"); os.makedirs(raw, exist_ok=True)
    rec = json.loads(subprocess.run(["curl", "-sf", f"https://zenodo.org/api/records/{REC}"], capture_output=True, text=True, check=True).stdout)
    files = {f["key"]: f for f in rec["files"]}
    man = {"record": f"10.5281/zenodo.{REC}", "licence": rec["metadata"]["license"]["id"], "files": {}}
    for k in a.files:
        f = files[k]; dst = os.path.join(raw, k); want = f["checksum"].split(":", 1)[1]
        if not (os.path.exists(dst) and os.path.getsize(dst) == f["size"] and md5(dst) == want):
            t0 = time.time(); print(f"[fetch] {k} ({f['size']/1e6:.0f} MB)", flush=True)
            subprocess.run(["curl", "-sfL", "--retry", "5", "-C", "-", "-o", dst, f["links"]["self"]], check=True)
            if md5(dst) != want: sys.exit(f"STOP: md5 mismatch for {k}")
            print(f"[fetch] {k} verified in {time.time()-t0:.0f} s", flush=True)
        else: print(f"[fetch] {k} present and verified")
        man["files"][k] = {"bytes": f["size"], "md5": want}
        if k.endswith((".tar.zst", ".tar.zstd")):
            stem = os.path.join(a.out, k.split(".tar")[0]); done = stem + ".unpacked"
            if not os.path.exists(done):
                print(f"[unpack] {k} -> {stem}", flush=True)
                with tarfile.open(dst, "r:zst") as tf: tf.extractall(stem, filter="data")
                n = sum(len(fs) for _, _, fs in os.walk(stem)); open(done, "w").write(f"{n} files\n"); print(f"[unpack] {n} files")
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(f"[done] {a.out}")

if __name__ == "__main__":
    main()
