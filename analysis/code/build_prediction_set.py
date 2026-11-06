#!/usr/bin/env python3
"""
build_prediction_set.py
=======================

Compose a leakage-stratified protein target set for a teammate to generate
AlphaFold3 + Boltz-2 structures, for the structure-quality / resolution-
equivalence project.

Each target is tagged on independent leakage axes so we can later ask: does the
predictor's accuracy/fingerprint depend on whether it had *seen* the protein (or
a homolog) during training?

Axes
----
TEMPORAL (was a structure of THIS protein in training?), from the target's own
earliest experimental-structure release date `d`:
  seen_all              d <= 2021-09-30   (in AF3, Boltz-1 AND Boltz-2 training)
  held_af3_seen_boltz2  2021-09-30 < d <= 2023-06-01  (blind to AF3/Boltz-1)
  held_all              d  > 2023-06-01   (blind to all three — gold-standard)

HOMOLOGY (was a homolog of known structure in training?), from the best %identity
to any PDB structure released <= 2021-09-30, excluding near-identical self-hits:
  close_homolog   best non-self identity >= 0.40
  remote_homolog  0.20 <= best non-self identity < 0.40
  orphan          no detectable pre-cutoff homolog (< 0.20 / no hit)

GROUND TRUTH:
  has_expt   target has an experimental structure (enables accuracy + true
             resolution comparison); resolution_tag = high/mid/low
  no_expt    no experimental structure at all (prediction only)

Outputs (under --out-dir, default analysis/prediction_set/):
  prediction_manifest.tsv   one row per target, all tags
  prediction_targets.fasta  unique sequences for the predictors
  GENERATE_PREDICTIONS.md   instructions for the teammate (AF3 + Boltz-2)
  _api_cache.json           cached API responses (safe to delete; enables resume)

Cutoffs are configurable; defaults from the AF3/Boltz papers (see project memory).
"""

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

# ── cutoffs ───────────────────────────────────────────────────────────────────
AF3_CUT = "2021-09-30"      # AlphaFold3 + Boltz-1 PDB training cutoff
BOLTZ2_CUT = "2023-06-01"   # Boltz-2 PDB training cutoff

SELF_ID = 0.98              # identity >= this counts as "self", not a homolog
CLOSE_ID = 0.40
REMOTE_ID = 0.20

# ── SSL (macOS python often lacks CA certs) ───────────────────────────────────
_ctx = ssl.create_default_context()
try:
    import certifi
    _ctx.load_verify_locations(certifi.where())
except Exception:
    _ctx.check_hostname = False
    _ctx.verify_mode = ssl.CERT_NONE

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{}"
DATA_ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}"

# ── simple on-disk cache ──────────────────────────────────────────────────────
_CACHE = {}
_CACHE_PATH = None


def cache_load(path):
    global _CACHE, _CACHE_PATH
    _CACHE_PATH = path
    if os.path.exists(path):
        try:
            _CACHE = json.load(open(path))
        except Exception:
            _CACHE = {}


def cache_save():
    if _CACHE_PATH:
        json.dump(_CACHE, open(_CACHE_PATH, "w"))


def _get_json(url, key, timeout=30):
    if key in _CACHE:
        return _CACHE[key]
    try:
        r = urllib.request.urlopen(url, timeout=timeout, context=_ctx)
        d = json.loads(r.read().decode())
    except Exception as e:
        d = {"__error__": str(e)}
    _CACHE[key] = d
    return d


def _post_json(url, payload, key, timeout=60):
    if key in _CACHE:
        return _CACHE[key]
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout, context=_ctx)
        body = r.read().decode()
        # RCSB search returns HTTP 204 / empty body when there are ZERO hits.
        if r.status == 204 or not body.strip():
            d = {"result_set": [], "total_count": 0}
        else:
            d = json.loads(body)
    except urllib.error.HTTPError as e:
        d = {"__error__": f"HTTP {e.code}: {e.read().decode()[:150]}"}
    except Exception as e:
        d = {"__error__": str(e)}
    _CACHE[key] = d
    return d


# ── RCSB queries ──────────────────────────────────────────────────────────────

def entry_meta(pdbid):
    """Return dict: release_date (YYYY-MM-DD), method, resolution (float|None)."""
    d = _get_json(DATA_ENTRY.format(pdbid), f"entry:{pdbid}")
    if "__error__" in d:
        return None
    rel = d.get("rcsb_accession_info", {}).get("initial_release_date", "")[:10]
    methods = [m.get("method") for m in d.get("exptl", [])]
    res = d.get("rcsb_entry_info", {}).get("resolution_combined")
    res = (res[0] if isinstance(res, list) and res else None)
    return {"release_date": rel, "method": methods[0] if methods else None,
            "resolution": res}


def entity_seq_uniprot(pdbid, entity="1"):
    """Return (sequence, uniprot_or_None) for a polymer entity."""
    d = _get_json(DATA_ENTITY.format(pdbid, entity), f"entity:{pdbid}:{entity}")
    if "__error__" in d:
        return None, None
    seq = d.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
    seq = "".join(seq.split())
    ups = d.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
    return (seq or None), (ups[0] if ups else None)


def search_entries(method, res_lo, res_hi, date_lo, date_hi, n, length=(50, 500)):
    """Single-protein-entity entries by method, resolution, release-date window."""
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "exact_match", "value": method}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
            "operator": "equals", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_DNA",
            "operator": "equals", "value": 0}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
            "operator": "equals", "value": 0}},
    ]
    if res_lo is not None:
        nodes.append({"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", "operator": "range",
            "value": {"from": res_lo, "to": res_hi,
                      "include_lower": True, "include_upper": False}}})
    nodes.append({"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_accession_info.initial_release_date", "operator": "range",
        "value": {"from": f"{date_lo}T00:00:00Z", "to": f"{date_hi}T00:00:00Z",
                  "include_lower": False, "include_upper": True}}})
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes},
         "return_type": "entry",
         "request_options": {"paginate": {"start": 0, "rows": n},
                             "sort": [{"sort_by": "rcsb_accession_info.initial_release_date",
                                       "direction": "desc"}]}}
    key = f"search:{method}:{res_lo}:{res_hi}:{date_lo}:{date_hi}:{n}"
    d = _post_json(SEARCH_URL, q, key)
    return [h["identifier"] for h in d.get("result_set", [])]


def best_precutoff_homolog(seq, cutoff):
    """
    Sequence-search `seq` vs PDB entities released <= cutoff.
    Returns (best_nonself_identity, self_in_training_bool, n_hits).
    """
    if not seq or len(seq) < 25:
        return 0.0, False, 0
    shash = hashlib.sha1(seq.encode()).hexdigest()[:12]
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "sequence", "parameters": {
                "evalue_cutoff": 1, "identity_cutoff": 0.0,
                "sequence_type": "protein", "value": seq}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "less_or_equal", "value": f"{cutoff}T00:00:00Z"}}]},
         "return_type": "polymer_entity",
         "request_options": {"scoring_strategy": "sequence",
                             "results_verbosity": "verbose",
                             "paginate": {"start": 0, "rows": 25}}}
    d = _post_json(SEARCH_URL, q, f"homol:{cutoff}:{shash}")
    if "__error__" in d:
        return None, None, None
    best_nonself = 0.0
    self_hit = False
    hits = d.get("result_set", [])
    for h in hits:
        ident = None
        for svc in h.get("services", []):
            if svc.get("service_type") != "sequence":
                continue
            for node in svc.get("nodes", []):
                for mc in node.get("match_context", []):
                    si = mc.get("sequence_identity")
                    if si is not None:
                        ident = max(ident or 0.0, si)
        if ident is None:
            continue
        if ident >= SELF_ID:
            self_hit = True
        else:
            best_nonself = max(best_nonself, ident)
    return best_nonself, self_hit, len(hits)


def precutoff_structural_homolog(pdbid, cutoff):
    """
    RCSB structure-similarity (shape) search of `pdbid` vs PDB structures released
    <= cutoff, EXCLUDING the query entry itself.  Catches fold-level leakage that
    sequence search misses (the residual-leakage problem from the literature).
    Returns (n_nonself_hits, best_nonself_score).  Only usable for targets that
    have a deposited structure (entry_id).
    """
    pdbid = pdbid.upper()
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "structure", "parameters": {
                "operator": "strict_shape_match",
                "value": {"entry_id": pdbid, "assembly_id": "1"}}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "less_or_equal", "value": f"{cutoff}T00:00:00Z"}}]},
         "return_type": "polymer_entity",
         "request_options": {"scoring_strategy": "structure",
                             "paginate": {"start": 0, "rows": 25}}}
    d = _post_json(SEARCH_URL, q, f"struct:{cutoff}:{pdbid}")
    if "__error__" in d:
        return None, None
    n_nonself = 0
    best = 0.0
    for h in d.get("result_set", []):
        ident = h.get("identifier", "")
        if ident.upper().startswith(pdbid):       # exclude self entry
            continue
        n_nonself += 1
        best = max(best, h.get("score", 0.0))
    return n_nonself, best


def uniprot_no_pdb(n, organism=9606):
    """Reviewed UniProt entries with NO PDB cross-ref (for no_expt targets)."""
    qs = (f"(organism_id:{organism}) AND (reviewed:true) AND NOT (database:pdb) "
          f"AND (length:[50 TO 500])")
    params = urllib.parse.urlencode({"query": qs, "size": min(n, 500),
                                     "fields": "accession,sequence", "format": "json"})
    url = f"https://rest.uniprot.org/uniprotkb/search?{params}"
    d = _get_json(url, f"uniprot_nopdb:{organism}:{n}")
    out = []
    for e in d.get("results", []):
        acc = e.get("primaryAccession")
        seq = e.get("sequence", {}).get("value")
        if acc and seq:
            out.append((acc, seq))
    return out


# ── tier assignment ───────────────────────────────────────────────────────────

def temporal_tier(date):
    if not date:
        return None
    if date <= AF3_CUT:
        return "seen_all"
    if date <= BOLTZ2_CUT:
        return "held_af3_seen_boltz2"
    return "held_all"


def homology_tier(best_id, struct_n=None):
    """Homology tier from sequence identity to pre-cutoff PDB, upgraded by
    STRUCTURAL homology: a sequence-orphan that still has a pre-cutoff structural
    (fold) relative is 'remote_homolog' (fold leakage), not a true orphan."""
    if best_id is None:
        return "unknown"
    if best_id >= CLOSE_ID:
        return "close_homolog"
    if best_id >= REMOTE_ID:
        return "remote_homolog"
    # sequence-orphan: check structural homology
    if struct_n and struct_n > 0:
        return "remote_homolog"          # fold present pre-cutoff (structural leakage)
    return "orphan"


def resolution_tag(res):
    if res is None:
        return "na"
    if res <= 2.0:
        return "high"
    if res <= 3.5:
        return "mid"
    return "low"


# ── main composition ──────────────────────────────────────────────────────────

def compose(args):
    targets = []          # list of manifest dicts
    seen_seqhash = {}      # dedup sequences -> first target_id

    def add_target(tid, uniprot, seq, has_expt, expt_pdb, meta):
        if not seq or len(seq) < 25:
            return False
        shash = hashlib.sha1(seq.encode()).hexdigest()[:12]
        best_id, self_hit, nh = best_precutoff_homolog(seq, AF3_CUT)
        time.sleep(args.sleep)
        # Structural homology (fold-level leakage) — only for targets with a
        # deposited structure to query by entry_id.
        struct_n, struct_score = (None, None)
        if has_expt and expt_pdb:
            struct_n, struct_score = precutoff_structural_homolog(expt_pdb, AF3_CUT)
            time.sleep(args.sleep)
        d = meta or {}
        rel = d.get("release_date")
        row = {
            "target_id": tid, "uniprot": uniprot or "", "seq_len": len(seq),
            "seq_sha1": shash,
            "has_expt": int(has_expt),
            "expt_pdb": expt_pdb or "", "expt_method": d.get("method") or "",
            "expt_resolution": d.get("resolution") if d.get("resolution") is not None else "",
            "expt_release_date": rel or "",
            "resolution_tag": resolution_tag(d.get("resolution")) if has_expt else "na",
            "temporal_tier": temporal_tier(rel) if has_expt else "no_self_structure",
            "self_in_training": int(bool(self_hit)),
            "best_homolog_id": round(best_id, 3) if best_id is not None else "",
            "struct_homolog_n": struct_n if struct_n is not None else "",
            "struct_homolog_score": round(struct_score, 3) if struct_score else "",
            "homology_tier": homology_tier(best_id, struct_n),
        }
        targets.append(row)
        seen_seqhash.setdefault(shash, (tid, seq))
        return True

    # ── has_expt: sample per temporal window × method ──────────────────────────
    # high-res preferred so we get clean ground truth; spread methods.
    windows = [
        ("seen_all", "1900-01-01", AF3_CUT),
        ("held_af3_seen_boltz2", AF3_CUT, BOLTZ2_CUT),
        ("held_all", BOLTZ2_CUT, "2099-01-01"),
    ]
    methods = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    per = max(args.n_expt, 1)
    for tier, dlo, dhi in windows:
        got = 0
        for method in methods:
            if got >= per:
                break
            # pull more candidates than needed (some fail seq/uniprot or dup)
            cand = search_entries(method, 0.0, args.res_max, dlo, dhi, per * 3)
            for pid in cand:
                if got >= per:
                    break
                meta = entry_meta(pid)
                if not meta or temporal_tier(meta["release_date"]) != tier:
                    continue
                seq, up = entity_seq_uniprot(pid, "1")
                if not seq:
                    continue
                shash = hashlib.sha1(seq.encode()).hexdigest()[:12]
                if shash in seen_seqhash:
                    continue
                if add_target(pid.lower(), up, seq, True, pid, meta):
                    got += 1
            cache_save()
        print(f"[predset] has_expt {tier}: {got} targets", flush=True)

    # ── no_expt: UniProt with no PDB ───────────────────────────────────────────
    if args.n_noexpt > 0:
        got = 0
        for acc, seq in uniprot_no_pdb(args.n_noexpt * 3):
            if got >= args.n_noexpt:
                break
            shash = hashlib.sha1(seq.encode()).hexdigest()[:12]
            if shash in seen_seqhash:
                continue
            if add_target(acc, acc, seq, False, None, None):
                got += 1
            cache_save()
        print(f"[predset] no_expt: {got} targets", flush=True)

    return targets, seen_seqhash


def write_outputs(targets, seqs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["target_id", "uniprot", "seq_len", "seq_sha1", "has_expt",
            "expt_pdb", "expt_method", "expt_resolution", "expt_release_date",
            "resolution_tag", "temporal_tier", "self_in_training",
            "best_homolog_id", "struct_homolog_n", "struct_homolog_score",
            "homology_tier"]
    man = os.path.join(out_dir, "prediction_manifest.tsv")
    with open(man, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in targets:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # FASTA: one entry per unique sequence (target_id of first occurrence)
    fa = os.path.join(out_dir, "prediction_targets.fasta")
    with open(fa, "w") as f:
        written = set()
        for r in targets:
            sh = r["seq_sha1"]
            if sh in written:
                continue
            tid, seq = seqs[sh]
            f.write(f">{r['target_id']}|{r['homology_tier']}|{r['temporal_tier']}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")
            written.add(sh)

    # cross-tab summary
    print("\n[predset] composition (temporal × homology):")
    ct = Counter((r["temporal_tier"], r["homology_tier"]) for r in targets)
    for k in sorted(ct):
        print(f"  {k[0]:24s} {k[1]:16s} {ct[k]}")
    print(f"[predset] {len(targets)} targets, "
          f"{len({r['seq_sha1'] for r in targets})} unique sequences")
    print(f"[predset] wrote {man}\n[predset] wrote {fa}")
    return man, fa


GEN_MD = """# Generating AF3 + Boltz-2 predictions for the structure-quality benchmark

Targets: `prediction_targets.fasta` (one record per unique sequence).
Tags per target: `prediction_manifest.tsv`.

## What to produce
For each FASTA record, a single best predicted model in **PDB** format:
  - AlphaFold3 → `af3/<target_id>.pdb`
  - Boltz-2    → `boltz2/<target_id>.pdb`
Keep the per-residue confidence in the B-factor column (pLDDT / pLDDT-equivalent),
exactly as the predictors emit it — the analysis reads it from there.
`<target_id>` is the first field of the FASTA header (before the first `|`).

## Important: templates / MSA settings (controls the leakage test)
The whole point is to compare predictions where the model had vs. had not seen the
fold. To keep the leakage axes meaningful:
  - Run **two conditions** where feasible: (a) default (MSA + templates on), and
    (b) **templates OFF** (MSA only). Save (b) under `af3_notmpl/` and
    `boltz2_notmpl/`. Condition (b) is the clean blind test for the
    `held_all` / `orphan` targets (no structure leaks in via templates).
  - Use the same MSA depth settings for all targets within a condition.

## Notes
  - Monomers only (targets are single-protein entities, 50-500 aa).
  - If a predictor outputs mmCIF, convert to PDB (e.g. `maxit -input x.cif
    -output x.pdb -o 2` or gemmi `gemmi convert x.cif x.pdb`).
  - Drop the finished folders into `HighResPDB/benchmark/` and they become
    drop-in categories for the feature builder + analysis.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="analysis/prediction_set")
    ap.add_argument("--n-expt", type=int, default=40,
                    help="targets per temporal tier WITH experimental ground truth")
    ap.add_argument("--n-noexpt", type=int, default=40,
                    help="targets with NO experimental structure")
    ap.add_argument("--res-max", type=float, default=2.5,
                    help="max resolution (A) for has_expt candidates")
    ap.add_argument("--sleep", type=float, default=0.15,
                    help="seconds between sequence searches (be nice to RCSB)")
    args = ap.parse_args()

    cache_load(os.path.join(args.out_dir, "_api_cache.json"))
    os.makedirs(args.out_dir, exist_ok=True)
    targets, seqs = compose(args)
    cache_save()
    man, fa = write_outputs(targets, seqs, args.out_dir)
    with open(os.path.join(args.out_dir, "GENERATE_PREDICTIONS.md"), "w") as f:
        f.write(GEN_MD)
    print(f"[predset] wrote {os.path.join(args.out_dir, 'GENERATE_PREDICTIONS.md')}")


if __name__ == "__main__":
    main()
