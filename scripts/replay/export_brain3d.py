#!/usr/bin/env python3
"""Export the complete FlyWire neuron index, coordinates, edges and trace for WebGL2."""
import argparse, base64, csv, json, struct, hashlib
from pathlib import Path
import numpy as np

def b64(a):
    return base64.b64encode(np.asarray(a).tobytes(order="C")).decode("ascii")

def load_annotations(path):
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            soma = [row.get("soma_" + axis) for axis in ("x", "y", "z")]
            coord = soma if all(v not in (None, "") for v in soma) else [row.get("pos_" + axis) for axis in ("x", "y", "z")]
            if any(value in (None, "") for value in coord):
                continue
            out[int(row["root_id"])] = {"x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2]), "type": row.get("cell_type", ""), "flow": row.get("flow", "")}
    return out

def read_map(path):
    # Canonical FLYMAP02 body: header u32[8], then CSR arrays in fly_map.rs order.
    with path.open("rb") as f:
        magic = f.read(8); head = struct.unpack("<8I", f.read(32)); n, m = head[:2]
        indptr = np.fromfile(f, dtype="<u4", count=n + 1); indices = np.fromfile(f, dtype="<u4", count=m)
        weights = np.fromfile(f, dtype="<f4", count=m)
        typ = np.fromfile(f, dtype="<u4", count=n); ing = np.fromfile(f, dtype="<u4", count=n)
        out_ind = np.fromfile(f, dtype="<u4", count=head[4] + 1); out_mem = np.fromfile(f, dtype="<u4", count=head[5])
        pre_ind = np.fromfile(f, dtype="<u4", count=n + 1) if head[6] & 1 else None
        pre_idx = np.fromfile(f, dtype="<u4", count=m) if pre_ind is not None else None
        pre_w = np.fromfile(f, dtype="<f4", count=m) if pre_ind is not None else None
    if magic not in (b"FLYMAP01", b"FLYMAP02") or len(indptr) != n + 1 or len(indices) != m:
        raise ValueError("invalid canonical flymap header/CSR")
    return indptr, indices, weights, pre_ind, pre_idx, pre_w

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--map", type=Path, required=True); ap.add_argument("--annotations", type=Path, required=True); ap.add_argument("--trace", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    indptr, map_indices, map_weights, pre_ind, pre_idx, pre_weights = read_map(a.map)
    z = np.load(a.out.parent / "edges_783.npz") if (a.out.parent / "edges_783.npz").exists() else np.load(a.map.parent / "cache" / "edges_783.npz")
    roots = z["root_ids"].astype("<u8", copy=False); pre = z["pre"].astype("<u4", copy=False); post = z["post"].astype("<u4", copy=False)
    if len(roots) != len(indptr) - 1 or len(pre) != len(map_indices):
        raise ValueError("cache/map neuron or edge counts do not match")
    # The cache preserves source-table order, while the canonical file stores a
    # postsynaptic CSR plus an independent presynaptic CSR. Export the canonical
    # endpoint arrays; comparing raw cache arrays would report a false ordering
    # mismatch even when the edge relation is identical.
    if pre_ind is None or pre_idx is None:
        raise ValueError("selected map lacks canonical presynaptic CSR")
    pre = np.repeat(np.arange(len(pre_ind) - 1, dtype="<u4"), np.diff(pre_ind).astype("<i8"))
    post = pre_idx.astype("<u4", copy=False)
    if not np.array_equal(pre, z["pre"]) or not np.array_equal(post, z["post"]) or not np.array_equal(pre_weights, z["weight"]):
        raise ValueError("map connectivity does not match the root-index cache")
    if len(np.unique(roots)) != len(roots) or not np.all(roots[1:] > roots[:-1]):
        raise ValueError("root IDs must follow the canonical sorted unique indexing")
    ann = load_annotations(a.annotations); missing = [int(r) for r in roots if int(r) not in ann]
    xyz = np.asarray([[ann[int(r)][k] for k in ("x", "y", "z")] if int(r) in ann else [np.nan, np.nan, np.nan] for r in roots], dtype="<f4")
    xyz *= np.asarray([.004, .004, .04], dtype="<f4")
    meta = [{"rootId": str(int(r)), "type": ann.get(int(r), {}).get("type", "unannotated"), "flow": ann.get(int(r), {}).get("flow", "unannotated")} for r in roots]
    trace = json.loads(a.trace.read_text())
    frames = trace.get("trace", trace).get("frames", [])
    payload = {"n": int(len(roots)), "m": int(len(pre)), "units": "micrometres", "coordinateSource": "https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.search_annotations.html", "voxelNanometres": [4,4,40], "missingCoordinates": len(missing), "tracePath": trace.get("tracePath"), "traceBeing": trace.get("trace", trace).get("being"), "traceStepsPerTick": trace.get("traceStepsPerTick"), "mapSha256": hashlib.sha256(a.map.read_bytes()).hexdigest(), "rootIndexSha256": hashlib.sha256(roots.tobytes()).hexdigest(), "coordinateRepresentation": "soma when complete; otherwise annotated position; no neuronal arbors", "rootIds": [str(int(x)) for x in roots], "meta": meta, "xyz": b64(xyz), "edgePre": b64(pre), "edgePost": b64(post), "frames": [{"step": f["step"], "v": f["v"], "spikes": f["spikes"], "commands": f["commands"]} for f in frames]}
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "brain3d.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {a.out} ({len(roots)} neurons, {len(pre)} indexed edges, {len(frames)} frames, {len(missing)} unannotated)")
if __name__ == "__main__": main()
