#!/usr/bin/env python3
"""Semantic code index for the Universe repository.

Why this exists: the repo is ~55k lines of Rust plus WGSL shaders, and the most
valuable knowledge lives in doc-comments that record *measured* findings ("the
tank was sized per lap but refilled per sector"). Grep finds those only if you
already know the words. This index lets a human or an agent ask a question in
plain language and get the exact `file:line` back.

Design constraints, in order:
  * offline — uses the all-MiniLM-L6-v2 ONNX model already cached on disk by
    chroma; no network access, no API key;
  * cheap — the whole index is a single .npz plus a .jsonl, a few megabytes;
  * rebuildable — any contributor runs `--build` and gets the same artifact, so
    the index is never committed (see .gitignore).

Usage
-----
    python tools/semantic_index.py --build            # index the repository
    python tools/semantic_index.py "how is fuel sized"
    python tools/semantic_index.py --symbol tanque    # exact-name fallback
    python tools/semantic_index.py --stats

Chunking is item-level, not line-level: a chunk is one Rust/WGSL top-level item
(fn, struct, impl, mod) together with the doc-comment above it, or one Markdown
section. That keeps a returned hit self-contained enough to act on.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO / ".semantic-index"
VECTORS = INDEX_DIR / "vectors.npz"
CHUNKS = INDEX_DIR / "chunks.jsonl"
MODEL_DIR = Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx"

# Directories that hold build output, third-party code or generated data. Indexing
# them would bury real hits under noise.
# `docs/arquivo` is the archive of superseded reports: indexing it buries live
# code under documents that were already replaced. Same for vendored specs.
SKIP_DIRS = {"target", ".git", "node_modules", ".semantic-index", "dna_backup",
             "arquivo", "superpowers"}
EXTENSIONS = {".rs", ".wgsl", ".md", ".toml", ".sh", ".py"}
MAX_CHARS = 2000  # a chunk longer than this is truncated before embedding


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

# A Rust/WGSL item starts at column 0 (or after `pub`) — nested items keep their
# parent's chunk, which is what we want: an `impl` block reads as one unit.
ITEM_RE = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
    r"(fn|struct|enum|trait|impl|mod|const|static|type)\s+([A-Za-z_][A-Za-z0-9_]*)",
)
WGSL_RE = re.compile(r"^(?:@[a-z_]+(?:\([^)]*\))?\s*)*(fn|struct|var|const)\s+([A-Za-z_][A-Za-z0-9_<>]*)")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)")


def chunk_source(path: Path, text: str) -> list[dict]:
    """Split Rust/WGSL into item-level chunks, carrying the doc-comment along."""
    lines = text.splitlines()
    pattern = WGSL_RE if path.suffix == ".wgsl" else ITEM_RE
    starts: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if not m:
            continue
        # Walk backwards over the attached doc-comment / attributes so the chunk
        # begins at the explanation, not at the signature.
        j = i
        while j > 0 and re.match(r"^\s*(///|//!|//|#\[)", lines[j - 1]):
            j -= 1
        starts.append((j, m.group(1), m.group(2)))
    if not starts:
        return [_chunk(path, 1, len(lines), "file", path.name, text)]

    chunks = []
    # Everything before the first item is the module header (//! doc-comment).
    if starts[0][0] > 0:
        chunks.append(_chunk(path, 1, starts[0][0], "header", path.name,
                             "\n".join(lines[: starts[0][0]])))
    for k, (start, kind, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        chunks.append(_chunk(path, start + 1, end, kind, name,
                             "\n".join(lines[start:end])))
    return chunks


def chunk_markdown(path: Path, text: str) -> list[dict]:
    """Split Markdown by heading; a section is the unit a reader actually wants."""
    lines = text.splitlines()
    starts = [(i, m.group(2)) for i, line in enumerate(lines)
              if (m := HEADING_RE.match(line))]
    if not starts:
        return [_chunk(path, 1, len(lines), "doc", path.name, text)]
    chunks = []
    if starts[0][0] > 0:
        chunks.append(_chunk(path, 1, starts[0][0], "doc", path.name,
                             "\n".join(lines[: starts[0][0]])))
    for k, (start, title) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        chunks.append(_chunk(path, start + 1, end, "section", title,
                             "\n".join(lines[start:end])))
    return chunks


def _chunk(path: Path, line_start: int, line_end: int, kind: str, name: str, body: str) -> dict:
    rel = str(path.relative_to(REPO))
    return {
        "file": rel,
        "line": line_start,
        "line_end": line_end,
        "kind": kind,
        "name": name,
        # The embedded text leads with the location so that the model sees the
        # module path as context, which measurably helps short items.
        "text": f"{rel} :: {kind} {name}\n{body}"[:MAX_CHARS],
        "preview": body.strip().splitlines()[0][:120] if body.strip() else "",
    }


def collect_chunks() -> list[dict]:
    chunks: list[dict] = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            path = Path(root) / fname
            if path.suffix not in EXTENSIONS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue
            chunks += (chunk_markdown(path, text) if path.suffix == ".md"
                       else chunk_source(path, text))
    return [c for c in chunks if c["text"].strip()]


# --------------------------------------------------------------------------- #
# Embedding (ONNX MiniLM, mean-pooled and L2-normalised)
# --------------------------------------------------------------------------- #

def load_model():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    if not MODEL_DIR.exists():
        sys.exit(f"embedding model not found at {MODEL_DIR}.\n"
                 "It ships with the chroma MCP server; install that or point "
                 "MODEL_DIR at any all-MiniLM-L6-v2 ONNX export.")
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=256)
    tok.enable_padding(length=256)
    sess = ort.InferenceSession(str(MODEL_DIR / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    return tok, sess


def embed(texts: list[str], tok, sess, batch: int = 64) -> np.ndarray:
    out = []
    names = {i.name for i in sess.get_inputs()}
    for i in range(0, len(texts), batch):
        enc = tok.encode_batch(texts[i:i + batch])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(None, feed)[0]                       # [B, T, H]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        out.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9))
        print(f"\r  embedding {min(i + batch, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    return np.vstack(out).astype(np.float32)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_build() -> None:
    chunks = collect_chunks()
    print(f"{len(chunks)} chunks from {len({c['file'] for c in chunks})} files")
    tok, sess = load_model()
    vecs = embed([c["text"] for c in chunks], tok, sess)
    INDEX_DIR.mkdir(exist_ok=True)
    np.savez_compressed(VECTORS, vectors=vecs)
    with CHUNKS.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({k: v for k, v in c.items() if k != "text"}) + "\n")
    print(f"index written to {INDEX_DIR} "
          f"({VECTORS.stat().st_size / 1e6:.1f} MB vectors)")


def load_index():
    if not VECTORS.exists():
        sys.exit("no index yet — run: python tools/semantic_index.py --build")
    vecs = np.load(VECTORS)["vectors"]
    meta = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines()]
    return vecs, meta


def lexical_scores(question: str, meta: list[dict]) -> np.ndarray:
    """Token-overlap score against file path and item name.

    The embedding model is English-centric while much of this repository is still
    being migrated from Portuguese, so pure vector search misses hits whose only
    English token is the identifier. Overlap on the *name* and *path* is immune to
    that, and it is also what rescues rare identifiers, which embeddings handle
    poorly. The two signals are summed, not multiplied: either one alone should be
    able to surface a chunk.
    """
    terms = [t for t in re.split(r"[^A-Za-z0-9_]+", question.lower()) if len(t) > 2]
    if not terms:
        return np.zeros(len(meta), dtype=np.float32)
    out = np.zeros(len(meta), dtype=np.float32)
    for i, m in enumerate(meta):
        hay = f"{m['file']} {m['name']} {m.get('preview', '')}".lower()
        out[i] = sum(1.0 for t in terms if t in hay) / len(terms)
    return out


def cmd_query(question: str, k: int, kind: str | None) -> None:
    vecs, meta = load_index()
    tok, sess = load_model()
    q = embed([question], tok, sess)[0]
    # Hybrid: cosine similarity plus lexical overlap (see lexical_scores).
    scores = vecs @ q + 0.5 * lexical_scores(question, meta)
    order = np.argsort(-scores)
    shown = 0
    for i in order:
        m = meta[i]
        if kind and m["kind"] != kind:
            continue
        print(f"{scores[i]:.3f}  {m['file']}:{m['line']}  [{m['kind']}] {m['name']}")
        if m["preview"]:
            print(f"        {m['preview']}")
        shown += 1
        if shown >= k:
            break


def cmd_symbol(name: str, k: int) -> None:
    """Exact-name fallback: semantic search is bad at rare identifiers."""
    _, meta = load_index()
    hits = [m for m in meta if name.lower() in m["name"].lower()]
    for m in hits[:k]:
        print(f"{m['file']}:{m['line']}  [{m['kind']}] {m['name']}")
    if not hits:
        print(f"no symbol matching {name!r}; try a plain-language query instead")


def cmd_stats() -> None:
    _, meta = load_index()
    by_kind: dict[str, int] = {}
    for m in meta:
        by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1
    print(f"{len(meta)} chunks · {len({m['file'] for m in meta})} files")
    for kind, n in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {kind:>8}: {n}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("question", nargs="*", help="plain-language question")
    p.add_argument("--build", action="store_true", help="(re)build the index")
    p.add_argument("--symbol", help="exact identifier lookup")
    p.add_argument("--stats", action="store_true")
    p.add_argument("-k", type=int, default=8, help="number of hits (default 8)")
    p.add_argument("--kind", help="filter by chunk kind (fn, struct, section, ...)")
    a = p.parse_args()
    if a.build:
        cmd_build()
    elif a.stats:
        cmd_stats()
    elif a.symbol:
        cmd_symbol(a.symbol, a.k)
    elif a.question:
        cmd_query(" ".join(a.question), a.k, a.kind)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
