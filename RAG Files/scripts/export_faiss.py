#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
RAG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # RAG Files/
INDEX_DIR = os.path.join(RAG_DIR, "indexes")
FAISS_DIR = os.path.join(RAG_DIR, "indexes_faiss")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_project_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        return load_json(path) or {}
    except Exception:
        return {}


def read_index(index_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    emb_path = os.path.join(index_dir, "embeddings.npy")
    md_path = os.path.join(index_dir, "metadata.jsonl")
    manifest_path = os.path.join(index_dir, "index_manifest.json")
    if not os.path.exists(emb_path):
        raise SystemExit(f"Missing embeddings at: {emb_path}")
    if not os.path.exists(md_path):
        raise SystemExit(f"Missing metadata at: {md_path}")
    embs = np.load(emb_path)
    metadatas: List[Dict[str, Any]] = []
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                metadatas.append(json.loads(line))
            except Exception:
                continue
    manifest: Dict[str, Any] = {}
    if os.path.exists(manifest_path):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            pass
    return embs, metadatas, manifest


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_faiss_index(embs: np.ndarray, metric: str = "ip", normalize: bool = True):
    try:
        import faiss  # type: ignore
    except Exception as ex:
        raise SystemExit("faiss is not installed. Install with: pip install faiss-cpu")

    x = embs.astype("float32")
    if normalize:
        # Avoid div by zero
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x = x / norms

    d = x.shape[1]
    if metric.lower() == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(x)
    return index


def write_faiss(index, out_file: str) -> None:
    import faiss  # type: ignore
    faiss.write_index(index, out_file)


def main():
    ap = argparse.ArgumentParser(description="Export existing numpy index to FAISS")
    default_config = os.path.join(RAG_DIR, "project.config.json")
    ap.add_argument("--config", default=default_config, help="Path to project config JSON (optional)")
    ap.add_argument("--index-dir", default=INDEX_DIR, help="Directory containing embeddings.npy and metadata.jsonl")
    ap.add_argument("--out", default=os.path.join(FAISS_DIR, "index.faiss"), help="Output FAISS index file path")
    ap.add_argument("--metric", choices=["ip", "l2"], default="ip", help="Similarity metric for FAISS index")
    ap.add_argument("--normalize", action="store_true", help="Normalize vectors (recommended for cosine/IP)")
    args = ap.parse_args()

    cfg = load_project_config(args.config)
    eff_index_dir = args.index_dir if args.index_dir != INDEX_DIR else (cfg.get("default_index_dir") or args.index_dir)

    embs, metadatas, manifest = read_index(eff_index_dir)

    ensure_dir(os.path.dirname(args.out))

    index = create_faiss_index(embs, metric=args.metric, normalize=bool(args.normalize))
    write_faiss(index, args.out)

    # Write sidecar metadata for convenience
    sidecar_manifest = {
        "source_index_dir": eff_index_dir,
        "rows": int(embs.shape[0]),
        "dim": int(embs.shape[1]),
        "metric": args.metric,
        "normalized": bool(args.normalize),
    }
    with open(os.path.join(os.path.dirname(args.out), "faiss_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(sidecar_manifest, f, ensure_ascii=False, indent=2)

    # Copy metadata.jsonl for lookup
    try:
        md_src = os.path.join(eff_index_dir, "metadata.jsonl")
        md_dst = os.path.join(os.path.dirname(args.out), "metadata.jsonl")
        with open(md_src, "r", encoding="utf-8") as fi, open(md_dst, "w", encoding="utf-8") as fo:
            for line in fi:
                fo.write(line)
    except Exception:
        pass

    print(f"Wrote FAISS index: {args.out}")


if __name__ == "__main__":
    main()
