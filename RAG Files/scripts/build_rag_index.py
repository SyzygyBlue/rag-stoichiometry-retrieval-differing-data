#!/usr/bin/env python3
"""
Build a simple local RAG index from scene JSON files (and optional knowledge JSONL).

Outputs to RAG Files/indexes/:
- embeddings.npy: float32 matrix [N, D]
- metadata.jsonl: one JSON per line describing each embedded chunk
- index_manifest.json: summary info (model, dimension, counts, timestamps)
- scene_entity_summary.json: scene_id -> [entity_ids]

This script:
1) Loads alias resolution rules from mappings/entity_aliases.json to attach canonical entity facets per scene
2) Loads entity/character catalogs to provide entity names/types
3) Reads scene JSONs at RAG Files/*.json
   - Prefers scene["embedding_payloads"] if present; else falls back to text_windows fields
4) Optionally ingests Q/A knowledge from knowledge.jsonl if configured
5) Embeds all chunks with a Sentence-Transformers model

Notes:
- No scene files are modified. We only read from them and write index artifacts.
- If you need to iterate quickly, use --limit to embed the first K chunks only.
"""

import os
import re
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit("sentence-transformers is required. Install with: pip install sentence-transformers numpy")

# -----------------
# Paths
# -----------------
SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # RAG Files/
RAG_DIR = BASE_DIR

ENTITIES_PATH = os.path.join(RAG_DIR, "entities", "entities.json")
CHAR_DIR = os.path.join(RAG_DIR, "character")
MAPPINGS_DIR = os.path.join(RAG_DIR, "mappings")
ALIAS_MAP_PATH = os.path.join(MAPPINGS_DIR, "entity_aliases.json")
INDEX_DIR = os.path.join(RAG_DIR, "indexes")

SCENE_GLOB = os.path.join(RAG_DIR, "*.json")

# -----------------
# Catalog helpers
# -----------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_project_config(path: str) -> Dict[str, Any]:
    """Load optional project config JSON. Returns {} if missing/invalid."""
    if not path or not os.path.exists(path):
        return {}
    try:
        return load_json(path) or {}
    except Exception:
        return {}


def build_entity_catalog(entities_path: str, char_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (name_by_id, type_by_id) from entities.json and character/*.json."""
    name_by_id: Dict[str, str] = {}
    type_by_id: Dict[str, str] = {}

    # entities.json can be wrapped or flat
    try:
        root_obj = load_json(entities_path)
    except Exception:
        root_obj = {}
    root = root_obj.get("entities", root_obj) if isinstance(root_obj, dict) else {}

    type_keys = [
        ("items", "item"),
        ("technologies", "technology"),
        ("locations", "location"),
        ("organizations", "organization"),
        ("factions", "faction"),
    ]

    for key, tname in type_keys:
        for obj in root.get(key, []) or []:
            eid = (obj or {}).get("id") or (obj or {}).get("name")
            if not eid:
                continue
            name_by_id[eid] = (obj or {}).get("name", eid)
            type_by_id[eid] = tname

    # characters
    for cpath in glob.glob(os.path.join(char_dir, "*.json")):
        try:
            c = load_json(cpath)
        except Exception:
            continue
        ident = (c or {}).get("identity", {})
        cid = ident.get("id") or os.path.splitext(os.path.basename(cpath))[0]
        display = ident.get("display_name") or cid
        name_by_id[cid] = display
        type_by_id[cid] = "character"

    return name_by_id, type_by_id


def load_scene_to_entities(alias_map_path: str) -> Dict[str, List[str]]:
    """Build mapping scene_id -> list of resolved entity_ids from alias rules."""
    out: Dict[str, List[str]] = {}
    if not os.path.exists(alias_map_path):
        return out
    try:
        data = load_json(alias_map_path)
    except Exception:
        return out

    rules = (data or {}).get("rules", []) or []
    tmp: Dict[str, set] = {}
    for r in rules:
        ctx = (r or {}).get("context", {})
        scenes = ctx.get("scenes") or []
        eids = (r or {}).get("entity_ids") or []
        for sid in scenes:
            s = tmp.setdefault(sid, set())
            for eid in eids:
                if isinstance(eid, str) and eid:
                    s.add(eid)
    # finalize lists sorted for stability
    for sid, s in tmp.items():
        out[sid] = sorted(s)
    return out


def load_manual_overrides(path: str) -> Dict[str, List[str]]:
    """
    Load manual overrides mapping scene_id -> [entity_id, ...].
    Accepts either a plain mapping or an object with key "overrides".
    """
    out: Dict[str, List[str]] = {}
    if not path or not os.path.exists(path):
        return out
    try:
        obj = load_json(path)
    except Exception:
        return out
    mapping = obj.get("overrides") if isinstance(obj, dict) else None
    if isinstance(mapping, dict):
        items = mapping.items()
    elif isinstance(obj, dict):
        items = obj.items()
    else:
        items = []
    for sid, arr in items:
        if not isinstance(sid, str):
            continue
        if isinstance(arr, list):
            vals = [e for e in arr if isinstance(e, str) and e]
            if vals:
                out[sid] = sorted(set(vals))
    return out


# -----------------
# Scene and chunk extraction
# -----------------

def collect_scene_files(rag_dir: str, scene_globs: List[str] = None) -> List[str]:
    """Return scene file paths. If scene_globs provided, use them; else default SCENE_GLOB."""
    files: List[str] = []
    patterns = scene_globs or [SCENE_GLOB]
    root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # project root
    for pat in patterns:
        if not isinstance(pat, str) or not pat.strip():
            continue
        if not os.path.isabs(pat):
            # make relative patterns absolute from project root
            ap = os.path.abspath(os.path.join(root, pat))
        else:
            ap = pat
        for p in glob.glob(ap):
            if os.path.isfile(p):
                files.append(p)
    if not files:
        # fallback to default
        files = [p for p in glob.glob(SCENE_GLOB) if os.path.isfile(p)]
    return sorted(set(files))


def extract_chunks(
    scene: Dict[str, Any],
    *,
    rechunk: bool = False,
    chunking: str = "payload",
    target_tokens: int = 120,
    overlap_tokens: int = 30,
) -> List[Dict[str, Any]]:
    """Return a list of chunk dicts with keys: chunk_id (optional), type, text.

    If rechunk=True, build chunks from long text (full_scene_clean preferred) using
    simple paragraph/sentence splitting and word-count-based windowing with overlap.
    """
    chunks: List[Dict[str, Any]] = []
    
    def word_count(s: str) -> int:
        return len(s.split())

    def chunk_windows(pieces: List[str], tgt: int, ovl: int) -> List[str]:
        out: List[str] = []
        buf: List[str] = []
        cur = 0
        for piece in pieces:
            wc = word_count(piece)
            if cur + wc <= max(1, tgt):
                buf.append(piece)
                cur += wc
            else:
                if buf:
                    out.append(" ".join(buf).strip())
                # start new window with overlap from previous buffer
                if ovl > 0 and out:
                    prev = out[-1].split()
                    back = " ".join(prev[-ovl:])
                    buf = [back, piece]
                    cur = len(back.split()) + wc
                else:
                    buf = [piece]
                    cur = wc
        if buf:
            out.append(" ".join(buf).strip())
        return [x for x in out if x]

    def split_paragraphs(text: str) -> List[str]:
        # split on blank lines or double newlines
        parts = re.split(r"\n\s*\n+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def split_sentences(text: str) -> List[str]:
        # naive sentence splitter on punctuation followed by space/case change
        # keeps punctuation
        parts = re.split(r"(?<=[\.!?])\s+(?=[A-Z0-9\[\(\"'])", text.strip())
        return [p.strip() for p in parts if p.strip()]

    if rechunk:
        tw = (scene or {}).get("text_windows") or {}
        full = tw.get("full_scene_clean")
        recap = tw.get("dense_recap_250w")
        base = None
        if isinstance(full, str) and full.strip():
            base = full.strip()
        elif isinstance(recap, str) and recap.strip():
            base = recap.strip()
        else:
            # fallback: concatenate payloads
            eps = (scene or {}).get("embedding_payloads") or []
            texts = [((it or {}).get("text") or "").strip() for it in eps]
            texts = [t for t in texts if t]
            base = "\n\n".join(texts)
        if not base:
            return []
        if chunking == "sentence":
            pieces = split_sentences(base)
            windows = chunk_windows(pieces, target_tokens, overlap_tokens)
            for i, t in enumerate(windows, start=1):
                chunks.append({"chunk_id": f"sent-{i:04d}", "type": "rechunk-sentence", "text": t})
        elif chunking == "paragraph":
            pieces = split_paragraphs(base)
            windows = chunk_windows(pieces, target_tokens, overlap_tokens)
            for i, t in enumerate(windows, start=1):
                chunks.append({"chunk_id": f"para-{i:04d}", "type": "rechunk-paragraph", "text": t})
        else:
            # default to payload behavior if unknown mode
            chunking = "payload"

    if not rechunk or chunking == "payload":
        # Prefer explicit embedding_payloads
        eps = (scene or {}).get("embedding_payloads") or []
        for it in eps:
            text = (it or {}).get("text")
            if isinstance(text, str) and text.strip():
                chunks.append({
                    "chunk_id": (it or {}).get("chunk_id"),
                    "type": (it or {}).get("type") or "payload",
                    "text": text.strip(),
                    "meta": (it or {}).get("meta") or (it or {}).get("metadata"),
                })
        if chunks:
            return chunks

        # Fallbacks
        tw = (scene or {}).get("text_windows") or {}
        recap = tw.get("dense_recap_250w")
        if isinstance(recap, str) and recap.strip():
            chunks.append({"type": "recap", "text": recap.strip()})
        full = tw.get("full_scene_clean")
        if isinstance(full, str) and full.strip():
            chunks.append({"type": "scene", "text": full.strip()})
    return chunks


# -----------------
# Index build
# -----------------

def build_index(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    limit: int = 0,
    manual_overrides_path: str = "",
    knowledge_jsonl: str = "",
    rechunk: bool = False,
    chunking: str = "payload",
    target_tokens: int = 120,
    overlap_tokens: int = 30,
    out_dir: str = "",
    scene_globs: List[str] = None,
) -> None:
    out_dir = out_dir or INDEX_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Catalogs
    name_by_id, type_by_id = build_entity_catalog(ENTITIES_PATH, CHAR_DIR)
    scene_to_entities = load_scene_to_entities(ALIAS_MAP_PATH)
    manual = load_manual_overrides(manual_overrides_path)
    if manual:
        # merge
        tmp: Dict[str, set] = {k: set(v) for k, v in scene_to_entities.items()}
        for sid, add in manual.items():
            s = tmp.setdefault(sid, set())
            for eid in add:
                if isinstance(eid, str) and eid:
                    s.add(eid)
        scene_to_entities = {k: sorted(v) for k, v in tmp.items()}

    # Scenes
    scene_files = sorted(collect_scene_files(RAG_DIR, scene_globs=scene_globs))

    # Collect all texts and metadata
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for spath in scene_files:
        try:
            scene = load_json(spath)
        except Exception:
            continue
        scene_id = os.path.splitext(os.path.basename(spath))[0]

        # Entities for this scene
        eids = scene_to_entities.get(scene_id, [])
        entities_meta = [{
            "id": eid,
            "type": type_by_id.get(eid, "unknown"),
            "name": name_by_id.get(eid, eid),
        } for eid in eids]

        # Basic source metadata
        src = (scene or {}).get("source", {})
        book_title = src.get("book_title")
        chapter_title = src.get("chapter_title")
        date_in_universe = src.get("date_in_universe")
        doc_type = (scene or {}).get("doc_type")

        # Chunks
        chs = extract_chunks(
            scene,
            rechunk=rechunk,
            chunking=chunking,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        if not chs:
            continue
        for idx, ch in enumerate(chs):
            text = ch.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            chunk_id = ch.get("chunk_id") or f"{scene_id}-{ch.get('type','chunk')}-{idx+1}"
            md = {
                "row": len(metadatas),
                "chunk_uid": f"{scene_id}::{chunk_id}",
                "scene_id": scene_id,
                "chunk_type": ch.get("type", "chunk"),
                "source": {
                    "book_title": book_title,
                    "chapter_title": chapter_title,
                    "date_in_universe": date_in_universe,
                    "doc_type": doc_type,
                },
                "entities": entities_meta,
                "file": os.path.basename(spath),
            }
            extra = ch.get("meta")
            if isinstance(extra, dict) and extra:
                md["extra"] = extra
            texts.append(text)
            metadatas.append(md)
            if limit and len(texts) >= limit:
                break
        if limit and len(texts) >= limit:
            break

    # Optional: ingest knowledge JSONL as additional chunks
    if knowledge_jsonl and os.path.exists(knowledge_jsonl) and (not limit or len(texts) < limit):
        try:
            with open(knowledge_jsonl, "r", encoding="utf-8") as f:
                k_idx = 0
                for line in f:
                    if limit and len(texts) >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    instr = (obj or {}).get("instruction")
                    outp = (obj or {}).get("output")
                    meta = (obj or {}).get("metadata") or {}
                    # Build a compact text for embedding
                    parts: List[str] = []
                    if isinstance(instr, str) and instr.strip():
                        parts.append(f"Q: {instr.strip()}")
                    if isinstance(outp, str) and outp.strip():
                        parts.append(f"A: {outp.strip()}")
                    text_k = "\n".join(parts).strip()
                    if not text_k:
                        continue
                    k_idx += 1
                    chunk_id = f"kqa-{k_idx:05d}"
                    md = {
                        "row": len(metadatas),
                        "chunk_uid": f"knowledge::{chunk_id}",
                        "scene_id": "knowledge",
                        "chunk_type": "knowledge_qa",
                        "source": {
                            "book_title": None,
                            "chapter_title": meta.get("source_chapter"),
                            "date_in_universe": None,
                            "doc_type": "knowledge_qa",
                        },
                        "entities": [],
                        "file": os.path.basename(knowledge_jsonl),
                        "extra": {
                            "instruction": instr,
                            "output": outp,
                            "knowledge_meta": meta,
                            "text": text_k,
                        },
                    }
                    texts.append(text_k)
                    metadatas.append(md)
        except Exception:
            pass

    if not texts:
        raise SystemExit("No texts found to embed. Ensure your scene JSONs have embedding_payloads or text_windows, or provide knowledge.jsonl.")

    # Embed
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embs = embs.astype(np.float32, copy=False)

    # Write outputs
    emb_path = os.path.join(out_dir, "embeddings.npy")
    md_path = os.path.join(out_dir, "metadata.jsonl")
    manifest_path = os.path.join(out_dir, "index_manifest.json")
    scene_entity_path = os.path.join(out_dir, "scene_entity_summary.json")

    np.save(emb_path, embs)

    with open(md_path, "w", encoding="utf-8") as f:
        for md in metadatas:
            f.write(json.dumps(md, ensure_ascii=False) + "\n")

    # Persist scene -> entities for inspection
    with open(scene_entity_path, "w", encoding="utf-8") as f:
        # rehydrate names/types for readability
        scene_summary = {}
        for md in metadatas:
            sid = md["scene_id"]
            if sid not in scene_summary:
                scene_summary[sid] = md["entities"]
        json.dump(scene_summary, f, ensure_ascii=False, indent=2)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "model": model_name,
        "dimension": dim,
        "num_vectors": int(embs.shape[0]),
        "files": {
            "embeddings": os.path.relpath(emb_path, RAG_DIR),
            "metadata": os.path.relpath(md_path, RAG_DIR),
            "scene_entity_summary": os.path.relpath(scene_entity_path, RAG_DIR),
        }
    }
    # annotate knowledge path if used
    if knowledge_jsonl and os.path.exists(knowledge_jsonl):
        try:
            manifest.setdefault("files", {})["knowledge_jsonl"] = os.path.relpath(knowledge_jsonl, RAG_DIR)
        except Exception:
            manifest.setdefault("files", {})["knowledge_jsonl"] = knowledge_jsonl
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote embeddings: {emb_path} -> {embs.shape}")
    print(f"Wrote metadata:   {md_path} ({len(metadatas)} rows)")
    print(f"Wrote manifest:   {manifest_path}")
    print(f"Wrote scene->entities summary: {scene_entity_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build RAG index from scenes")
    # Optional project config; CLI overrides config values
    default_config = os.path.join(RAG_DIR, "project.config.json")
    p.add_argument("--config", default=default_config, help="Path to project config JSON (CLI overrides config)")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    p.add_argument("--limit", type=int, default=0, help="Embed only the first K chunks (for quick iteration)")
    p.add_argument("--manual-overrides", default=os.path.join(MAPPINGS_DIR, "manual_alias_overrides.json"), help="Path to manual scene->entity overrides JSON")
    p.add_argument("--knowledge-jsonl", default="", help="Path to knowledge JSONL (optional)")
    p.add_argument("--rechunk", action="store_true", help="Rechunk long text instead of using payloads")
    p.add_argument("--chunking", choices=["payload", "sentence", "paragraph"], default="payload", help="Chunking mode when --rechunk is used")
    p.add_argument("--target-tokens", type=int, default=120, help="Approx word target per chunk when rechunking")
    p.add_argument("--overlap-tokens", type=int, default=30, help="Approx word overlap between chunks when rechunking")
    p.add_argument("--out-dir", default=INDEX_DIR, help="Directory to write index artifacts to")
    p.add_argument("--scene-glob", action="append", help="Additional scene file glob(s). Can be repeated.")
    args = p.parse_args()
    cfg = load_project_config(args.config)

    # Merge precedence: CLI > config > defaults (approximate by checking parser defaults)
    parser_defaults = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "limit": 0,
        "manual_overrides": os.path.join(MAPPINGS_DIR, "manual_alias_overrides.json"),
        "knowledge_jsonl": "",
        "rechunk": False,
        "chunking": "payload",
        "target_tokens": 120,
        "overlap_tokens": 30,
        "out_dir": INDEX_DIR,
    }

    eff_model = args.model if args.model != parser_defaults["model"] else (cfg.get("model") or args.model)
    eff_limit = args.limit if args.limit != parser_defaults["limit"] else int(cfg.get("limit", args.limit))
    eff_overrides = args.manual_overrides if args.manual_overrides != parser_defaults["manual_overrides"] else (cfg.get("manual_overrides") or args.manual_overrides)
    eff_knowledge = args.knowledge_jsonl if args.knowledge_jsonl != parser_defaults["knowledge_jsonl"] else (cfg.get("knowledge_jsonl") or args.knowledge_jsonl)
    eff_rechunk = args.rechunk if args.rechunk != parser_defaults["rechunk"] else bool(cfg.get("rechunk", args.rechunk))
    eff_chunking = args.chunking if args.chunking != parser_defaults["chunking"] else (cfg.get("chunking") or args.chunking)
    eff_target = args.target_tokens if args.target_tokens != parser_defaults["target_tokens"] else int(cfg.get("target_tokens", args.target_tokens))
    eff_overlap = args.overlap_tokens if args.overlap_tokens != parser_defaults["overlap_tokens"] else int(cfg.get("overlap_tokens", args.overlap_tokens))
    eff_outdir = args.out_dir if args.out_dir != parser_defaults["out_dir"] else (cfg.get("default_index_dir") or args.out_dir)
    eff_scene_globs = args.scene_glob or cfg.get("scene_globs") or None

    build_index(
        model_name=eff_model,
        limit=eff_limit,
        manual_overrides_path=eff_overrides,
        knowledge_jsonl=eff_knowledge,
        rechunk=eff_rechunk,
        chunking=eff_chunking,
        target_tokens=eff_target,
        overlap_tokens=eff_overlap,
        out_dir=eff_outdir,
        scene_globs=eff_scene_globs,
    )
