#!/usr/bin/env python3
r"""
RAG query runner: retrieves top-k chunks from the built index and asks a local llama.cpp server.

Usage examples (PowerShell):
  .\RAG Files\.venv\Scripts\python.exe \
    ".\RAG Files\scripts\rag_query.py" \
    --query "Who visits Demmy in his dorm and how do they enter?" \
    --k 5 \
    --llama-url "http://127.0.0.1:8080" \
    --max-tokens 512

Requirements: sentence-transformers, numpy, requests
Assumes the index artifacts exist under RAG Files/indexes/ from build_rag_index.py
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import fnmatch

SCRIPT_DIR = os.path.dirname(__file__)
RAG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # RAG Files/
INDEX_DIR = os.path.join(RAG_DIR, "indexes")
 

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


def load_index(index_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
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
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass
    return embs, metadatas, manifest


def encode_query(model_name: str, query: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return q.astype(np.float32, copy=False)[0]


def top_k(embs: np.ndarray, q: np.ndarray, k: int) -> List[Tuple[int, float]]:
    # embs expected to be L2-normalized; cosine = dot
    sims = embs @ q
    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        # efficient top-k
        part = np.argpartition(-sims, k)[:k]
        idx = part[np.argsort(-sims[part])]
    return [(int(i), float(sims[i])) for i in idx[:k]]


# -----------------
# Stratified retrieval helpers
# -----------------

def _parse_kv_map(spec: Any, val_type: str = "int") -> Dict[str, float]:
    """Parse mapping from either a dict or a string like 'a:2,b:1'. Returns {label: value}.
    val_type in {"int","float"} determines parsing of values.
    """
    if isinstance(spec, dict):
        out: Dict[str, float] = {}
        for k, v in spec.items():
            if k is None:
                continue
            try:
                out[str(k)] = float(int(v)) if val_type == "int" else float(v)
            except Exception:
                continue
        return out
    out: Dict[str, float] = {}
    if not isinstance(spec, str) or not spec.strip():
        return out
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                out[k] = float(int(v)) if val_type == "int" else float(v)
            except Exception:
                continue
    return out


def _bucket_match_label(md: Dict[str, Any], buckets_conf: List[Dict[str, Any]]) -> str:
    """Return the first matching bucket label based on priority (higher wins). Fallbacks below if none match.
    Supported match keys: doc_type, scene_id, chunk_type, chunk_type_prefix, file.
    """
    # Sort by priority DESC so higher number wins
    buckets_sorted = sorted((b for b in (buckets_conf or []) if isinstance(b, dict)), key=lambda x: int((x or {}).get("priority", 0)), reverse=True)
    src = (md.get("source") or {})
    doc_type = str(src.get("doc_type") or "").lower()
    scene_id = str(md.get("scene_id") or "")
    chunk_type = str(md.get("chunk_type") or "").lower()
    file_base = str(md.get("file") or "").lower()

    def matches(match: Dict[str, Any]) -> bool:
        if not isinstance(match, dict):
            return False
        # Each key present must match one of its values (AND across keys, OR within a key)
        for key, vals in match.items():
            vals = vals if isinstance(vals, list) else [vals]
            vals = [str(v).lower() for v in vals if isinstance(v, (str, int, float))]
            if key == "doc_type":
                if vals and doc_type not in vals:
                    return False
            elif key == "scene_id":
                if vals and scene_id.lower() not in vals:
                    return False
            elif key == "chunk_type":
                if vals and chunk_type not in vals:
                    return False
            elif key == "chunk_type_prefix":
                if vals and not any(chunk_type.startswith(v) for v in vals):
                    return False
            elif key == "file":
                if vals and file_base not in vals:
                    return False
            elif key == "path_glob":
                # We only have the basename in metadata; attempt fnmatch against basename
                if vals and not any(fnmatch.fnmatch(file_base, v) for v in vals):
                    return False
            else:
                # Unknown matcher key: ignore
                continue
        return True

    for b in buckets_sorted:
        label = str(b.get("label") or "").strip()
        if not label:
            continue
        if matches(b.get("match") or {}):
            return label

    # Fallback heuristics
    if "character" in doc_type:
        return "character"
    if ("knowledge" in doc_type) or (scene_id in {"knowledge", "knowledge_qa"}) or chunk_type.startswith("knowledge"):
        return "knowledge"
    if "scene" in doc_type:
        return "scene"
    return "__unassigned__"


def _allocate_counts(total_k: int, ratio_map: Dict[str, float]) -> Dict[str, int]:
    """Allocate integer counts that sum to total_k using largest-remainder method from a ratio map."""
    ratio_map = {k: float(v) for k, v in (ratio_map or {}).items() if float(v) > 0.0}
    if not ratio_map:
        return {}
    s = sum(ratio_map.values())
    if s <= 0:
        return {}
    ideal = {k: (total_k * v / s) for k, v in ratio_map.items()}
    counts = {k: int(v) for k, v in ideal.items()}
    rem = total_k - sum(counts.values())
    if rem > 0:
        # Distribute remaining to the largest fractional remainders
        fracs = sorted(((k, ideal[k] - counts[k]) for k in counts.keys()), key=lambda x: x[1], reverse=True)
        for i in range(rem):
            counts[fracs[i % len(fracs)][0]] += 1
    return counts


def select_indices(
    embs: np.ndarray,
    metadatas: List[Dict[str, Any]],
    q: np.ndarray,
    total_k: int,
    mix_mode: str,
    buckets_conf: List[Dict[str, Any]],
    mix_ratio: Any,
    k_per_bucket: Any,
    mix_weights: Any,
    mix_order: str = "by_similarity",
) -> List[Tuple[int, float, str]]:
    """Return a list of (index, sim, bucket_label) selected per the mixing strategy.
    mix_mode: 'flat' or 'stratified'
    When stratified, k_per_bucket (counts) takes precedence over mix_ratio.
    mix_ratio/weights may be dicts (from config) or strings (from CLI)."""
    sims = embs @ q
    n = len(metadatas)
    total_k = max(1, min(int(total_k), n))

    if mix_mode != "stratified":
        # Flat top-k
        if total_k >= n:
            idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, total_k)[:total_k]
            idx = part[np.argsort(-sims[part])]
        out: List[Tuple[int, float, str]] = []
        for i in idx[:total_k]:
            md = metadatas[int(i)]
            label = _bucket_match_label(md, buckets_conf)
            out.append((int(i), float(sims[int(i)]), label))
        return out

    # Stratified mode
    weights = _parse_kv_map(mix_weights, val_type="float")
    counts_map = _parse_kv_map(k_per_bucket, val_type="int")
    if not counts_map:
        counts_map = _allocate_counts(total_k, _parse_kv_map(mix_ratio, val_type="int"))

    # Group candidates per bucket
    pools: Dict[str, List[Tuple[int, float, float]]] = {}
    for i in range(n):
        md = metadatas[i]
        label = _bucket_match_label(md, buckets_conf)
        w = float(weights.get(label, 1.0))
        sim = float(sims[i])
        simw = sim * w
        pools.setdefault(label, []).append((i, simw, sim))

    # Sort each pool by weighted similarity
    for label in pools.keys():
        pools[label].sort(key=lambda t: t[1], reverse=True)

    selected: List[Tuple[int, float, str]] = []
    taken: set = set()

    # First pass: take per-bucket counts
    want_total = 0
    for label, cnt in counts_map.items():
        c = int(cnt)
        want_total += c
        pool = pools.get(label, [])
        for (i, simw, sim) in pool[:c]:
            if i in taken:
                continue
            taken.add(i)
            selected.append((i, sim, label))

    # If counts_map sums to less than total_k or some buckets were empty, fill remaining globally
    remain = max(total_k, want_total) - len(selected)
    if remain > 0:
        # Build a global leftover list excluding already taken
        leftovers: List[Tuple[int, float, float, str]] = []
        for label, pool in pools.items():
            for (i, simw, sim) in pool:
                if i in taken:
                    continue
                leftovers.append((i, simw, sim, label))
        leftovers.sort(key=lambda t: t[1], reverse=True)
        for (i, _simw, sim, label) in leftovers[:remain]:
            if i in taken:
                continue
            taken.add(i)
            selected.append((i, sim, label))

    # Order final list
    if mix_order == "interleave" and counts_map:
        # Build order by round-robin over buckets present in counts_map
        buckets_order = [lbl for lbl in counts_map.keys() if any(lbl == s[2] for s in selected)]
        buckets_iters: Dict[str, List[Tuple[int, float, str]]] = {lbl: [] for lbl in buckets_order}
        # For fairness, sort within each bucket by original similarity desc
        for lbl in buckets_order:
            items = [s for s in selected if s[2] == lbl]
            items.sort(key=lambda x: x[1], reverse=True)
            buckets_iters[lbl] = items
        out: List[Tuple[int, float, str]] = []
        pos = 0
        while any(buckets_iters.values()) and len(out) < total_k:
            lbl = buckets_order[pos % len(buckets_order)]
            if buckets_iters[lbl]:
                out.append(buckets_iters[lbl].pop(0))
            pos += 1
        # If still short due to exhausted buckets, append remaining sorted by similarity
        if len(out) < total_k:
            rem2 = [s for s in selected if s not in out]
            rem2.sort(key=lambda x: x[1], reverse=True)
            out.extend(rem2[: total_k - len(out)])
        return out[:total_k]
    else:
        # Sort globally by original similarity desc
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[:total_k]


def hydrate_text(md: Dict[str, Any]) -> str:
    """Rehydrate a short text for a metadata row using metadata extras or scene JSON."""
    try:
        # 1) Knowledge or metadata-provided text
        ex = md.get("extra") or {}
        if isinstance(ex, dict):
            t = ex.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()
            instr = ex.get("instruction")
            outp = ex.get("output")
            parts = []
            if isinstance(instr, str) and instr.strip():
                parts.append("Q: " + instr.strip())
            if isinstance(outp, str) and outp.strip():
                parts.append("A: " + outp.strip())
            if parts:
                return "\n".join(parts)

        # 2) Scene-backed text from scene JSON
        chunk_uid = md.get("chunk_uid", "")
        scene_id = md.get("scene_id")
        scene_path = os.path.join(RAG_DIR, f"{scene_id}.json")
        if not os.path.exists(scene_path):
            return ""
        with open(scene_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
        # Try to find matching embedding_payloads by chunk_id suffix
        eps = scene.get("embedding_payloads") or []
        # chunk_uid like: "scene::chunkid"
        if "::" in chunk_uid:
            cid = chunk_uid.split("::", 1)[1]
        else:
            cid = None
        if cid:
            for it in eps:
                if it.get("chunk_id") == cid:
                    t2 = it.get("text")
                    if isinstance(t2, str):
                        return t2.strip()
        # Fallbacks: use text_windows
        tw = scene.get("text_windows") or {}
        for key in ("dense_recap_250w", "full_scene_clean"):
            val = tw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    except Exception:
        return ""


def build_prompt(contexts: List[Dict[str, Any]], question: str, verbosity: str = "concise", creativity: float = 0.1) -> Tuple[str, str]:
    # Clamp creativity to [0, 1]
    try:
        c = float(creativity)
    except Exception:
        c = 0.1
    c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)

    # Policy based on creativity (canon adherence vs. invention)
    if c <= 0.2:
        policy = (
            "Adhere strictly to the provided CONTEXT. Do not invent facts. "
            "If information is missing, reply that it is unknown. Treat canon as immutable."
        )
    elif c <= 0.6:
        policy = (
            "Ground answers in the CONTEXT. You may make small, clearly-labeled inferences that do not contradict the CONTEXT."
        )
    else:
        policy = (
            "Ground answers in the CONTEXT, but you may extend creatively when appropriate. "
            "Clearly label any invented or speculative content as 'Speculative'. Do not contradict established world rules."
        )

    # Verbosity instruction
    if verbosity == "verbose":
        verbosity_instr = "Answer thoroughly with short sections and bullet points; aim for 300-600 words."
    elif verbosity == "voluminous":
        verbosity_instr = "Answer voluminously with long sections and bullet points; aim for 1200-2400 words."
    else:
        verbosity_instr = "Answer concisely (<=150 words or 3-6 bullets)."

    sys_prompt = (
        "You are a helpful assistant.\n"
        f"{policy}\n"
        "Cite scene_id(s) in a short References section."
    )

    ctx_lines: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        scene_id = ctx.get("scene_id")
        ctype = ctx.get("chunk_type")
        source = ctx.get("source", {})
        chap = source.get("chapter_title")
        dateu = source.get("date_in_universe")
        ents = ", ".join(sorted({e.get("name", e.get("id", "")) for e in (ctx.get("entities") or [])}))
        ctx_lines.append(
            f"[CTX {i}] scene_id={scene_id} chunk_type={ctype} chapter={chap} date={dateu} entities=[{ents}]\n"
            f"{ctx.get('text','').strip()}\n"
        )

    instr_lines: List[str] = []
    instr_lines.append("- " + ("Answer using only the CONTEXT." if c <= 0.2 else "Answer primarily from the CONTEXT."))
    instr_lines.append(f"- {verbosity_instr}")
    if 0.2 < c <= 0.6:
        instr_lines.append("- If you infer beyond the text, mark it as 'Inference' and keep it minimal.")
    if c > 0.6:
        instr_lines.append("- Clearly mark any invented content as 'Speculative'.")
    instr_lines.append("- Include a References line listing relevant scene_id(s).")

    user_prompt = (
        "CONTEXT\n"
        + "\n".join(ctx_lines)
        + "\nQUESTION\n" + question
        + "\n\nInstructions:\n" + "\n".join(instr_lines) + "\n"
    )
    return sys_prompt, user_prompt


def call_llama_chat(url: str, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": "local-llama",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    # Unlimited if max_tokens <= 0: omit the field and let server decide, or fallback to completion path
    if max_tokens is not None and max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    r = requests.post(endpoint, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def call_llama_completion(url: str, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    endpoint = url.rstrip("/") + "/completion"
    prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}"
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        # Unlimited if max_tokens <= 0 per llama.cpp semantics
        "n_predict": (-1 if (max_tokens is None or max_tokens <= 0) else int(max_tokens)),
        "stream": False,
    }
    r = requests.post(endpoint, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    # Try common fields
    if isinstance(data, dict):
        if "content" in data and isinstance(data["content"], str):
            return data["content"].strip()
        if "generation" in data and isinstance(data["generation"], str):
            return data["generation"].strip()
        if "choices" in data and data["choices"]:
            ch0 = data["choices"][0]
            # OpenAI-like completion shape
            if isinstance(ch0, dict):
                if "text" in ch0:
                    return str(ch0["text"]).strip()
                if "message" in ch0 and isinstance(ch0["message"], dict) and "content" in ch0["message"]:
                    return str(ch0["message"]["content"]).strip()
    return json.dumps(data, ensure_ascii=False)


def call_openai_chat(api_base: str, api_key: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call OpenAI Chat Completions API.
    api_base example: https://api.openai.com
    model example: gpt-5, gpt-4o, gpt-4.1, etc.
    api_key is required (or use Azure-compatible base/headers if proxied).
    """
    endpoint = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None and max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    r = requests.post(endpoint, headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def read_api_key_from_file(path: str) -> str:
    """Read an API key from a file. Accepts either 'key=...'^ style or a raw key line.
    Returns empty string if not found or on error.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().strip('"').strip("'")
                if not s:
                    continue
                if s.lower().startswith("key="):
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
                # Fallback: return first non-empty line
                return s
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description="Query RAG index and ask local llama.cpp server")
    ap.add_argument("--query", required=True, help="User question")
    ap.add_argument("--k", type=int, default=5, help="Top-k contexts")
    # Optional project config; CLI overrides config values
    default_config = os.path.join(RAG_DIR, "project.config.json")
    ap.add_argument("--config", default=default_config, help="Path to project config JSON (CLI overrides config)")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080", help="Base URL of llama.cpp server")
    ap.add_argument("--provider", choices=["llama", "openai"], default="llama", help="LLM provider to use")
    ap.add_argument("--openai-model", default="gpt-4o", help="OpenAI model name (e.g., gpt-5, gpt-4o)")
    ap.add_argument("--openai-base", default="https://api.openai.com", help="OpenAI API base URL")
    ap.add_argument("--openai-api-key", default="", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    ap.add_argument("--openai-key-file", default="", help="Path to file containing OpenAI key (supports 'key=...' or raw key)")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max new tokens for generation (0 = unlimited)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model used for index")
    ap.add_argument("--index-dir", default=INDEX_DIR, help="Directory containing index artifacts (embeddings.npy, metadata.jsonl)")
    ap.add_argument("--no-llm", action="store_true", help="Skip llama call and only print/export contexts")
    ap.add_argument("--contexts-out", default="", help="Optional path to write retrieved contexts as JSON")
    ap.add_argument("--answer-out", default="", help="Optional path to write the generated answer to a file")
    ap.add_argument("--append-context", action="append", default=[], help="Path to a text/markdown file to append as an extra CONTEXT block (repeatable)")
    ap.add_argument("--verbosity", choices=["concise", "verbose", "voluminous"], default="concise", help="Control answer length and structure.")
    ap.add_argument("--creativity", type=float, default=0.1, help="0.0-1.0 creative freedom: 0=strict canon, 1=full creative latitude.")
    # Mixing controls
    ap.add_argument("--mix", choices=["flat", "stratified"], default="", help="Retrieval mixing strategy: flat (single top-k) or stratified (per-bucket allocation)")
    ap.add_argument("--mix-ratio", default="", help="Bucket ratios, e.g., 'scene:4,character:1,knowledge:1'")
    ap.add_argument("--k-per-bucket", default="", help="Explicit per-bucket counts, e.g., 'scene:5,character:2,knowledge:1' (overrides --mix-ratio)")
    ap.add_argument("--mix-weights", default="", help="Per-bucket similarity weights, e.g., 'scene:1.0,character:1.05,knowledge:0.85'")
    ap.add_argument("--mix-order", choices=["by_similarity", "interleave"], default="", help="Final ordering of selected contexts")
    args = ap.parse_args()

    cfg = load_project_config(args.config)

    # Precedence: CLI > config > defaults
    parser_defaults = {
        "llama_url": "http://127.0.0.1:8080",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "index_dir": INDEX_DIR,
        "verbosity": "concise",
        "creativity": 0.1,
        "provider": "llama",
        "openai_model": "gpt-4o",
        "openai_base": "https://api.openai.com",
        "openai_key_file": "",
    }
    eff_llama = args.llama_url if args.llama_url != parser_defaults["llama_url"] else (cfg.get("llama_url") or args.llama_url)
    eff_provider = args.provider if args.provider != parser_defaults["provider"] else (cfg.get("provider") or args.provider)
    eff_openai_model = args.openai_model if args.openai_model != parser_defaults["openai_model"] else (cfg.get("openai_model") or args.openai_model)
    eff_openai_base = args.openai_base if args.openai_base != parser_defaults["openai_base"] else (cfg.get("openai_base") or args.openai_base)
    # API key precedence: CLI > env > file > config
    eff_openai_api_key = args.openai_api_key.strip() if args.openai_api_key.strip() else os.environ.get("OPENAI_API_KEY", "").strip()
    eff_openai_key_file = args.openai_key_file if getattr(args, "openai_key_file", "") else (cfg.get("openai_key_file") or "")
    if not eff_openai_api_key and eff_openai_key_file:
        eff_openai_api_key = read_api_key_from_file(eff_openai_key_file)
    if not eff_openai_api_key:
        eff_openai_api_key = str(cfg.get("openai_api_key", "")).strip()
    eff_model = args.model if args.model != parser_defaults["model"] else (cfg.get("model") or args.model)
    eff_index = args.index_dir if args.index_dir != parser_defaults["index_dir"] else (cfg.get("default_index_dir") or args.index_dir)
    # config keys allow overrides: default_verbosity, default_creativity
    eff_verbosity = args.verbosity if args.verbosity != parser_defaults["verbosity"] else (cfg.get("default_verbosity") or args.verbosity)
    eff_creativity = args.creativity if args.creativity != parser_defaults["creativity"] else (cfg.get("default_creativity") or args.creativity)
    try:
        eff_creativity = float(eff_creativity)
    except Exception:
        eff_creativity = parser_defaults["creativity"]
    if eff_creativity < 0.0:
        eff_creativity = 0.0
    elif eff_creativity > 1.0:
        eff_creativity = 1.0

    # Load index
    embs, metadatas, manifest = load_index(eff_index)
    if manifest:
        idx_dim = int(manifest.get("dimension", embs.shape[1]))
        if embs.shape[1] != idx_dim:
            print(f"[warn] embedding dimension mismatch: {embs.shape[1]} vs manifest {idx_dim}", file=sys.stderr)
        idx_model = manifest.get("model")
        if idx_model and idx_model != eff_model:
            print(f"[warn] embedding model differs from manifest: {eff_model} vs {idx_model}", file=sys.stderr)

    # Encode query
    q = encode_query(eff_model, args.query)

    # Retrieve (flat or stratified)
    tk = max(1, min(args.k, embs.shape[0]))
    strat_cfg = (cfg.get("stratification") or {}) if isinstance(cfg, dict) else {}
    strat_defaults = (strat_cfg.get("defaults") or {}) if isinstance(strat_cfg, dict) else {}
    buckets_conf = (strat_cfg.get("buckets") or []) if isinstance(strat_cfg, dict) else []
    eff_mix = args.mix if args.mix else str(strat_defaults.get("mix_mode", "flat"))
    eff_mix_ratio = args.mix_ratio if args.mix_ratio else (strat_defaults.get("ratio") or {})
    eff_k_per_bucket = args.k_per_bucket if args.k_per_bucket else (strat_defaults.get("k_per_bucket") or "")
    eff_mix_weights = args.mix_weights if args.mix_weights else (strat_defaults.get("weights") or {})
    eff_mix_order = args.mix_order if args.mix_order else str(strat_defaults.get("ordering", "by_similarity"))

    selected = select_indices(
        embs,
        metadatas,
        q,
        tk,
        eff_mix,
        buckets_conf,
        eff_mix_ratio,
        eff_k_per_bucket,
        eff_mix_weights,
        eff_mix_order,
    )

    # Attach text to metadatas for prompt (we didn't persist text; re-extract from scenes is heavy)
    # As a pragmatic approach, include only metadata + chunk_uid, and remind the model to answer from metadata. If you need full text, extend index to store text.
    # However, our build stored only metadata and embeddings. To provide text, we can reconstruct from scene files using chunk_uid.
    # For now, we will try to fetch a short text surrogate: use chunk_type and known fields from the scene payloads.

    # Better approach: reopen the scene JSON and pull text via chunk_uid when available.
    contexts: List[Dict[str, Any]] = []
    for rank, (idx, sim, label) in enumerate(selected, start=1):
        md = dict(metadatas[idx])
        md["similarity"] = sim
        md["bucket"] = label
        md["text"] = hydrate_text(md)
        contexts.append(md)

    # Optionally add user-supplied extra contexts (e.g., files to compare styles against)
    for p in (args.append_context or []):
        try:
            absp = os.path.abspath(p)
            with open(absp, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            # Trim excessively long additions to keep prompt size manageable
            max_len = 16000
            if len(txt) > max_len:
                txt = txt[:max_len]
            contexts.append({
                "scene_id": f"user-file:{os.path.basename(absp)}",
                "chunk_type": "user_supplied",
                "similarity": 1.0,
                "text": txt,
                "entities": [],
                "source": {"chapter_title": "user-supplied", "path": absp},
            })
        except Exception as ex:
            print(f"[warn] failed to append context from {p}: {ex}", file=sys.stderr)

    sys_prompt, user_prompt = build_prompt(contexts, args.query, eff_verbosity, eff_creativity)

    # Always export contexts if requested (regardless of LLM call)
    if args.contexts_out:
        try:
            with open(args.contexts_out, "w", encoding="utf-8") as f:
                json.dump({"question": args.query, "contexts": contexts}, f, ensure_ascii=False, indent=2)
            print(f"[info] wrote contexts to {args.contexts_out}")
        except Exception as ex:
            print(f"[warn] failed to write contexts: {ex}", file=sys.stderr)

    if args.no_llm:
        print("=== Question ===")
        print(args.query)
        print("\n=== Top-K Contexts (scene_id :: chunk_type :: bucket :: similarity) ===")
        for c in contexts:
            print(f"- {c.get('scene_id')} :: {c.get('chunk_type')} :: {c.get('bucket')} :: {c.get('similarity'):.4f}")
        return

    # Try OpenAI-compatible chat endpoint first, then fallback
    # Determine effective max tokens: if user left default, pick by verbosity; 0 or negative => unlimited
    parser_default_max = 512
    if args.max_tokens != parser_default_max:
        eff_max_tokens = args.max_tokens
    else:
        eff_max_tokens = 256 if eff_verbosity == "concise" else (800 if eff_verbosity == "verbose" else 2048)

    if eff_provider == "openai":
        if not eff_openai_api_key:
            print("[error] OPENAI_API_KEY is required for provider=openai. Set --openai-api-key or env var.", file=sys.stderr)
            sys.exit(2)
        try:
            answer = call_openai_chat(eff_openai_base, eff_openai_api_key, eff_openai_model, sys_prompt, user_prompt, max_tokens=eff_max_tokens, temperature=args.temperature)
        except Exception as e:
            print(f"[error] OpenAI chat call failed: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        try:
            answer = call_llama_chat(eff_llama, sys_prompt, user_prompt, max_tokens=eff_max_tokens, temperature=args.temperature)
        except Exception as e1:
            try:
                answer = call_llama_completion(eff_llama, sys_prompt, user_prompt, max_tokens=eff_max_tokens, temperature=args.temperature)
            except Exception as e2:
                print("[error] llama.cpp call failed:\n- chat endpoint error: ", repr(e1), "\n- completion endpoint error: ", repr(e2), file=sys.stderr)
                sys.exit(2)

    # Display
    print("=== Question ===")
    print(args.query)
    print("\n=== Top-K Contexts (scene_id :: chunk_type :: bucket :: similarity) ===")
    for c in contexts:
        print(f"- {c.get('scene_id')} :: {c.get('chunk_type')} :: {c.get('bucket')} :: {c.get('similarity'):.4f}")
    print("\n=== Answer ===")
    print(answer)

    # Optional: write answer to file
    if args.answer_out:
        try:
            out_dir = os.path.dirname(args.answer_out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.answer_out, "w", encoding="utf-8") as f:
                f.write(answer)
            print(f"[info] wrote answer to {args.answer_out}")
        except Exception as ex:
            print(f"[warn] failed to write answer: {ex}", file=sys.stderr)
    # End


if __name__ == "__main__":
    main()
