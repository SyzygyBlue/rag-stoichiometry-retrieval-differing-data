#!/usr/bin/env python3
"""
DevOps orchestrator to build the RAG index and run a contexts-only smoke test
without relying on PowerShell piping/quoting.

- Imports and calls the in-repo scripts directly:
  * RAG Files/scripts/build_rag_index.py -> build_index(...)
  * RAG Files/scripts/rag_query.py       -> load_index, encode_query, top_k

Usage examples (run with your venv Python on Windows):

  "D:\\Windsurf Projects\\Create-RAG-Vector-Database\\RAG Files\\.venv\\Scripts\\python.exe" \
    "D:\\Windsurf Projects\\Create-RAG-Vector-Database\\DevOps\\orchestrate_build_and_smoke.py" \
    --limit 200 \
    --out-dir "D:\\Windsurf Projects\\Create-RAG-Vector-Database\\RAG Files\\indexes\\with_knowledge_smoke" \
    --query "What is the Three Shovels Caravan?" \
    --k 5

If sentence-transformers is not installed for the interpreter you use to run this
script, please run it with the project's venv Python above.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------
# Resolve project paths
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
RAG_DIR = os.path.join(PROJECT_ROOT, "RAG Files")
SCRIPTS_DIR = os.path.join(RAG_DIR, "scripts")

# Ensure we can import the repo scripts as modules
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Lazy-import to produce clear errors if deps are missing
try:
    import build_rag_index as bri
    import rag_query as rqi
except Exception as ex:
    print("[fatal] Failed to import repo scripts from:", SCRIPTS_DIR)
    print("        ", repr(ex))
    print("Tip: run this with the project's venv Python, e.g.\n  D:\\Windsurf Projects\\Create-RAG-Vector-Database\\RAG Files\\.venv\\Scripts\\python.exe ")
    sys.exit(1)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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




# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    default_config = os.path.join(RAG_DIR, "project.config.json")
    default_out = os.path.join(RAG_DIR, "indexes", "with_knowledge_smoke")

    ap = argparse.ArgumentParser(description="DevOps orchestrator: build the RAG index and run a retrieval/QA smoke test.")
    # General/config
    ap.add_argument("--config", default=default_config, help="Path to project config JSON (CLI overrides config)")
    ap.add_argument("--out-dir", default=default_out, help="Directory to write index artifacts to (build output)")
    ap.add_argument("--limit", type=int, default=200, help="Embed only first K chunks during build (0 = all)")
    ap.add_argument("--skip-build", action="store_true", help="Skip building index; run only retrieval/QA")
    ap.add_argument("--model", default=None, help="Embedding model for build/query (overrides config)")

    # Retrieval/QA controls (parity with rag_query.py)
    ap.add_argument("--query", default="What is the Three Shovels Caravan?", help="User question for smoke test")
    ap.add_argument("--k", type=int, default=5, help="Top-k contexts to retrieve")
    ap.add_argument("--index-dir", default="", help="If set, load index from here instead of --out-dir")
    ap.add_argument("--llama-url", default=None, help="Base URL of llama.cpp server (overrides config)")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max new tokens for generation (0 = unlimited)")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for generation")
    ap.add_argument("--no-llm", action="store_true", help="Skip llama call and only print/export contexts")
    ap.add_argument("--contexts-out", default="", help="Optional path to write retrieved contexts as JSON")
    ap.add_argument("--answer-out", default="", help="Optional path to write the generated answer to a file")
    ap.add_argument("--append-context", action="append", default=[], help="Path to a text/markdown file to append as extra CONTEXT (repeatable)")
    ap.add_argument("--verbosity", choices=["concise", "verbose", "voluminous"], default="concise", help="Control answer length and structure")
    ap.add_argument("--creativity", type=float, default=0.1, help="0.0-1.0 creative freedom: 0=strict canon, 1=creative")
    # Mixing (parity with rag_query)
    ap.add_argument("--mix", choices=["flat", "stratified"], default="", help="Retrieval mixing strategy: flat (single top-k) or stratified (per-bucket allocation)")
    ap.add_argument("--mix-ratio", default="", help="Bucket ratios, e.g., 'scene:4,character:1,knowledge:1'")
    ap.add_argument("--k-per-bucket", default="", help="Explicit per-bucket counts, e.g., 'scene:5,character:2,knowledge:1' (overrides --mix-ratio)")
    ap.add_argument("--mix-weights", default="", help="Per-bucket similarity weights, e.g., 'scene:1.0,character:1.05,knowledge:0.85'")
    ap.add_argument("--mix-order", choices=["by_similarity", "interleave"], default="", help="Final ordering of selected contexts")

    args = ap.parse_args()

    cfg = load_project_config(args.config)

    # Merge precedence: CLI > config > defaults
    parser_defaults: Dict[str, Any] = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "limit": 200,
        "manual_overrides": os.path.join(RAG_DIR, "mappings", "manual_alias_overrides.json"),
        "rechunk": False,
        "chunking": "payload",
        "target_tokens": 120,
        "overlap_tokens": 30,
        "out_dir": default_out,
        "llama_url": "http://127.0.0.1:8080",
        "verbosity": "concise",
        "creativity": 0.1,
    }

    eff_model = args.model if args.model is not None else cfg.get("model", parser_defaults["model"])  # build + query
    eff_limit = args.limit if args.limit != parser_defaults["limit"] else int(cfg.get("limit", args.limit))
    eff_overrides = cfg.get("manual_overrides", parser_defaults["manual_overrides"])  # build
    eff_rechunk = bool(cfg.get("rechunk", parser_defaults["rechunk"]))
    eff_chunking = cfg.get("chunking", parser_defaults["chunking"])  # payload|sentence|paragraph
    eff_target = int(cfg.get("target_tokens", parser_defaults["target_tokens"]))
    eff_overlap = int(cfg.get("overlap_tokens", parser_defaults["overlap_tokens"]))
    eff_outdir = args.out_dir if args.out_dir != parser_defaults["out_dir"] else (cfg.get("default_index_dir") or args.out_dir)
    # Ensure we end up in the requested folder, not the generic default root
    if eff_outdir == (cfg.get("default_index_dir") or parser_defaults["out_dir"]):
        eff_outdir = args.out_dir

    eff_scene_globs = cfg.get("scene_globs") or None
    eff_knowledge = cfg.get("knowledge_jsonl") or ""
    eff_llama = args.llama_url if args.llama_url is not None else (cfg.get("llama_url") or parser_defaults["llama_url"])
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

    # --------------------------------------------------------
    # Build phase
    # --------------------------------------------------------
    if not args.skip_build:
        os.makedirs(eff_outdir, exist_ok=True)
        print(f"[build] model={eff_model}")
        print(f"[build] out_dir={eff_outdir}")
        print(f"[build] limit={eff_limit}")
        try:
            bri.build_index(
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
        except SystemExit as se:
            print("[build][error]", se)
            sys.exit(2)
        except Exception as ex:
            print("[build][fatal]", repr(ex))
            sys.exit(2)

    # --------------------------------------------------------
    # Smoke test: contexts-only (no LLM)
    # --------------------------------------------------------
    test_index_dir = args.index_dir or eff_outdir
    emb_path = os.path.join(test_index_dir, "embeddings.npy")
    md_path = os.path.join(test_index_dir, "metadata.jsonl")
    if not (os.path.exists(emb_path) and os.path.exists(md_path)):
        print(f"[smoke][error] Missing index artifacts in {eff_outdir}.")
        print(f"                 Expected: embeddings.npy and metadata.jsonl")
        sys.exit(3)

    try:
        embs, metadatas, manifest = rqi.load_index(test_index_dir)
    except SystemExit as se:
        print("[smoke][error]", se)
        sys.exit(4)

    # Warn on model mismatch if present
    if manifest:
        idx_model = manifest.get("model")
        if idx_model and idx_model != eff_model:
            print(f"[smoke][warn] embedding model differs from manifest: using {eff_model} vs built {idx_model}")

    # Encode and retrieve
    try:
        q = rqi.encode_query(eff_model, args.query)
    except Exception as ex:
        print("[smoke][fatal] failed to encode query; ensure sentence-transformers is installed.")
        print("                ", repr(ex))
        sys.exit(5)

    tk = max(1, min(args.k, embs.shape[0]))
    # Stratification config and effective mixing options
    strat_cfg = (cfg.get("stratification") or {}) if isinstance(cfg, dict) else {}
    strat_defaults = (strat_cfg.get("defaults") or {}) if isinstance(strat_cfg, dict) else {}
    buckets_conf = (strat_cfg.get("buckets") or []) if isinstance(strat_cfg, dict) else []
    eff_mix = args.mix if args.mix else str(strat_defaults.get("mix_mode", "flat"))
    eff_mix_ratio = args.mix_ratio if args.mix_ratio else (strat_defaults.get("ratio") or {})
    eff_k_per_bucket = args.k_per_bucket if args.k_per_bucket else (strat_defaults.get("k_per_bucket") or "")
    eff_mix_weights = args.mix_weights if args.mix_weights else (strat_defaults.get("weights") or {})
    eff_mix_order = args.mix_order if args.mix_order else str(strat_defaults.get("ordering", "by_similarity"))

    selected = rqi.select_indices(
        embs=embs,
        metadatas=metadatas,
        q=q,
        total_k=tk,
        mix_mode=eff_mix,
        buckets_conf=buckets_conf,
        mix_ratio=eff_mix_ratio,
        k_per_bucket=eff_k_per_bucket,
        mix_weights=eff_mix_weights,
        mix_order=eff_mix_order,
    )

    contexts: List[Dict[str, Any]] = []
    for rank, (idx, sim, label) in enumerate(selected, start=1):
        md = dict(metadatas[idx])
        md["similarity"] = float(sim)
        md["bucket"] = label
        # Reuse rag_query's hydrate_text to avoid duplication
        md["text"] = rqi.hydrate_text(md)
        contexts.append(md)

    # Optionally add user-supplied extra contexts
    for p in (args.append_context or []):
        try:
            absp = os.path.abspath(p)
            with open(absp, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
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
            print(f"[smoke][warn] failed to append context from {p}: {ex}")

    # Print a concise summary to stdout
    print("=== Question ===")
    print(args.query)
    print("\n=== Top-K Contexts (scene_id :: chunk_type :: bucket :: similarity) ===")
    for c in contexts:
        print(f"- {c.get('scene_id')} :: {c.get('chunk_type')} :: {c.get('bucket')} :: {c.get('similarity'):.4f}")

    # Optionally write full contexts JSON
    out_contexts = args.contexts_out or os.path.join(test_index_dir, "contexts_sample.json")
    try:
        with open(out_contexts, "w", encoding="utf-8") as f:
            json.dump({"question": args.query, "contexts": contexts}, f, ensure_ascii=False, indent=2)
        print(f"[smoke] wrote contexts to {out_contexts}")
    except Exception as ex:
        print(f"[smoke][warn] failed to write contexts: {ex}")

    # If requested, run LLM step (rag_query parity)
    if not args.no_llm:
        sys_prompt, user_prompt = rqi.build_prompt(contexts, args.query, eff_verbosity, eff_creativity)
        # Determine effective max tokens: if user left default, pick by verbosity; 0 or negative => unlimited
        parser_default_max = 512
        if args.max_tokens != parser_default_max:
            eff_max_tokens = args.max_tokens
        else:
            eff_max_tokens = 256 if eff_verbosity == "concise" else (800 if eff_verbosity == "verbose" else 2048)

        try:
            answer = rqi.call_llama_chat(eff_llama, sys_prompt, user_prompt, max_tokens=eff_max_tokens, temperature=args.temperature)
        except Exception as e1:
            try:
                answer = rqi.call_llama_completion(eff_llama, sys_prompt, user_prompt, max_tokens=eff_max_tokens, temperature=args.temperature)
            except Exception as e2:
                print("[smoke][error] llama.cpp call failed:\n- chat endpoint error: ", repr(e1), "\n- completion endpoint error: ", repr(e2))
                sys.exit(6)
        print("\n=== Answer ===")
        print(answer)
        if args.answer_out:
            try:
                out_dir = os.path.dirname(args.answer_out)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.answer_out, "w", encoding="utf-8") as f:
                    f.write(answer)
                print(f"[smoke] wrote answer to {args.answer_out}")
            except Exception as ex:
                print(f"[smoke][warn] failed to write answer: {ex}")


if __name__ == "__main__":
    main()
