#!/usr/bin/env python3
"""
Convert Q&A knowledge JSONL to a scene-like JSON with embedding_payloads
so it can be indexed by build_rag_index.py without code changes.

Usage:
  python ingest_knowledge_jsonl.py \
    --input "d:/Windsurf Projects/Create-RAG-Vector-Database/RAG Files/knowledgeFile/knowledge.jsonl" \
    --output "d:/Windsurf Projects/Create-RAG-Vector-Database/RAG Files/knowledge_qa.json" \
    --book-title "Neon Titan Awakens"

Notes:
- Each JSONL line must contain at least keys: instruction (Q), output (A), metadata (object optional)
- We write a single JSON file with fields:
  {
    "doc_type": "knowledge_qa",
    "source": {"book_title": <str>, "chapter_title": "Knowledge Base"},
    "embedding_payloads": [
      {"chunk_id": "qa-000001", "type": "knowledge-qa", "text": "Q: ...\nA: ...", "meta": {...}},
      ...
    ]
  }
"""

import argparse
import json
import os
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception as e:
                raise SystemExit(f"Invalid JSON on line {i}: {e}")
    return rows


def build_scene(rows: List[Dict[str, Any]], *, book_title: str = "", chapter_title: str = "Knowledge Base", limit: int = 0) -> Dict[str, Any]:
    payloads = []
    count = 0
    for i, r in enumerate(rows, start=1):
        instr = (r or {}).get("instruction")
        out = (r or {}).get("output")
        if not isinstance(instr, str) or not instr.strip():
            continue
        if not isinstance(out, str) or not out.strip():
            continue
        meta = (r or {}).get("metadata")
        # Build a compact QA text block
        text = f"Q: {instr.strip()}\nA: {out.strip()}"
        payloads.append({
            "chunk_id": f"qa-{i:06d}",
            "type": "knowledge-qa",
            "text": text,
            "meta": meta if isinstance(meta, dict) else None,
        })
        count += 1
        if limit and count >= limit:
            break

    scene: Dict[str, Any] = {
        "doc_type": "knowledge_qa",
        "source": {
            "book_title": book_title or None,
            "chapter_title": chapter_title,
        },
        "embedding_payloads": payloads,
    }
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge JSONL to scene-like JSON")
    parser.add_argument("--input", required=True, help="Path to knowledge JSONL file")
    parser.add_argument("--output", required=True, help="Output path for scene JSON (will overwrite)")
    parser.add_argument("--book-title", default="Neon Titan Awakens", help="Book title for source metadata")
    parser.add_argument("--chapter-title", default="Knowledge Base", help="Chapter title for source metadata")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of QA entries (0 = all)")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit("No rows found in input JSONL")

    scene = build_scene(rows, book_title=args.book_title, chapter_title=args.chapter_title, limit=args.limit)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)

    print(f"Wrote knowledge scene: {args.output}")
    print(f"Chunks: {len(scene.get('embedding_payloads') or [])}")


if __name__ == "__main__":
    main()
