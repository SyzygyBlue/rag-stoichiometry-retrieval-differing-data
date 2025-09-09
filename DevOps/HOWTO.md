# HOWTO: Build, Query, and Test the RAG Pipeline (DevOps Orchestrator)

This guide explains how to use `DevOps/orchestrate_build_and_smoke.py` to build the RAG index and run a retrieval/QA smoke test. It mirrors key options from `RAG Files/scripts/rag_query.py` for consistent behavior.

## What the orchestrator does

- Imports and runs `build_rag_index.py` to build embeddings + metadata into an index directory.
- Imports `rag_query.py` helpers to encode the query, retrieve top-K contexts, build prompts, and optionally call a local llama.cpp server.
- Hydrates context text for both:
  - Scene-backed chunks (via `scene_id`, `chunk_uid`, `embedding_payloads`, and `text_windows`).
  - Knowledge-backed chunks from `metadata.extra` (e.g., `instruction`/`output` or `text`).

Data sources are configured via `RAG Files/project.config.json` (CLI overrides config): scenes, entities, characters, mappings, and optional `knowledge_jsonl`.

## Prerequisites

- Use the project venv: `RAG Files/.venv/` (Windows path: `RAG Files\.venv\Scripts\python.exe`).
- Required packages installed in that venv: sentence-transformers, numpy, requests.
- Input data present under `RAG Files/`:
  - Scenes and related JSON.
  - `mappings/manual_alias_overrides.json` for alias-to-entity normalization (if used).
  - `entities/entities.json` and `character/`.
  - Optional knowledge at `knowledgeFile/knowledge.jsonl` (path set in config as `knowledge_jsonl`).

## Quick commands

- Show help (recommended):
  ```bat
  DevOps\run_orchestrator.cmd --help
  ```

- Build index and retrieve top-5 contexts (no LLM):
  ```bat
  DevOps\run_orchestrator.cmd ^
    --limit 200 ^
    --out-dir "RAG Files\indexes\with_knowledge_smoke" ^
    --query "What is the Three Shovels Caravan?" --k 5 --no-llm
  ```

- Query an existing index (skip build):
  ```bat
  DevOps\run_orchestrator.cmd ^
    --skip-build ^
    --index-dir "RAG Files\indexes\with_knowledge_smoke" ^
    --query "Who runs the Caravan and where do they travel?" --k 6 --no-llm
  ```

- Full end-to-end with llama.cpp answer:
  ```bat
  DevOps\run_orchestrator.cmd ^
    --out-dir "RAG Files\indexes\with_knowledge_smoke" ^
    --query "Summarize the Caravan's mission and key members" --k 6 ^
    --llama-url http://127.0.0.1:8080 --max-tokens 512 --temperature 0.2
  ```

## CLI options (parity with `rag_query.py`)

- General/config
  - `--config`: Project config path. Default: `RAG Files/project.config.json`.
  - `--out-dir`: Build output for index artifacts (embeddings + metadata).
  - `--limit`: Build-time limit of chunks to embed (0 = all).
  - `--skip-build`: Skip building; only run retrieval/QA.
  - `--model`: Embedding model for both build and query (overrides config).

- Retrieval/QA
  - `--query`: User question for the smoke test.
  - `--k`: Top-K contexts to retrieve.
  - `--index-dir`: Load an existing index instead of `--out-dir`.
  - `--llama-url`: Base URL of llama.cpp server (overrides config), e.g., `http://127.0.0.1:8080`.
  - `--max-tokens`: Max new tokens for generation.
  - `--temperature`: Sampling temperature.
  - `--no-llm`: Skip llama call; only print/export contexts.
  - `--contexts-out`: Optional path to write contexts as JSON (defaults to `<index>/contexts_sample.json`).
  - `--answer-out`: Optional path to write the generated answer.
  - `--append-context`: File to append as extra context (repeatable).
  - `--verbosity {concise, verbose, voluminous}`: Controls answer length/structure.
  - `--creativity [0..1]`: 0.0=strict canon, 1.0=creative; also supports config key `default_creativity`.

Notes:
- CLI overrides config values.
- The orchestrator warns if the loaded index model differs from the current `--model`.
- If chat endpoint fails, it automatically falls back to the completion endpoint.

## Knowledge integration and verification

- Ensure `project.config.json` has: `"knowledge_jsonl": "RAG Files/knowledgeFile/knowledge.jsonl"`.
- During build, knowledge entries are embedded alongside scene chunks.
- During retrieval, knowledge chunks typically have `chunk_type: "knowledge_qa"` and their text is hydrated from `metadata.extra`:
  - `extra.text` if present, otherwise `extra.instruction`/`extra.output` are combined into Q/A.

To verify:
1. Run a query that should hit knowledge.
2. Open the written contexts file: `RAG Files\indexes\with_knowledge_smoke\contexts_sample.json` (or your `--contexts-out`).
3. Look for entries with `chunk_type` = `knowledge_qa` and confirm `text` contains the knowledge Q/A.

## LLM usage (optional)

- Start llama.cpp server (examples vary). Set `--llama-url` if not the default `http://127.0.0.1:8080`.
- Control style with `--verbosity` and `--creativity` (or config keys `default_verbosity`, `default_creativity`).
- Outputs:
  - Printed to console.
  - Written to `--answer-out` if provided.

## Troubleshooting

- Missing `embeddings.npy` or `metadata.jsonl` in your index directory:
  - Ensure you built to the same `--out-dir` or pass `--index-dir`.
- "Failed to import repo scripts":
  - Run with the project venv Python or via `DevOps\run_orchestrator.cmd`.
- Query encoding failure:
  - Install sentence-transformers in the project venv.
- No knowledge results:
  - Confirm `knowledge_jsonl` path in config and rerun the build.

## Help

```bat
DevOps\run_orchestrator.cmd --help
```
