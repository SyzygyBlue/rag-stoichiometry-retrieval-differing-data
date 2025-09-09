# ETL: Chapter-to-RAG pipeline for "Beyond the Neon Veil"

This document explains how to design the scripts (no code here) that convert a new MS Word chapter into RAG-ready assets (scenes, characters, entities, knowledge) and place them under `RAG Files/` so the DevOps orchestrator can build the index and run a retrieval/QA smoke test.

Key downstream components these scripts must satisfy:
- `RAG Files/scripts/build_rag_index.py` expects scene JSONs and optional knowledge JSONL.
- `DevOps/orchestrate_build_and_smoke.py` builds and smoke-tests using config from `RAG Files/project.config.json`.

## Goals
- Take `ChapterXX.docx` and produce:
  - One or more scene JSON files with clean text and chunk payloads.
  - Updated character and entity catalogs.
  - Optional knowledge Q/A appended to `knowledge.jsonl`.
- Place outputs in the correct `RAG Files/` subfolders, or configure `scene_globs` if kept elsewhere.

## Inputs and outputs
- Input: a formatted `.docx` chapter (e.g., `Chapter 07.docx`).
- Outputs:
  - Scene files: `RAG Files/<scene_id>.json` (one file per scene)
  - Characters: `RAG Files/character/*.json` (one file per character)
  - Entities catalog: `RAG Files/entities/entities.json`
  - Entity-scene mapping: `RAG Files/mappings/entity_aliases.json` (and optional `manual_alias_overrides.json`)
  - Knowledge: `RAG Files/knowledgeFile/knowledge.jsonl`

## Recommended folder layout
- Keep your ETL code (parsers, segmenters, exporters) in a separate private repo or under `Beyond the Neon Veil/ETL-Chapter-Into-RAG-process/` (not required by the builder).
- Ensure the final artifacts land in `RAG Files/` (or add your custom location to `scene_globs` in `RAG Files/project.config.json`).

## Pipeline stages

1) Ingest the Word chapter (.docx)
- Use a reliable .docx text extractor (e.g., `python-docx`) to obtain plain text while preserving paragraph boundaries.
- Normalize whitespace, strip headers/footers/page numbers.
- Keep original chapter title and any canonical chapter numbering for metadata.

2) Scene segmentation
- Apply rules or markers to break the chapter into scenes. Common options:
  - Explicit scene breaks (***, ###, blank lines x2+). 
  - Heuristics: time/place shifts, POV changes.
- Assign a stable `scene_id` per scene. Recommended naming:
  - `<CC>-chapter<CC>-scene<CC>` (e.g., `16-chapter07-scene02`).
  - The builder uses the filename (without extension) as `scene_id`. Keep it unique and stable.
- Capture source metadata in each scene:
  - `source.book_title`: "Beyond the Neon Veil"
  - `source.chapter_title`: e.g., "Chapter 7 — Arc Disruptions"
  - `source.date_in_universe`: optional string/date
  - `doc_type`: e.g., `scene`

3) Character and entity extraction
- Run an entity extraction pass (e.g., spaCy NER + custom dictionaries) to detect:
  - Characters (canonical IDs), locations, organizations, items, technology, factions.
- Update catalogs:
  - Characters: one JSON per character at `RAG Files/character/<id>.json` with at least:
    ```json
    {
      "identity": { "id": "quincy", "display_name": "Quincy" }
    }
    ```
  - Entities: edit `RAG Files/entities/entities.json` and append to lists like `locations`, `organizations`, `items`, `technologies`, `factions`.
- Associate scenes to entities:
  - Add rules in `RAG Files/mappings/entity_aliases.json` mapping `rules[].context.scenes` to `rules[].entity_ids`.
  - Optionally use `RAG Files/mappings/manual_alias_overrides.json` for quick corrections.

4) Build scene JSONs (RAG-ready)
- Preferred: provide explicit `embedding_payloads` per scene — small, semantically coherent chunks with stable IDs.
- Minimum viable alternative: provide `text_windows.full_scene_clean` and (optionally) `text_windows.dense_recap_250w`; the builder can rechunk when needed.

Minimal scene JSON skeleton (preferred payloads):
```json
{
  "source": {
    "book_title": "Beyond the Neon Veil",
    "chapter_title": "Chapter 7 — Arc Disruptions",
    "date_in_universe": "2079-11-04"
  },
  "doc_type": "scene",
  "embedding_payloads": [
    {
      "chunk_id": "c001",
      "type": "scene",
      "text": "Cleaned scene text segment 1...",
      "meta": { "notes": "optional" }
    },
    {
      "chunk_id": "c002",
      "type": "entities_focus",
      "text": "A segment emphasizing important entities..."
    }
  ],
  "text_windows": {
    "full_scene_clean": "(optional) full cleaned scene text",
    "dense_recap_250w": "(optional) ~250-word recap"
  }
}
```
Notes:
- Keep `chunk_id` stable; it’s used during hydration when displaying contexts.
- `type` is free-form but commonly `scene`, `recap`, `entities_focus`.

5) Author knowledge Q/A (optional but recommended)
- For material that benefits retrieval, append Q/A pairs to `RAG Files/knowledgeFile/knowledge.jsonl`:
```jsonl
{"instruction": "What is the Three Shovels Caravan?", "output": "A nomadic convoy...", "metadata": {"source_chapter": "Chapter 6"}}
{"instruction": "Who recruits Quincy in Chapter 7?", "output": "...", "metadata": {"source_chapter": "Chapter 7"}}
```
- Set the path once in `RAG Files/project.config.json`:
```json
{ "knowledge_jsonl": "RAG Files/knowledgeFile/knowledge.jsonl" }
```

6) Place outputs and configure
- Scenes: put files at `RAG Files/<scene_id>.json`. If you keep scenes elsewhere (e.g., `Beyond the Neon Veil/NextBook/scenes/`), add to `scene_globs` in `RAG Files/project.config.json`:
```json
{
  "scene_globs": [
    "RAG Files/*.json",
    "Beyond the Neon Veil/NextBook/scenes/**/*.json"
  ]
}
```
- Characters: `RAG Files/character/*.json` (one file per character).
- Entities: update `RAG Files/entities/entities.json`.
- Mappings: `RAG Files/mappings/entity_aliases.json` (and `manual_alias_overrides.json`).
- Knowledge: append to `RAG Files/knowledgeFile/knowledge.jsonl` and ensure the config points to it.

7) Rebuild and validate (DevOps orchestrator)
- Fast iteration (limit first K chunks):
```bat
DevOps\run_orchestrator.cmd --limit 200 ^
  --out-dir "RAG Files\indexes\next_book_drop01" ^
  --query "What changed in the new chapter?" --k 5 --no-llm
```
- Full rebuild: omit `--limit`. The orchestrator reads `scene_globs` and `knowledge_jsonl` from `RAG Files/project.config.json`.
- Inspect outputs in your chosen `--out-dir`:
  - `embeddings.npy`, `metadata.jsonl`, `index_manifest.json`, `scene_entity_summary.json`.
  - `contexts_sample.json` lists the top-K with hydrated `text` (including `knowledge_qa` chunks when present).

## Implementation notes (for your ETL scripts)
- Cleaning: normalize quotes/dashes, strip editorial markup, remove scene headers not part of narrative.
- IDs: use lowercase, hyphen/underscore-safe IDs for characters/entities; keep `scene_id` and `chunk_id` stable across rebuilds.
- Safety: never overwrite catalogs without backups; treat additions as append-only where possible.
- Performance: small chunk payloads (50–200 words) embed well; prefer semantically coherent spans.
- Rechunk fallback: if you can’t generate `embedding_payloads`, you can rely on `--rechunk` with `--chunking sentence|paragraph` during builds, but explicit payloads give better control.

## Reference files
- Builder: `RAG Files/scripts/build_rag_index.py`
- Orchestrator: `DevOps/orchestrate_build_and_smoke.py`
- Config: `RAG Files/project.config.json` (keys: `scene_globs`, `knowledge_jsonl`, `manual_overrides`, `model`, `default_verbosity`, `default_creativity`, `default_index_dir`)
- Catalogs: `RAG Files/entities/entities.json`, `RAG Files/character/*.json`
- Mappings: `RAG Files/mappings/entity_aliases.json`, `RAG Files/mappings/manual_alias_overrides.json`
- Knowledge: `RAG Files/knowledgeFile/knowledge.jsonl`
