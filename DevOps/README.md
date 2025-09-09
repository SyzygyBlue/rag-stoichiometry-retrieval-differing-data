# DevOps Orchestrator

This folder contains the Windows-friendly orchestrator to build the RAG index and run a retrieval/QA smoke test without PowerShell quirks.

- Script: `DevOps/orchestrate_build_and_smoke.py`
- One-click runner: `DevOps/run_orchestrator.cmd` (detects project venv at `RAG Files/.venv/` and passes through all arguments)

The orchestrator:
- Builds the index by importing `RAG Files/scripts/build_rag_index.py` and passing config/CLI options.
- Runs a retrieval/QA smoke test by importing `RAG Files/scripts/rag_query.py` utilities (encode, retrieve, prompt, llama call).
- Hydrates text for both scene chunks and knowledge chunks (from `metadata.extra`).

See `DevOps/HOWTO.md` for complete usage and all available options. For a comprehensive manual covering setup, config, providers, and retrieval mixing, see `DevOps/MANPAGE.md`.

## Quick start

Using the one-click runner (recommended):

```bat
DevOps\run_orchestrator.cmd --limit 200 ^
  --out-dir "D:\Windsurf Projects\Create-RAG-Vector-Database\RAG Files\indexes\with_knowledge_smoke" ^
  --query "What is the Three Shovels Caravan?" --k 5 --no-llm
```

Using Python directly (ensure you use the project venv Python):

```bat
"D:\Windsurf Projects\Create-RAG-Vector-Database\RAG Files\.venv\Scripts\python.exe" ^
  "DevOps\orchestrate_build_and_smoke.py" ^
  --limit 200 ^
  --out-dir "D:\Windsurf Projects\Create-RAG-Vector-Database\RAG Files\indexes\with_knowledge_smoke" ^
  --query "What is the Three Shovels Caravan?" --k 5 --no-llm
```

## Prerequisites

- Python 3.10+ with the project virtual environment at `RAG Files/.venv/`
- Required packages installed in that venv (e.g., sentence-transformers, numpy, requests)
- Input data present under `RAG Files/` (scenes, `mappings/`, `entities/`, `character/`, and optional `knowledgeFile/knowledge.jsonl`)

## Adding new chapters and characters (rolling updates for the next book)

Use this checklist whenever you deliver a new chapter or introduce new characters:

- **Add new scenes (chapters)**
  - Place compiled scene JSON files under `RAG Files/` so they match the default pattern `RAG Files/*.json` used by `build_rag_index.py`.
  - If you keep new scenes elsewhere, add globs to `RAG Files/project.config.json` under `scene_globs`, for example:
    ```json
    {
      "scene_globs": [
        "RAG Files/*.json",
        "NextBook/scenes/**/*.json"
      ]
    }
    ```
  - Each scene should include either `embedding_payloads` (preferred) or `text_windows.full_scene_clean`/`text_windows.dense_recap_250w` so it can be chunked/embedded.

- **Add or update characters**
  - Create one JSON per character under `RAG Files/character/*.json`. Minimal shape:
    ```json
    {
      "identity": {
        "id": "new_character_id",
        "display_name": "New Character"
      }
    }
    ```
  - Characters are auto-discovered by `build_rag_index.py` and used to label entities in retrieval results.

- **Update entities catalog (items/locations/orgs/etc.)**
  - Edit `RAG Files/entities/entities.json` and add entries to the appropriate lists (e.g., `items`, `locations`, `organizations`, `factions`, `technologies`).

- **Map scenes to entities (optional but recommended)**
  - Add rules to `RAG Files/mappings/entity_aliases.json` (`rules[].context.scenes` -> `rules[].entity_ids`) to associate scenes with canonical entity IDs.
  - For quick fixes, you can also use `RAG Files/mappings/manual_alias_overrides.json` and pass it via `--manual-overrides` (or set in `project.config.json`).

- **Append knowledge Q/A (optional)**
  - Append lines to `RAG Files/knowledgeFile/knowledge.jsonl` with the shape:
    ```json
    {"instruction": "Q text", "output": "A text", "metadata": {"source_chapter": "Chapter 12"}}
    ```
  - Set the knowledge path once in `RAG Files/project.config.json` so builds automatically ingest it:
    ```json
    {"knowledge_jsonl": "RAG Files/knowledgeFile/knowledge.jsonl"}
    ```

- **Rebuild the index and smoke test**
  - Fast iteration (limit first K chunks):
    ```bat
    DevOps\run_orchestrator.cmd --limit 200 ^
      --out-dir "RAG Files\indexes\next_book_drop01" ^
      --query "What changed in the new chapter?" --k 5 --no-llm
    ```
  - Full rebuild (omit `--limit`). If your scenes are stored outside `RAG Files/`, configure `scene_globs` in `RAG Files/project.config.json` so the orchestrator picks them up. If you run `build_rag_index.py` directly, you can also pass one or more `--scene-glob` values on the CLI.

- **Validate the update**
  - Check `index_manifest.json` and `metadata.jsonl` in the chosen `--out-dir` for increased counts and correct file references.
  - Inspect `scene_entity_summary.json` to ensure new scenes show expected characters/entities.
  - Review `contexts_sample.json` (written by the smoke test) to see top-K, including any `knowledge_qa` chunks if you appended knowledge.

Notes:
- Both `build_rag_index.py` and `rag_query.py` accept `--config` and merge with CLI; CLI overrides config.
- If your new scenes lack `embedding_payloads`, consider `--rechunk` with `--chunking {sentence|paragraph}` for quick chunking.

## Help

All options are documented by the script itself:

```bat
DevOps\run_orchestrator.cmd --help
```

or

```bat
"D:\Windsurf Projects\Create-RAG-Vector-Database\RAG Files\.venv\Scripts\python.exe" DevOps\orchestrate_build_and_smoke.py --help
```

For complete explanations and examples, see `DevOps/HOWTO.md`.
