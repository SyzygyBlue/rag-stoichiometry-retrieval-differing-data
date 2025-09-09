# RAG Toolkit Manual (MANPAGE)

This manual documents the end-to-end workflow and knobs for building and querying a local RAG index for projects ranging from novels to technical RFPs. It covers setup, building, querying, model providers, API key sourcing, context handling, and retrieval mixing strategies (flat and planned stratified retrieval).

- Project root: `d:/Windsurf Projects/Create-RAG-Vector-Database/`
- Data root: `RAG Files/`
- DevOps scripts: `DevOps/`

## 1) Quick Start

- Build an index (uses `RAG Files/project.config.json`):
```powershell
DevOps\run_orchestrator.cmd --out-dir "RAG Files\indexes\with_chars_knowledge" --limit 200
```

- Query the index (local llama.cpp):
```powershell
DevOps\run_rag_query.cmd --index-dir "RAG Files\indexes\with_chars_knowledge" --query "What is the Three Shovels Caravan?" --k 5
```

- Query the index (OpenAI provider):
```powershell
DevOps\run_rag_query.cmd `
  --provider openai `
  --openai-model gpt-4o `
  --openai-key-file "D:\Windsurf Projects\Node-Red-AI\node-red sombrero key (openai).txt" `
  --index-dir "RAG Files\indexes\with_chars_knowledge" `
  --query "Summarize Demmy after Fox River." `
  --k 5 `
  --verbosity concise `
  --max-tokens 0
```

## 2) Environment Setup

- Create the project venv and install packages:
```powershell
DevOps\setup_venv.cmd
```
- One-click runners:
  - Build + smoke: `DevOps/run_orchestrator.cmd`
  - Query only: `DevOps/run_rag_query.cmd`

## 3) Data Layout and Config

- Scenes: `RAG Files/*.json` (can extend via `scene_globs` in config)
- Characters: `RAG Files/character/*.json`
- Entities: `RAG Files/entities/entities.json`
- Alias mapping: `RAG Files/mappings/entity_aliases.json`
- Knowledge Q/A: `RAG Files/knowledgeFile/knowledge.jsonl`
- Config file: `RAG Files/project.config.json`

Example keys (abbreviated):
```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "llama_url": "http://127.0.0.1:8080",
  "default_index_dir": "RAG Files/indexes",
  "scene_globs": ["RAG Files/*.json", "RAG Files/character/*.json"],
  "knowledge_jsonl": "RAG Files/knowledgeFile/knowledge.jsonl",
  "default_verbosity": "concise",
  "default_creativity": 0.1
}
```

## 4) Building the Index

- Orchestrator (recommended):
```powershell
DevOps\run_orchestrator.cmd --out-dir "RAG Files\indexes\with_chars_knowledge" --limit 0
```
- Direct builder:
```powershell
"RAG Files\.venv\Scripts\python.exe" "RAG Files\scripts\build_rag_index.py" --out-dir "RAG Files\indexes\with_chars_knowledge"
```
Outputs:
- `embeddings.npy` (float32 [N,D])
- `metadata.jsonl` (one JSON per embedded chunk)
- `index_manifest.json` (summary)
- `scene_entity_summary.json` (mapping scene_id -> entities)

## 5) Querying the Index

CLI (via `DevOps/run_rag_query.cmd`):
- Core flags:
  - `--query`, `--k`, `--index-dir`, `--model` (embed), `--config`
- Providers:
  - `--provider {llama, openai}`
  - llama: `--llama-url`
  - openai: `--openai-model`, `--openai-base`, `--openai-api-key` or `--openai-key-file` or env `OPENAI_API_KEY`
- Output controls:
  - `--no-llm` (contexts only)
  - `--contexts-out`, `--answer-out`
  - `--append-context <path>` (repeatable; appends file content as an extra CONTEXT)
- Generation controls:
  - `--max-tokens` (0 = unlimited; applies per provider semantics)
  - `--temperature`
  - `--verbosity {concise, verbose, voluminous}`
  - `--creativity [0..1]`

Behavior:
- Retrieval forms a prompt with top‑K contexts and your question, then calls the selected provider.
- Contexts are always exported when `--contexts-out` is provided (even with LLM on).

## 6) Providers

- Local llama.cpp
  - Chat endpoint: `/v1/chat/completions` (OpenAI compatible)
  - Completion endpoint fallback: `/completion` (`n_predict = -1` when unlimited)
- OpenAI
  - Chat endpoint: `https://api.openai.com/v1/chat/completions`
  - API key sourcing precedence: CLI `--openai-api-key` > `OPENAI_API_KEY` env > `--openai-key-file` > config
  - Unlimited (`--max-tokens 0`) omits `max_tokens` and relies on provider limits

## 7) Retrieval Mixing (“Data Stoichiometry”)

Two complementary strategies to control which types of knowledge appear in the top‑K:

- Flat (current default):
  - Single top‑K by cosine similarity over the unified embedding matrix.
  - `--k 6` means 6 total across all sources.

- Planned: Config‑Driven Stratified Retrieval (coverage guarantees)
  - Define buckets in `project.config.json` with flexible matchers (doc_type, path globs, scene_id, chunk_type, etc.).
  - Allocate per‑bucket K via ratios or explicit counts and redistribute shortfalls.
  - Optional per‑bucket weights to nudge scoring.

Proposed config schema:
```json
{
  "stratification": {
    "buckets": [
      { "label": "scene", "priority": 10, "match": { "doc_type": ["neon_titan_scene_v1"] } },
      { "label": "character", "priority": 20, "match": { "doc_type": ["neon_titan_character_v1"], "path_glob": ["**/character/*.json"] } },
      { "label": "knowledge", "priority": 30, "match": { "doc_type": ["knowledge_qa"], "scene_id": ["knowledge", "knowledge_qa"], "chunk_type_prefix": ["knowledge"] } }
    ],
    "defaults": {
      "mix_mode": "stratified",                  
      "ratio": { "scene": 4, "character": 1, "knowledge": 1 },
      "weights": { "scene": 1.0, "character": 1.0, "knowledge": 1.0 },
      "ordering": "by_similarity"                
    }
  }
}
```

Proposed CLI additions:
- `--mix {flat,stratified}`
- `--mix-ratio "scene:4,character:1,knowledge:1"`
- `--k-per-bucket "scene:4,character:1,knowledge:1"` (sum forms total)
- `--mix-weights "scene:1.0,character:1.05,knowledge:0.85"`
- `--mix-order {by_similarity,interleave}`

Allocation & merging:
- Compute similarities once, bucket by match rules, allocate per ratio (largest remainder) or exact counts.
- If a bucket runs short, redistribute to other buckets by similarity.
- Merge final list either fully sorted by similarity or interleaved for diversity.

Examples:
```powershell
# Ensure coverage for k=6 with a 4:1:1 mix
devops\run_rag_query.cmd --index-dir "RAG Files\indexes\with_chars_knowledge" `
  --mix stratified --mix-ratio "scene:4,character:1,knowledge:1" --k 6 `
  --query "Create a DALL·E prompt of Lesha as per dossier." --verbosity verbose --max-tokens 0

# Exact per-bucket counts (total 8):
--k-per-bucket "scene:5,character:2,knowledge:1"

# Nudge knowledge down without strict quotas:
--mix flat --mix-weights "scene:1.0,character:1.05,knowledge:0.85" --k 8
```

## 8) Prompt Controls

- `--verbosity` tunes instruction detail and target length.
- `--creativity` governs strictness vs. invention; higher values allow labeled speculation.
- The prompt always includes a reference section encouraging citation of `scene_id`.

## 9) Appending Extra Context

- Add ad‑hoc files to the prompt with `--append-context <path>` (repeatable).
- Large files are trimmed defensively (e.g., ~16k chars) to protect prompt budget.

## 10) Outputs and Logging

- Contexts JSON: `--contexts-out path.json` (always written when provided)
- Answer text: `--answer-out path.txt`
- Console shows question, top‑K summary, and generated answer.

## 11) Troubleshooting

- No module named `sentence_transformers`:
  - Run `DevOps\setup_venv.cmd` to recreate venv and install packages.
- OpenAI 401 or missing key:
  - Prefer `--openai-key-file` or set `$env:OPENAI_API_KEY`.
- Long answers cut off:
  - Use `--max-tokens 0` (provider-limited) or a concrete number (e.g., 1200).
- `--append-context` not applied:
  - Check the flag spelling; ensure the file path is quoted when it contains spaces.

## 12) Roadmap

- Implement config‑driven stratified retrieval with CLI parity in orchestrator.
- Optional MMR/xQuAD re‑ranking for aspect diversification.
- Streaming output (`--stream`) with progressive printing and final write to file.
- FAISS export (already available via `RAG Files/scripts/export_faiss.py`).

## 13) Glossary

- Bucket: A labeled category of knowledge (e.g., scene/character/knowledge), defined by match rules.
- Mix ratio: Desired proportions per bucket when selecting top‑K contexts.
- Weights: Multipliers applied to similarity per bucket to nudge selection.
- Interleave: Round‑robin merge to keep visible diversity in the final ordered contexts.

---

For questions or feature requests, see `DevOps/README.md` and `DevOps/HOWTO.md`, or open an issue in your project tracker.
