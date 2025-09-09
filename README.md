# RAG Stoichiometry Retrieval Toolkit

Config-driven Retrieval-Augmented Generation (RAG) toolkit focused on high-quality retrieval mixing ("data stoichiometry"). It helps you build a local vector index from your documents and then query it with balanced coverage across multiple knowledge sources (e.g., scenes/chapters, character/persona dossiers, knowledge bases, wikis, specs, RFPs).

What you get when you clone this repo:
- A ready-to-run pipeline to embed your corpus, store vectors, and retrieve top-K contexts.
- A flexible, config-driven stratified retrieval engine to control how different knowledge types appear in the final context set.
- Windows-friendly one-click scripts, plus Python modules you can run on macOS/Linux with standard Python commands.
- A comprehensive manual and HOWTO to get you from zero to answers.

Key paths:
- Project root: `d:/Windsurf Projects/Create-RAG-Vector-Database/`
- Data root: `RAG Files/`
- Scripts: `RAG Files/scripts/`
- DevOps utilities: `DevOps/`

## Features

- Stratified retrieval (“data stoichiometry”):
  - Config-driven buckets with flexible matching rules (`doc_type`, `scene_id`, `chunk_type[_prefix]`, `path_glob`, `file`).
  - Per-bucket allocation by ratio or explicit counts with shortfall redistribution.
  - Interleaved ordering (default) or global similarity ordering.
  - Optional per-bucket similarity weights.
- Providers:
  - Local llama.cpp via HTTP.
  - OpenAI API (key via CLI/env/file/config; never hardcoded).
- Orchestrator and one-click Windows scripts.
- Context-only workflows and JSON exports for inspection.

## Quick Start

1) Create/refresh the virtual environment and install deps:
```bat
DevOps\setup_venv.cmd
```

2) Build an index (uses `RAG Files/project.config.json`):
```bat
DevOps\run_orchestrator.cmd --out-dir "RAG Files\indexes\with_chars_knowledge" --limit 200
```

3) Query the index (default stratified 3:2:1 with interleave):
```bat
DevOps\run_rag_query.cmd --index-dir "RAG Files\indexes\with_chars_knowledge" ^
  --query "Create a DALL·E prompt that accurately depicts Lesha’s physical appearance." ^
  --k 9 --verbosity verbose --max-tokens 0
```

- To override mixing at runtime:
```bat
--mix stratified --mix-ratio "scene:3,character:2,knowledge:1" --mix-order interleave
```
- To set strict counts per bucket:
```bat
--k-per-bucket "scene:5,character:3,knowledge:1"
```
- To nudge without quotas (flat mode):
```bat
--mix flat --mix-weights "scene:1.0,character:1.1,knowledge:0.9"
```

## Configuration

- Primary config: `RAG Files/project.config.json`
  - `stratification.defaults` (project-wide defaults):
    - `mix_mode`: `stratified`
    - `ratio`: `{ "scene": 3, "character": 2, "knowledge": 1 }`
    - `ordering`: `interleave`
    - `weights`: per-bucket similarity multipliers
  - `stratification.buckets`: array of bucket definitions with `label`, `priority`, and `match` rules.

## Documentation

- MANPAGE (full manual): `DevOps/MANPAGE.md`
- HOWTO (options and examples): `DevOps/HOWTO.md`
- Orchestrator overview: `DevOps/README.md`

## Security

- OpenAI API key sourcing precedence: CLI `--openai-api-key` > env `OPENAI_API_KEY` > `--openai-key-file` > config key.
- No API keys are stored in the repository. Use environment variables or a local key file.

## Notes

- Large data (scenes, character dossiers, knowledge JSONL, and generated indexes) are intentionally excluded from version control via `.gitignore`.
- For history cleanup of previously committed data, use a history-rewrite tool (e.g., git-filter-repo or BFG). Contact maintainers before proceeding.

---

## What this toolkit does (high level)

- Builds a local vector index from your source documents (JSON/Markdown/text) using a Sentence Transformers embedding model.
- Retrieves top-K contexts by cosine similarity for a user query.
- Optionally mixes contexts from different "buckets" (types of documents) to ensure well-rounded answers. This prevents the model from over-focusing on a single source.
- Sends the contexts and your question to an LLM (local llama.cpp or OpenAI) to generate an answer.

## Why it’s useful

- Ensures balanced coverage of diverse knowledge sources in your prompts ("data stoichiometry").
- Configurable per project, not hard-coded to a single domain.
- Works offline with llama.cpp, or online with OpenAI.
- Windows one-click scripts reduce friction; Python modules allow cross-platform use.

## Platforms and requirements

- Tested on Windows 10/11 with Python 3.10+.
- Also works on macOS/Linux if you use the Python modules directly (replace `.cmd` with equivalent `python` invocations).
- Required Python libraries (installed by `DevOps/setup_venv.cmd`):
  - `sentence-transformers`, `numpy`, `requests`
  - Optionally `transformers` or other ecosystem tools you may add later
- LLM providers:
  - Local llama.cpp server over HTTP (specify `--llama-url`), or
  - OpenAI API (provide key via CLI/env/file/config)

## Setup

1) Create a virtual environment and install dependencies:
```bat
DevOps\setup_venv.cmd
```
On macOS/Linux, the equivalent steps are:
```bash
python -m venv .venv
. .venv/bin/activate
pip install -U sentence-transformers numpy requests
```

2) Configure your project in `RAG Files/project.config.json`.
   - Set your `model` (defaults to `sentence-transformers/all-MiniLM-L6-v2`).
   - Configure `scene_globs` or corpus locations.
   - (Optional) Point `knowledge_jsonl` at a JSONL with Q/A items.
   - Adjust `stratification` buckets/ratios/ordering as needed.

## Preparing your data

The toolkit is content-agnostic. You can use:
- Narrative projects (scenes/chapters + character/persona dossiers)
- Enterprise docs (wikis, specs, RFCs, SOPs)
- Proposal/RFP workflows (RFP sections + product literature + policy + past performance)

Data placement options:
- Put your files under `RAG Files/` and point `scene_globs` to them (e.g., `RAG Files/*.json`, `RAG Files/tech/*.md`).
- Provide a knowledge Q/A JSONL via `knowledge_jsonl` in `project.config.json` for targeted factual retrieval.

Minimal knowledge JSONL row:
```json
{"instruction": "Q text", "output": "A text", "metadata": {"source": "optional"}}
```

## Build the index

Use the orchestrator (recommended on Windows):
```bat
DevOps\run_orchestrator.cmd --out-dir "RAG Files\indexes\my_index" --limit 200
```
Notes:
- `--limit 200` embeds only the first 200 chunks for a fast smoke test (omit for full build).
- The orchestrator imports `RAG Files/scripts/build_rag_index.py` and applies `project.config.json` plus CLI overrides.

## Retrieval/QA smoke test (contexts only)

After building, the orchestrator can run a no-LLM test that writes contexts to JSON:
```bat
DevOps\run_orchestrator.cmd --skip-build ^
  --index-dir "RAG Files\indexes\my_index" ^
  --query "What are the core constraints we must satisfy?" ^
  --k 6 --no-llm --contexts-out "RAG Files\indexes\my_index\contexts_sample.json"
```
This prints a ranked list (with bucket labels) and writes the full payload for inspection.

## Full QA generation

Use `run_rag_query.cmd` to generate an answer via llama.cpp (default) or OpenAI:
```bat
DevOps\run_rag_query.cmd --index-dir "RAG Files\indexes\my_index" ^
  --query "Summarize the key risks and mitigations" ^
  --k 8 --verbosity verbose --max-tokens 0 ^
  --answer-out "RAG Files\indexes\my_index\answer.txt"
```
Provider switches:
- llama.cpp: `--llama-url http://127.0.0.1:8080`
- OpenAI: `--provider openai --openai-model gpt-4o --openai-api-key ...`
  - Or pass `--openai-key-file path/to/key.txt` or set `OPENAI_API_KEY` in the environment.

## Stratified retrieval in practice

Project defaults live in `project.config.json` under `stratification`:
- `buckets`: Define labels, priority, and flexible match rules (`doc_type`, `scene_id`, `chunk_type`, `chunk_type_prefix`, `path_glob`, `file`).
- `defaults`: Set `mix_mode`, `ratio` (e.g., `{"scene":3,"character":2,"knowledge":1}`), `ordering` (e.g., `interleave`), and optional `weights`.

Per-run overrides:
```bat
--mix stratified --mix-ratio "scene:3,character:2,knowledge:1" --mix-order interleave
--k-per-bucket "scene:5,character:2,knowledge:1"
--mix flat --mix-weights "scene:1.0,tech:1.1,policy:0.9"
```

## Outputs and inspection

- `contexts-out`: Writes selected contexts (with `bucket` and `similarity`) to JSON for debugging/audit.
- `answer-out`: Writes the generated answer to a file.
- Index artifacts: `embeddings.npy`, `metadata.jsonl`, `index_manifest.json` (written to `--out-dir`).

## Troubleshooting

- Sentence Transformers missing:
  - Recreate venv via `DevOps\setup_venv.cmd`.
- llama.cpp connection errors:
  - Check `--llama-url` and ensure the server is running.
- OpenAI key issues:
  - Provide `--openai-api-key` or `--openai-key-file`, or set `OPENAI_API_KEY`.
- Unexpected retrieval distribution:
  - Inspect `project.config.json` `stratification` section or pass per-run overrides.

## Adapting to other domains

- Redefine `buckets` to match your sources (e.g., `rfp`, `product`, `policy`, `past_perf`).
- Create targeted knowledge JSONL with frequent Q/As.
- Use per-bucket ratios or explicit counts to steer coverage.

For a step-by-step setup and additional examples, see the HOWTO: `DevOps/HOWTO.md`. For deeper background and design details, see the MANPAGE: `DevOps/MANPAGE.md`.

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes. For pull requests:
- Keep changes focused and well-described.
- Include clear commit messages and rationale.
- Add/update documentation where relevant (README/HOWTO/MANPAGE).

## License

This project is licensed under the MIT License. See `LICENSE` for details.
