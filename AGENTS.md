# Repository Guidelines

This guide aligns contributors across the LiquidAI multi-agent stack (ETL, RAG, MCP tools, fine-tuning). Keep changes small and runnable.

## Project Structure & Module Organization
- `liquid-shared-core/`: shared configs, devices, schemas; install first.
- `liquid-etl-pipeline/`: Pydantic AI graphs for chunking/metadata/QA and FT dataset creation; reads from `data/raw/`, writes to `data/processed/` and `data/ft/`.
- `liquid-rag-runtime/`: FastAPI inference service and agents for retrieval/QA; depends on Chroma data in `data/vectordb/`.
- `liquid-mcp-tools/`: Model Context Protocol servers (filesystem, RAG helpers).
- `liquid-ft-trainer/`: LoRA/QLoRA fine-tuning utilities.
- `models/`: downloaded LiquidAI weights (git-ignored).
- `data/`: raw/processed/vectordb/ft outputs (git-ignored).

## Build, Test, and Development Commands
- `uv sync`: create .venv and install all local packages editable.
- `uv run bootstrap.py --model 700M`: quick env + small model download; omit flag for all models.
- `uv run liquid-etl`: run full ETL pipeline against `data/raw/`.
- `uv run liquid-rag-server --port 8000`: start RAG API; pairs with existing vectordb.
- `docker-compose --profile cpu up` | `--profile gpu up`: compose orchestration for CPU/GPU.
- `uv run ruff check .` / `uv run black .` / `uv run mypy .` / `uv run pytest`: lint/format/typecheck/tests.

## Coding Style & Naming Conventions
- Python 4-space indent, target 100 cols; ruff ignores E501 only when unavoidable.
- Type hints required; mypy is strict—fix warnings.
- Modules/files snake_case; classes PascalCase; functions/vars snake_case; tests `test_<behavior>.py`.
- Keep side effects behind `if __name__ == "__main__":`; CLI entrypoints live in package `__main__`.

## Testing Guidelines
- Use `pytest`; place tests beside packages (e.g., `liquid-etl-pipeline/tests/`), mirroring module paths.
- Name tests `test_<module>.py` with functions `test_<case>`; include happy path + failure/edge coverage.
- For async FastAPI handlers, use `pytest-asyncio`; for APIs, prefer httpx client.
- Regenerate small deterministic fixtures; never commit model weights or large corpora.

## Commit & Pull Request Guidelines
- Follow existing history: Conventional prefixes (`feat:`, `docs:`, `chore:`, `fix:`). Subject ≤72 chars; body for rationale and links.
- Each PR: purpose, scope, run/verify steps (`uv run ...` or curl), linked issue if any.
- Add screenshots or curl transcripts for API changes; sample payloads for agent updates.
- Rebase over main before merge; ensure `ruff`, `black`, `mypy`, and `pytest` are clean.

## Data, Models, and Secrets
- Large assets are git-ignored; keep downloads in `models/` and generated artifacts in `data/`.
- Do not commit `.env` or API keys; prefer environment variables and document required keys in PR description when adding new ones.
