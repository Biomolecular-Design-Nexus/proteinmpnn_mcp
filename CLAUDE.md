# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

ProteinMPNN MCP is a Model Context Protocol (MCP) server that wraps the ProteinMPNN deep learning model for protein sequence design. It exposes 11 tools via FastMCP that allow AI assistants to design protein sequences, score them, and run constrained design — all through MCP.

## Quick Setup & Running

```bash
# Full setup (creates conda env, clones ProteinMPNN repo, installs deps)
bash quick_setup.sh

# Run the MCP server
env/bin/python src/server.py

# Register with Claude Code
claude mcp add proteinmpnn -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Docker build & run
docker build -t proteinmpnn-mcp .
docker run proteinmpnn-mcp
```

## Architecture

### Two-Layer Design

**MCP Server Layer** (`src/server.py`): Defines 11 `@mcp.tool()` endpoints using FastMCP. Each tool delegates to either a script function (sync) or the job manager (async). Returns `{"status": "success"|"error", ...}` dicts.

**Script Layer** (`scripts/*.py`): Each script has a `run_*()` core function that builds a subprocess command to invoke `repo/ProteinMPNN/protein_mpnn_run.py`, executes it, then parses the FASTA/NPZ output. Scripts are both importable (called by server) and CLI-executable via argparse.

### Tool Categories

| Category | Tools | Pattern |
|----------|-------|---------|
| Sync (<10 min) | simple_design, sequence_scoring, constrained_design, ca_only_design | Import `run_*()` from scripts, return result directly |
| Async (>10 min) | submit_batch_design, submit_large_design | Submit to `JobManager`, return `job_id` |
| Job Management | get_job_status, get_job_result, get_job_log, cancel_job, list_jobs | Query job metadata in `jobs/{job_id}/` |
| Utility | validate_pdb_structure, list_example_structures | Direct logic in server.py |

### Key Path Conventions

All paths are resolved relative to `MCP_ROOT` (parent of `src/`):
- **Model weights**: `examples/data/{vanilla,ca,soluble}_model_weights/*.pt` — used by `scripts/lib/utils.py:get_model_path()`
- **ProteinMPNN repo**: `repo/ProteinMPNN/` — used by `scripts/lib/utils.py:get_repo_path()`
- **Job storage**: `jobs/{job_id}/metadata.json` and `job.log`
- **Output structure**: Each run creates `{output_dir}/seqs/`, `scores/`, `logs/`

### Config Precedence

Each script has a `DEFAULT_CONFIG` dict. Overrides merge as: `DEFAULT_CONFIG` → JSON config file → keyword arguments/CLI args.

### How Scripts Invoke ProteinMPNN

Every design script follows this pattern:
1. Validate input PDB and config params
2. Resolve paths via `get_model_path()` and `get_repo_path()`
3. Build command: `[sys.executable, repo/protein_mpnn_run.py, --flags...]`
4. Execute: `subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)`
5. Parse FASTA output, extract scores from headers
6. Save `execution_metadata.json` alongside results

### Job Manager (`src/jobs/manager.py`)

Thread-based async execution. Each job gets a UUID, runs as a subprocess in a background thread, persists state as JSON in `jobs/{job_id}/`. Status flow: PENDING → RUNNING → COMPLETED/FAILED/CANCELLED.

## Adding a New Design Tool

1. Create `scripts/new_design.py` following the pattern in `scripts/simple_design.py` — define `DEFAULT_CONFIG`, a `run_new_design()` function, and a `main()` CLI
2. Use shared utilities from `scripts/lib/utils.py` and `scripts/lib/io.py`
3. Add sync tool in `src/server.py` with `@mcp.tool()` that imports and calls `run_new_design()`
4. If it may run >10 min, add a `submit_new_design()` async tool that uses `job_manager.submit_job()`

## Docker

The Dockerfile clones ProteinMPNN from GitHub at build time and bakes all model weights (~71MB) into the image. The GitHub Actions workflow (`.github/workflows/docker.yml`) builds and pushes to GHCR on push to main or version tags.
