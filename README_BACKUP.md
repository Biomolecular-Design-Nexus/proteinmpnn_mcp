# ProteinMPNN MCP

> An MCP (Model Context Protocol) server providing both synchronous and asynchronous tools for protein sequence design using ProteinMPNN

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

ProteinMPNN MCP provides scaffold-based protein sequence generation and likelihood calculation tools through a unified interface. This server wraps the powerful ProteinMPNN neural network model for AI-powered protein design, enabling both fast single-structure design and large-scale batch processing.

### Features
- **Fast sequence design** - Generate protein sequences for given backbone structures in ~10 seconds
- **Sequence scoring** - Calculate ProteinMPNN likelihood scores for sequence-structure compatibility
- **Constrained design** - Design with fixed positions to preserve functional sites
- **Batch processing** - High-throughput design for multiple structures
- **CA-only design** - Sequence generation from carbon alpha traces
- **Job management** - Asynchronous processing with real-time monitoring

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   └── server.py           # MCP server
├── scripts/
│   ├── simple_design.py      # Basic sequence generation
│   ├── sequence_scoring.py   # Likelihood calculation
│   ├── batch_design.py       # Batch processing
│   ├── constrained_design.py # Design with fixed positions
│   ├── ca_only_design.py     # CA-only sequence design
│   └── lib/                  # Shared utilities
├── examples/
│   └── data/               # Demo data
│       ├── inputs/         # Sample PDB structures
│       ├── vanilla_model_weights/ # Standard model weights
│       ├── ca_model_weights/      # CA-only model weights
│       └── soluble_model_weights/ # Soluble protein models
├── configs/                # Configuration files
├── jobs/                   # Job execution directory
└── repo/                   # Original ProteinMPNN repository
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- Git (for cloning ProteinMPNN repository)

### Step 1: Create Environment

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpnn_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env
```

### Step 2: Install Dependencies

```bash
# Install MCP dependencies
pip install fastmcp loguru

# Install ProteinMPNN dependencies (if needed)
pip install torch numpy biopython
```

### Step 3: Verify Installation

```bash
# Test imports
python -c "from src.server import mcp; print(f'Found {len(list(mcp.list_tools()))} tools')"
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Runtime | Example |
|--------|-------------|---------|---------|
| `scripts/simple_design.py` | Generate protein sequences for a given PDB structure | ~10s | See below |
| `scripts/sequence_scoring.py` | Score sequences using ProteinMPNN likelihood | ~8s | See below |
| `scripts/constrained_design.py` | Design with fixed positions | ~15s | See below |
| `scripts/ca_only_design.py` | Design using only CA atoms | ~12s | See below |
| `scripts/batch_design.py` | Process multiple PDB files | Variable | See below |

### Script Examples

#### Simple Design

```bash
# Activate environment
mamba activate ./env

# Basic sequence design
python scripts/simple_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/simple_design \
  --chains "A B" \
  --num_sequences 3
```

**Parameters:**
- `--input, -i`: Path to input PDB file (required)
- `--output, -o`: Output directory (default: results/)
- `--chains`: Space-separated chain IDs to design (default: auto-detect)
- `--num_sequences`: Number of sequences to generate (default: 3)
- `--temperature`: Sampling temperature (default: 0.1)
- `--config, -c`: JSON configuration file (optional)

#### Sequence Scoring

```bash
python scripts/sequence_scoring.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/scoring \
  --chains "A B" \
  --save_probs
```

**Parameters:**
- `--input, -i`: Path to input PDB structure file (required)
- `--output, -o`: Output directory (default: results/)
- `--fasta_sequences`: Custom sequences to score (format: "SEQ1/SEQ2")
- `--save_probs`: Save per-residue probabilities (flag)
- `--chains`: Chains to score (default: auto-detect)

#### Constrained Design

```bash
python scripts/constrained_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/constrained \
  --chains_to_design "A C" \
  --fixed_positions "1 2 3 4 5, 10 11 12"
```

**Parameters:**
- `--input, -i`: Path to input PDB file (required)
- `--chains_to_design`: Chains to design (required)
- `--fixed_positions`: Fixed positions per chain, comma-separated (required)
- `--num_sequences`: Number of sequences to generate (default: 3)

#### CA-Only Design

```bash
python scripts/ca_only_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/ca_only \
  --chains "A B" \
  --model v_48_020
```

#### Batch Processing

```bash
python scripts/batch_design.py \
  --input_dir examples/data/inputs/PDB_monomers/pdbs \
  --output results/batch \
  --num_sequences 2
```

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name ProteinMPNN
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add ProteinMPNN -- /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpnn_mcp/env/bin/python /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpn_mcp/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "ProteinMPNN": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpn_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpn_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What ProteinMPNN tools are available?
```

#### Basic Sequence Design
```
Use simple_design with input file @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb to generate 5 sequences for chains A and B
```

#### Sequence Scoring
```
Score sequences in @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb and save probabilities
```

#### Constrained Design
```
Run constrained_design on @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb with fixed positions "1 2 3" for chain A and "10 11 12" for chain C
```

#### Long-Running Tasks (Submit API)
```
Submit large_design for @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb with 50 sequences, then check the job status
```

#### Batch Processing
```
Submit batch processing for all PDB files in @examples/data/inputs/PDB_monomers/pdbs/
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb` | Reference a specific PDB file |
| `@configs/simple_design_config.json` | Reference a config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "ProteinMPNN": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpn_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpn_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use simple_design with file examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `simple_design` | Generate protein sequences for a PDB structure | `input_file`, `chains`, `num_sequences`, `temperature` |
| `sequence_scoring` | Score sequences using ProteinMPNN likelihood | `input_file`, `fasta_sequences`, `save_probs` |
| `constrained_design` | Design with constrained/fixed positions | `input_file`, `chains_to_design`, `fixed_positions` |
| `ca_only_design` | Design using only carbon alpha atoms | `input_file`, `chains`, `model`, `num_sequences` |
| `validate_pdb_structure` | Validate PDB structure compatibility | `input_file` |
| `list_example_structures` | List available demo structures | None |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_large_design` | Large-scale sequence design (background) | `input_file`, `num_sequences`, `chains`, `job_name` |
| `submit_batch_design` | Batch process multiple PDB files | `input_dir`, `file_pattern`, `chains`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

---

## Examples

### Example 1: Single Structure Design

**Goal:** Generate 5 diverse sequences for a protein complex

**Using Script:**
```bash
python scripts/simple_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/example1/ \
  --chains "A B" \
  --num_sequences 5 \
  --temperature 0.15
```

**Using MCP (in Claude Code):**
```
Use simple_design to process @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb and generate 5 sequences for chains A and B with temperature 0.15
```

**Expected Output:**
- FASTA files with 5 generated sequences
- Sequence metadata with ProteinMPNN scores
- Execution summary with timing information

### Example 2: Sequence Validation

**Goal:** Score existing sequences against a protein structure

**Using Script:**
```bash
python scripts/sequence_scoring.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/example2/ \
  --save_probs \
  --chains "A B"
```

**Using MCP (in Claude Code):**
```
Run sequence_scoring on @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb with save_probs=true for chains A and B
```

**Expected Output:**
- NPZ files with likelihood scores
- Per-residue probability arrays
- Sequence compatibility analysis

### Example 3: Batch Processing

**Goal:** Process multiple protein structures at once

**Using Script:**
```bash
python scripts/batch_design.py \
  --input_dir examples/data/inputs/PDB_monomers/pdbs \
  --output results/example3/ \
  --num_sequences 3
```

**Using MCP (in Claude Code):**
```
Submit batch_design for all PDB files in @examples/data/inputs/PDB_monomers/pdbs/ with 3 sequences per structure
```

**Expected Output:**
- FASTA files for each input structure
- JSONL file with parsed PDB data
- Batch processing summary

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `inputs/PDB_complexes/pdbs/3HTN.pdb` | Multi-chain protein complex | All design tools |
| `inputs/PDB_complexes/pdbs/4YOW.pdb` | Alternative protein complex | All design tools |
| `inputs/PDB_monomers/pdbs/6MRR.pdb` | Single chain protein | Simple design, CA-only |
| `inputs/PDB_monomers/pdbs/5L33.pdb` | Single chain protein | Simple design, scoring |
| `inputs/PDB_homooligomers/pdbs/6EHB.pdb` | Homooligomer structure | Constrained design |
| `inputs/PDB_homooligomers/pdbs/4GYT.pdb` | Homooligomer structure | Constrained design |

**Model Weights:**
- `vanilla_model_weights/` - Standard ProteinMPNN models for full-atom design
- `ca_model_weights/` - CA-only models for backbone-only design
- `soluble_model_weights/` - Soluble protein-specific models

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `simple_design_config.json` | Basic sequence design settings | chains, num_sequences, temperature, model |
| `sequence_scoring_config.json` | Scoring configuration | chains, fasta_sequences, save_probs |
| `batch_design_config.json` | Batch processing settings | file_pattern, num_sequences, chains |
| `constrained_design_config.json` | Constrained design parameters | chains_to_design, fixed_positions |
| `ca_only_design_config.json` | CA-only model settings | chains, model, num_sequences |
| `default_config.json` | Global default parameters | Common settings for all tools |

### Config Example

```json
{
  "chains": "A B",
  "num_sequences": 5,
  "temperature": 0.1,
  "model": "v_48_020",
  "use_soluble": false
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install fastmcp loguru torch numpy biopython
```

**Problem:** Import errors
```bash
# Verify installation
python -c "from src.server import mcp; print('Server loaded successfully')"
```

**Problem:** Model weights missing
```bash
# Verify model weights exist
ls -la examples/data/vanilla_model_weights/
ls -la examples/data/ca_model_weights/
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove ProteinMPNN
claude mcp add ProteinMPNN -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
python src/server.py
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job log for specific job
cat jobs/<job_id>/job.log
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

### Script Issues

**Problem:** ProteinMPNN script not found
```bash
# Verify repository structure
ls -la repo/ProteinMPNN/protein_mpnn_run.py
```

**Problem:** Permission denied
```bash
# Fix script permissions
chmod +x scripts/*.py
```

### Common Error Solutions

**Error:** "CUDA out of memory"
- Reduce `num_sequences` parameter
- Use smaller batch sizes
- Switch to CPU mode if available

**Error:** "Invalid chain ID"
- Check PDB file for available chains
- Use `validate_pdb_structure` tool to inspect structure

**Error:** "Model weights not found"
- Ensure model weights are copied to `examples/data/`
- Check file permissions on weight directories

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test scripts directly
python scripts/simple_design.py --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb --output test_results/

# Test MCP server
python src/server.py
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py
```

### Adding New Tools

1. Create script in `scripts/` following existing patterns
2. Add configuration file in `configs/`
3. Add tool wrapper in `src/server.py`
4. Test functionality and update documentation

---

## Performance Notes

### Typical Runtimes
- **Simple design (1-3 sequences):** ~10 seconds
- **Sequence scoring:** ~8 seconds
- **Constrained design:** ~15 seconds
- **CA-only design:** ~12 seconds
- **Batch processing:** Variable (depends on number of files)

### Memory Requirements
- **Minimum:** 4GB RAM
- **Recommended:** 8GB+ RAM for batch processing
- **GPU:** Optional but significantly faster with CUDA

### Optimization Tips
- Use smaller `num_sequences` for faster results
- Batch multiple small proteins together
- Use submit API for large jobs to avoid timeouts
- Monitor job logs for performance issues

---

## License

Based on [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) by Dauparas et al.

Original ProteinMPNN paper:
> Dauparas et al. "Robust deep learning–based protein sequence design using ProteinMPNN." Science 378.6615 (2022): 49-56.

## Credits

- **ProteinMPNN:** Original implementation by Justas Dauparas and team
- **MCP Integration:** Adapted for Model Context Protocol
- **FastMCP:** Built using FastMCP framework