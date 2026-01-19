# ProteinMPNN MCP

> Model Context Protocol (MCP) server providing access to ProteinMPNN protein design capabilities through Claude Code and other MCP-compatible clients.

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

ProteinMPNN MCP provides a unified interface to ProteinMPNN's protein design capabilities through the Model Context Protocol. This server enables AI assistants like Claude to design protein sequences, score mutations, and perform constrained design operations directly through natural language interactions.

### Features
- **11 MCP tools** covering all major ProteinMPNN use cases
- **Dual API design**: Sync (fast) and async (long-running) operations
- **Job management**: Full background processing with status tracking
- **Multiple model types**: Standard, CA-only, and soluble protein models
- **Batch processing**: Handle multiple structures simultaneously
- **Flexible constraints**: Fixed positions, chain-specific design, custom scoring
- **Comprehensive validation**: PDB structure checking and example data discovery

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment (Python 3.10.19)
├── src/
│   ├── server.py           # Main MCP server (11 tools)
│   └── jobs/               # Background job management
├── scripts/
│   ├── simple_design.py    # Basic sequence design
│   ├── sequence_scoring.py # Sequence evaluation
│   ├── constrained_design.py # Position-constrained design
│   ├── ca_only_design.py   # Backbone-only design
│   ├── batch_design.py     # Multi-file processing
│   └── lib/                # Shared utilities
├── examples/
│   └── data/               # Demo data and model weights
│       ├── inputs/         # Sample PDB structures
│       ├── vanilla_model_weights/ # Standard models
│       ├── ca_model_weights/     # CA-only models
│       └── soluble_model_weights/ # Soluble protein models
├── configs/                # Configuration templates
├── tests/                  # Integration tests and prompts
└── repo/                   # Original ProteinMPNN repository
```

---

## Installation

### Quick Setup (Recommended)

Run the automated setup script:

```bash
cd proteinmpnn_mcp
bash quick_setup.sh
```

The script will create the conda environment, clone the ProteinMPNN repository, install all dependencies, and display the Claude Code configuration. See `quick_setup.sh --help` for options like `--skip-env` or `--skip-repo`.

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- CUDA-compatible GPU (recommended for performance)

### Manual Installation (Alternative)

If you prefer manual installation or need to customize the setup, follow `reports/environment_setup.md`:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpnn_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install PyTorch and dependencies
mamba run -p ./env pip install torch torchvision numpy
# or: conda run -p ./env pip install torch torchvision numpy

# Install MCP dependencies
mamba run -p ./env pip install loguru click pandas tqdm fastmcp
# or: conda run -p ./env pip install loguru click pandas tqdm fastmcp
```

### Verify Installation
```bash
# Test environment
mamba run -p ./env python -c "import torch; import numpy; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Test ProteinMPNN import
mamba run -p ./env python -c "import sys; sys.path.append('repo/ProteinMPNN'); import protein_mpnn_utils; print('ProteinMPNN import successful')"

# Test MCP server
mamba run -p ./env python -c "from src.server import mcp; print('MCP Server OK')"
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/simple_design.py` | Basic sequence design for PDB structures | See below |
| `scripts/sequence_scoring.py` | Score sequences using ProteinMPNN likelihood | See below |
| `scripts/constrained_design.py` | Design with fixed position constraints | See below |
| `scripts/ca_only_design.py` | Design using carbon alpha coordinates only | See below |
| `scripts/batch_design.py` | Batch process multiple PDB files | See below |

### Script Examples

#### Simple Design
Generate protein sequences for a PDB structure (completes in ~10 seconds):

```bash
# Activate environment
mamba activate ./env

# Basic usage
python scripts/simple_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/simple_design \
  --chains "A B" \
  --num_sequences 5 \
  --temperature 0.1

# With config file
python scripts/simple_design.py \
  --config configs/simple_design_config.json \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb
```

**Parameters:**
- `--input, -i`: Path to input PDB file (required)
- `--output, -o`: Output directory (default: results/simple_design)
- `--chains`: Space-separated chain IDs to design (e.g., "A B")
- `--num_sequences`: Number of sequences to generate (default: 3)
- `--temperature`: Sampling temperature (default: 0.1, lower = less diverse)
- `--config, -c`: JSON configuration file (optional)

#### Sequence Scoring
Score protein sequences using ProteinMPNN likelihood (completes in ~8 seconds):

```bash
# Score native sequence
python scripts/sequence_scoring.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/scoring \
  --chains "A B" \
  --save_probs

# Score custom sequences
python scripts/sequence_scoring.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/scoring \
  --fasta_sequences "MKTAYIAK.../DVFSLREM..."
```

#### Constrained Design
Design with fixed position constraints (completes in ~15 seconds):

```bash
python scripts/constrained_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/constrained \
  --chains_to_design "A B" \
  --fixed_positions "1 2 3 25 26, 10 11 12 15" \
  --num_sequences 3
```

#### CA-Only Design
Design using backbone coordinates only (completes in ~12 seconds):

```bash
python scripts/ca_only_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/ca_only \
  --chains "A B" \
  --model "v_48_020" \
  --num_sequences 3
```

#### Batch Design
Process multiple PDB files (time varies with number of files):

```bash
python scripts/batch_design.py \
  --input_dir examples/data/inputs/PDB_monomers/pdbs \
  --output results/batch \
  --file_pattern "*.pdb" \
  --num_sequences 2
```

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
mamba activate ./env
fastmcp install src/server.py --name ProteinMPNN
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add ProteinMPNN -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
# Should show: ProteinMPNN: ... - ✓ Connected
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "ProteinMPNN": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpnn_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpnn_mcp/src/server.py"]
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
Use simple_design to generate 5 sequences for @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb with chains A and B
```

#### Sequence Scoring
```
Score the native sequences in @examples/data/inputs/PDB_monomers/pdbs/5L33.pdb and save probabilities
```

#### Constrained Design
```
Design sequences for @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb with fixed positions 1,2,3 in chain A and 10,11,12 in chain B
```

#### Long-Running Tasks (Submit API)
```
Submit a large design job for @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb generating 50 sequences, then check the status
```

#### Batch Processing
```
Submit batch processing for all PDB files in @examples/data/inputs/PDB_monomers/pdbs/ generating 3 sequences each
```

#### Job Management
```
List all my ProteinMPNN jobs and show the status of the most recent one
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb` | Reference a specific PDB file |
| `@examples/data/inputs/PDB_monomers/pdbs/` | Reference a directory of PDB files |
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
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpnn_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpnn_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What ProteinMPNN tools are available?
> Generate protein sequences for examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb
> Score sequences in examples/data/inputs/PDB_monomers/pdbs/5L33.pdb
```

---

## Available Tools

### Synchronous Tools (Immediate Response)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters | Typical Runtime |
|------|-------------|------------|-----------------|
| `simple_design` | Basic sequence design | `input_file`, `chains`, `num_sequences`, `temperature`, `output_dir` | ~10 sec |
| `sequence_scoring` | Sequence likelihood scoring | `input_file`, `fasta_sequences`, `save_probs`, `output_dir` | ~8 sec |
| `constrained_design` | Position-constrained design | `input_file`, `chains_to_design`, `fixed_positions`, `num_sequences`, `output_dir` | ~15 sec |
| `ca_only_design` | Backbone-only design | `input_file`, `chains`, `model`, `num_sequences`, `output_dir` | ~12 sec |
| `validate_pdb_structure` | PDB file validation | `input_file` | <1 sec |
| `list_example_structures` | Find demo data | None | <1 sec |

### Asynchronous Tools (Submit API)

These tools return a job_id for tracking (> 10 minutes or many sequences):

| Tool | Description | Parameters | Use Case |
|------|-------------|------------|----------|
| `submit_batch_design` | Multi-file processing | `input_dir`, `file_pattern`, `chains`, `num_sequences`, `output_dir`, `job_name` | Process many PDB files |
| `submit_large_design` | Large sequence generation | `input_file`, `chains`, `num_sequences` (>10), `temperature`, `output_dir`, `job_name` | Generate many sequences |

### Job Management Tools

| Tool | Description | Usage |
|------|-------------|--------|
| `get_job_status` | Check job progress | `get_job_status(job_id="abc123")` |
| `get_job_result` | Get results when completed | `get_job_result(job_id="abc123")` |
| `get_job_log` | View execution logs | `get_job_log(job_id="abc123", tail=50)` |
| `cancel_job` | Cancel running job | `cancel_job(job_id="abc123")` |
| `list_jobs` | List all jobs | `list_jobs(status="running")` |

---

## Examples

### Example 1: Basic Protein Design

**Goal:** Generate diverse sequences for a protein complex

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
Design 5 diverse protein sequences for @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb with chains A and B using temperature 0.15
```

**Expected Output:**
- 5 FASTA sequences with ProteinMPNN scores
- Execution metadata with parameters and timing
- Score data in NPZ format

### Example 2: Sequence Evaluation

**Goal:** Score and analyze existing protein sequences

**Using Script:**
```bash
python scripts/sequence_scoring.py \
  --input examples/data/inputs/PDB_monomers/pdbs/5L33.pdb \
  --output results/example2/ \
  --save_probs
```

**Using MCP (in Claude Code):**
```
Score the native sequence in @examples/data/inputs/PDB_monomers/pdbs/5L33.pdb and save per-residue probabilities
```

**Expected Output:**
- Sequence likelihood scores
- Per-residue probabilities (if requested)
- Analysis of sequence quality

### Example 3: Constrained Design

**Goal:** Design sequences with specific residues fixed

**Using Script:**
```bash
python scripts/constrained_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/example3/ \
  --chains_to_design "A B" \
  --fixed_positions "1 2 3 25 26, 10 11 12 15" \
  --num_sequences 3
```

**Using MCP (in Claude Code):**
```
Design sequences for @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb with positions 1,2,3,25,26 fixed in chain A and positions 10,11,12,15 fixed in chain B
```

**Expected Output:**
- Sequences with specified positions preserved
- Constraint validation results
- Design quality metrics

### Example 4: Batch Processing

**Goal:** Process multiple protein structures at once

**Using Script:**
```bash
python scripts/batch_design.py \
  --input_dir examples/data/inputs/PDB_monomers/pdbs \
  --output results/example4/ \
  --num_sequences 2
```

**Using MCP (in Claude Code):**
```
Submit batch processing for all PDB files in @examples/data/inputs/PDB_monomers/pdbs/ generating 2 sequences each, then track the job status
```

**Expected Output:**
- Job ID for tracking
- Results for each processed structure
- Batch summary statistics

### Example 5: CA-Only Design

**Goal:** Design sequences using only backbone coordinates

**Using Script:**
```bash
python scripts/ca_only_design.py \
  --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb \
  --output results/example5/ \
  --chains "A B" \
  --model "v_48_020" \
  --num_sequences 3
```

**Using MCP (in Claude Code):**
```
Use CA-only design with model v_48_020 to generate 3 sequences for @examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb chains A and B
```

**Expected Output:**
- Sequences designed from backbone geometry only
- CA model performance metrics
- Comparison with full-atom design (if available)

---

## Demo Data

The `examples/data/inputs/` directory contains sample data for testing:

| File | Description | Use With | Chains |
|------|-------------|----------|---------|
| `PDB_complexes/pdbs/3HTN.pdb` | Multi-chain protein complex | All tools | A, B |
| `PDB_complexes/pdbs/4YOW.pdb` | Protein-protein complex | All tools | A, B, C, D |
| `PDB_monomers/pdbs/5L33.pdb` | Single-chain protein | simple_design, sequence_scoring | A |
| `PDB_monomers/pdbs/6MRR.pdb` | Membrane protein monomer | simple_design, ca_only_design | A |
| `PDB_homooligomers/pdbs/6EHB.pdb` | Symmetric homooligomer | constrained_design, batch_design | A, B, C, D |
| `PDB_homooligomers/pdbs/4GYT.pdb` | Tetrameric enzyme | All tools | A, B, C, D |

### Model Weights Available
- **Vanilla models**: `v_48_002`, `v_48_010`, `v_48_020` (recommended), `v_48_030`
- **CA-only models**: `v_48_002`, `v_48_010`, `v_48_020`
- **Soluble models**: `v_48_002`, `v_48_010`, `v_48_020`, `v_48_030`

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `simple_design_config.json` | Basic sequence generation | chains, num_sequences, temperature, model |
| `sequence_scoring_config.json` | Likelihood calculation | chains, save_probs, model |
| `constrained_design_config.json` | Fixed position design | chains_to_design, fixed_positions, num_sequences |
| `ca_only_design_config.json` | Backbone-only design | chains, model, num_sequences |
| `batch_design_config.json` | Multi-file processing | file_pattern, chains, num_sequences |
| `default_config.json` | Template and documentation | All parameters with descriptions |

### Config Example

```json
{
  "chains": "A B",
  "num_sequences": 5,
  "temperature": 0.1,
  "seed": 37,
  "model": "v_48_020",
  "use_soluble": false
}
```

### Using Configs

```bash
# Load config file
python scripts/simple_design.py --config configs/simple_design_config.json --input input.pdb

# Override specific parameters
python scripts/simple_design.py --config configs/simple_design_config.json --input input.pdb --temperature 0.2 --num_sequences 10
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found or activation fails
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
mamba run -p ./env pip install torch torchvision numpy loguru click pandas tqdm fastmcp
```

**Problem:** Import errors or missing dependencies
```bash
# Verify environment
mamba activate ./env
python -c "import torch, numpy, pandas; print('Dependencies OK')"

# Check ProteinMPNN import
python -c "import sys; sys.path.append('repo/ProteinMPNN'); import protein_mpnn_utils; print('ProteinMPNN OK')"

# Reinstall if needed
mamba run -p ./env pip install --force-reinstall torch numpy
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove ProteinMPNN
claude mcp add ProteinMPNN -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Test server directly
mamba run -p ./env python src/server.py
```

**Problem:** Tools not working or import errors
```bash
# Test server imports
mamba run -p ./env python -c "
from src.server import mcp
print('MCP server imports successfully')
"

# Check tool count
mamba run -p ./env python -c "
import asyncio
from src.server import mcp
async def test():
    tools = await mcp.get_tools()
    print(f'Found {len(tools)} tools')
asyncio.run(test())
"

# Verify paths
python -c "
import sys
from pathlib import Path
script_dir = Path(__file__).parent.resolve()
print('Script dir:', script_dir)
print('Scripts dir exists:', (script_dir / 'scripts').exists())
print('Examples dir exists:', (script_dir / 'examples').exists())
"
```

### Script Issues

**Problem:** FileNotFoundError when running scripts
```bash
# Use absolute paths
python scripts/simple_design.py --input $(realpath examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb)

# Check file exists
ls -la examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb

# Check current directory
pwd
# Should be: /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/proteinmpnn_mcp
```

**Problem:** Model weights not found
```bash
# Check model directories
ls -la examples/data/
# Should show: vanilla_model_weights, ca_model_weights, soluble_model_weights

# Check specific model
ls -la examples/data/vanilla_model_weights/
# Should contain .pt files for different model versions
```

### Job Issues

**Problem:** Jobs not starting or getting stuck
```bash
# Check job directory
ls -la jobs/

# Check recent job logs
find jobs/ -name "*.log" -exec tail -20 {} \;

# Clean stuck jobs
rm -rf jobs/*/
```

**Problem:** Job failed with errors
```bash
# In Claude Code, check specific job
get_job_log with job_id "your-job-id" and tail 100 to see error details

# Check job status
get_job_status with job_id "your-job-id"

# Cancel if stuck
cancel_job with job_id "your-job-id"
```

### Performance Issues

**Problem:** Slow execution times
```bash
# Check CUDA availability
mamba run -p ./env python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())"

# Reduce batch size in configs
# Edit configs/*.json and set "batch_size": 1

# Use fewer sequences for testing
# Set "num_sequences": 1 or 2
```

**Problem:** Out of memory errors
```bash
# Reduce batch size
# Edit config files to use batch_size: 1

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

# Monitor memory usage
nvidia-smi  # For GPU memory
htop        # For system memory
```

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Run integration tests
python tests/run_integration_tests.py

# Test individual scripts
python scripts/simple_design.py --input examples/data/inputs/PDB_monomers/pdbs/5L33.pdb --output test_output/
```

### Starting Dev Server

```bash
# Run MCP server in development mode
mamba activate ./env
fastmcp dev src/server.py

# Test server in another terminal
mamba activate ./env
python -c "
import asyncio
from src.server import mcp
async def test():
    result = await mcp.call_tool('list_example_structures', {})
    print('Test result:', result)
asyncio.run(test())
"
```

### Adding New Tools

1. Add the script to `scripts/`
2. Add the tool function to `src/server.py`
3. Update this README with the new tool documentation
4. Add integration tests in `tests/`

---

## License

This project uses the MIT License. Based on the original ProteinMPNN implementation.

## Credits

Based on [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) by Justas Dauparas et al.

**Citation:** If you use this software, please cite the original ProteinMPNN paper:
```
Dauparas, J., Anishchenko, I., Bennett, N. et al. Robust deep learning–based protein sequence design using ProteinMPNN. Science 378, 49-56 (2022).
```

---

**Quick Start Summary:**
1. Install environment: `mamba create -p ./env python=3.10 && mamba activate ./env`
2. Install dependencies: `mamba run -p ./env pip install torch fastmcp loguru`
3. Install MCP: `fastmcp install src/server.py --name ProteinMPNN`
4. Use in Claude Code: `What ProteinMPNN tools are available?`

For detailed installation instructions, see the [Installation](#installation) section above.