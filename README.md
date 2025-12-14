# ProteinMPNN MCP

> An MCP (Model Context Protocol) server providing both synchronous and asynchronous tools for protein sequence design using ProteinMPNN

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Integration with Claude Code](#integration-with-claude-code)
- [Quick Start Examples](#quick-start-examples)
- [Available Tools](#available-tools)
- [Architecture](#architecture)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

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

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- Git (for cloning ProteinMPNN repository)
- Claude CLI for testing
- ~4GB disk space for model weights

### Quick Setup
```bash
# Navigate to the MCP directory
cd /path/to/proteinmpnn_mcp

# Activate the pre-configured environment
eval "$(mamba shell hook --shell bash)"
mamba activate ./env

# Verify installation
python -c "from src.server import mcp; print('MCP Server OK')"
```

### Full Setup (if needed)
```bash
# Create environment and install dependencies
mamba create -p ./env python=3.10 pip -y
mamba activate ./env
pip install fastmcp loguru torch torchvision numpy click pandas tqdm

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## Integration with Claude Code

### Installation
```bash
# Register MCP server with Claude Code
claude mcp add ProteinMPNN -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify registration
claude mcp list
# Should show: ProteinMPNN: ... - ✓ Connected
```

### Verification
```bash
# Test server startup
python src/server.py &
sleep 5
pkill -f "python src/server.py"

# Check tool availability
python -c "
import asyncio
from src.server import mcp
async def test():
    tools = await mcp.get_tools()
    print(f'Found {len(tools)} tools')
asyncio.run(test())
"
```

### With Claude Desktop (Alternative)
Add to your Claude configuration file:
```json
{
  "mcpServers": {
    "ProteinMPNN": {
      "type": "stdio",
      "command": "/absolute/path/to/proteinmpnn_mcp/env/bin/python",
      "args": ["/absolute/path/to/proteinmpnn_mcp/src/server.py"]
    }
  }
}
```

### With Gemini CLI (Optional)
Add to `~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "ProteinMPNN": {
      "command": "/absolute/path/to/proteinmpnn_mcp/env/bin/python",
      "args": ["/absolute/path/to/proteinmpn_mcp/src/server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/proteinmpnn_mcp"
      }
    }
  }
}
```

## Quick Start Examples

### In Claude Code:
```
# List available tools
"""markdown
What MCP tools are available from ProteinMPNN?
"""

# Validate a structure
"""markdown
Use validate_pdb_structure to check examples/data/inputs/PDB_monomers/pdbs/5L33.pdb
"""

# Design sequences
"""markdown
Use simple_design with input_file='examples/data/inputs/PDB_monomers/pdbs/5L33.pdb' chains='A' and num_sequences=3

Please use absolution path to call the mcp servers.
"""

# Submit a long-running job
"""markdown
Use submit_large_design for examples/data/inputs/PDB_monomers/pdbs/5L33.pdb with 20 sequences.

Please use absolution path to call the mcp servers.
"""

# Check job status
"""
Check the status of job f021435b
"""
```


## Troubleshooting

### Claude Code connection issues
```bash
# Remove and re-add server
claude mcp remove ProteinMPNN
claude mcp add ProteinMPNN -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Check server health
claude mcp list
```

### Tools not found in Claude
- Ensure server shows "✓ Connected" in `claude mcp list`
- Restart Claude CLI if needed
- Check absolute paths in registration command

### Jobs stuck in pending
```bash
# Check job directory permissions
ls -la jobs/

# View job logs
ls jobs/*/
cat jobs/*/job.log
```

### File path errors
- Use absolute paths for input files
- Ensure files exist: `ls -la examples/data/inputs/PDB_monomers/pdbs/5L33.pdb`
- Check current working directory

### Import errors
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Verify dependencies
pip list | grep -E "fastmcp|torch|loguru"
```

## Testing & Validation

### Pre-deployment Checklist
- [ ] `claude mcp list` shows ProteinMPNN as Connected
- [ ] All 13 tools discoverable
- [ ] `validate_pdb_structure` works with example files
- [ ] `list_example_structures` returns data
- [ ] `simple_design` executes without errors
- [ ] Submit API returns job IDs
- [ ] Job status tracking works
- [ ] Error handling for invalid inputs

### Performance Expectations
- **Tool discovery**: < 2 seconds
- **File validation**: < 5 seconds
- **Simple design (1-3 sequences)**: < 30 seconds
- **Job submission**: < 1 second
- **Status checks**: < 2 seconds

### Test Files Available
- `examples/data/inputs/PDB_monomers/pdbs/5L33.pdb` (181KB, Chain A)
- `examples/data/inputs/PDB_monomers/pdbs/6MRR.pdb` (Chain A)
- `examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb` (Multi-chain)
- `examples/data/inputs/PDB_homooligomers/pdbs/6EHB.pdb` (Oligomer)
```

### Direct Server Start
```bash
mamba run -p ./env python src/server.py
```

## Available Tools

### Quick Operations (Synchronous API)
These tools return results immediately (< 10 minutes):

| Tool | Description | Runtime | Usage |
|------|-------------|---------|-------|
| `simple_design` | Generate sequences for PDB structure | ~10 sec | Single structure design |
| `sequence_scoring` | Score sequences with ProteinMPNN | ~8 sec | Evaluate sequence fitness |
| `constrained_design` | Design with fixed positions | ~15 sec | Position-specific constraints |
| `ca_only_design` | Design from CA-only structures | ~12 sec | Backbone-only input |
| `validate_pdb_structure` | Check PDB compatibility | <1 sec | Input validation |
| `list_example_structures` | List example files | <1 sec | Find test data |

### Long-Running Tasks (Submit API)
These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Runtime | Usage |
|------|-------------|---------|-------|
| `submit_batch_design` | Process multiple PDB files | >10 min | Batch processing |
| `submit_large_design` | Generate many sequences | >10 min | Large-scale design |

### Job Management
| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and status |
| `get_job_result` | Retrieve completed job results |
| `get_job_log` | View job execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all submitted jobs |

## Workflow Examples

### Quick Analysis (Synchronous)
```
Use the simple_design tool with:
- input_file: "examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb"
- chains: "A B"
- num_sequences: 3
→ Returns: Generated sequences immediately
```

### Long-Running Prediction (Asynchronous)
```
1. Submit: Use submit_batch_design with:
   - input_dir: "examples/data/inputs/PDB_monomers/pdbs"
   - file_pattern: "*.pdb"
   → Returns: {"job_id": "abc123", "status": "submitted"}

2. Monitor: Use get_job_status with job_id "abc123"
   → Returns: {"status": "running", "started_at": "...", ...}

3. Get logs: Use get_job_log with job_id "abc123"
   → Returns: Real-time execution logs

4. Get results: Use get_job_result with job_id "abc123"
   → Returns: All generated sequences and files
```

### Batch Processing
```
Use submit_batch_design with:
- input_dir: "path/to/multiple/pdbs"
- num_sequences: 5
→ Processes all PDB files in a single tracked job
```

### Constrained Design
```
Use constrained_design with:
- input_file: "structure.pdb"
- chains_to_design: "A B"
- fixed_positions: "1 2 3, 10 11 12"
→ Designs sequences with specified positions fixed
```

## Tool Parameters

### Common Parameters
- **input_file**: Path to PDB structure file
- **chains**: Space-separated chain IDs (e.g., "A B C")
- **num_sequences**: Number of sequences to generate
- **temperature**: Sampling temperature (default: 0.1)
- **output_dir**: Directory for output files

### Async-Specific Parameters
- **job_name**: Optional name for easier job tracking
- **input_dir**: Directory containing multiple PDB files
- **file_pattern**: Pattern to match files (e.g., "*.pdb")

### Advanced Parameters
- **fixed_positions**: Constraint specification (format: "1 2 3, 10 11 12")
- **model**: Model version for CA-only design
- **save_probs**: Save per-residue probabilities (scoring)

## Output Formats

### Synchronous Tools
```json
{
  "status": "success",
  "sequences": [...],
  "metadata": {
    "execution_time": "8.2 seconds",
    "model_used": "v_48_020",
    "input_chains": ["A", "B"]
  },
  "output_files": ["/path/to/sequences.fasta"]
}
```

### Asynchronous Jobs
```json
{
  "status": "submitted",
  "job_id": "abc123",
  "message": "Job submitted. Use get_job_status('abc123') to check progress."
}
```

### Job Status Response
```json
{
  "job_id": "abc123",
  "job_name": "batch_design_monomers",
  "status": "completed",
  "submitted_at": "2024-12-14T14:30:00",
  "started_at": "2024-12-14T14:30:05",
  "completed_at": "2024-12-14T14:45:32",
  "output_files": ["/path/to/results.fasta", "/path/to/metadata.json"]
}
```

## Example Files

The server includes example structures for testing:

```bash
examples/data/inputs/
├── PDB_complexes/pdbs/3HTN.pdb     # Multi-chain complex
├── PDB_homooligomers/pdbs/6EHB.pdb # Homooligomer
├── PDB_monomers/pdbs/              # Single chain structures
└── ...
```

Use `list_example_structures` to see all available examples.

## Architecture

### Directory Structure
```
src/
├── server.py              # Main MCP server
├── tools/                 # Tool definitions
├── jobs/                  # Job management
│   ├── manager.py         # Job queue and execution
│   └── store.py          # Job persistence
└── utils.py              # Shared utilities

scripts/                   # Clean implementation scripts
├── simple_design.py      # Basic sequence design
├── sequence_scoring.py   # Sequence evaluation
├── batch_design.py       # Multi-file processing
├── constrained_design.py # Position constraints
├── ca_only_design.py     # CA-only design
└── lib/                  # Shared library functions
```

### API Design

**Synchronous API** (< 10 minutes)
- Direct function calls with immediate response
- Suitable for single structures, quick analysis
- Error handling with structured responses

**Submit API** (> 10 minutes)
- Background job execution with job_id tracking
- Real-time log monitoring and status updates
- Persistent job state across server restarts

### Job Management Features
- **Threading**: Background execution without blocking
- **Persistence**: Jobs survive server restarts
- **Logging**: Real-time execution monitoring
- **Cancellation**: Stop running jobs gracefully
- **Output Collection**: Structured result aggregation

## Development

### Testing
```bash
# Run component tests
mamba run -p ./env python test_tools.py

# Test server startup
mamba run -p ./env python test_server.py

# Development mode
fastmcp dev src/server.py
```

### Adding New Tools
1. Implement tool function in `src/server.py`
2. Choose API type (sync vs submit)
3. Add parameter validation and error handling
4. Update documentation in `reports/mcp_tools.json`
5. Add tests to verify functionality

### Configuration
- **Model weights**: Place in `examples/data/`
- **Environment**: Use conda/mamba environment in `./env`
- **Scripts**: Core functionality in `scripts/` directory
- **Config files**: JSON configs in `configs/` directory

## Troubleshooting

### Common Issues
1. **Environment activation**: Use `mamba run -p ./env` instead of `mamba activate`
2. **CUDA availability**: Verify with `torch.cuda.is_available()`
3. **Model weights**: Ensure files are in `examples/data/`
4. **Path issues**: Use absolute paths for input files

### Error Handling
- All tools return structured error responses
- Check `status` field in responses
- Use `get_job_log` to debug failed jobs
- Validate inputs with `validate_pdb_structure`

### Performance Tips
- Use GPU when available for faster processing
- Prefer batch processing for multiple files
- Monitor job logs for progress tracking
- Cancel unnecessary jobs to free resources

## License

This MCP server wraps the original ProteinMPNN implementation. Please cite the original work and ProteinMCP:

```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

---

**Status**: ✅ Fully functional MCP server with 11 tools (4 sync, 2 submit, 5 management)
**Version**: 1.0.0
**Last Updated**: 2024-12-14