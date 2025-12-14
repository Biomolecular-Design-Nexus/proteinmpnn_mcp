# ProteinMPNN MCP Scripts

Clean, self-contained scripts extracted from verified use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported (argparse, subprocess, pathlib, json)
2. **Self-Contained**: Repository functions inlined where possible using shared library
3. **Configurable**: Parameters externalized to config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts Overview

| Script | Description | Repo Dependent | Config File |
|--------|-------------|----------------|-------------|
| `simple_design.py` | Generate protein sequences for PDB structure | Yes (model loading) | `configs/simple_design_config.json` |
| `sequence_scoring.py` | Score sequences using ProteinMPNN likelihood | Yes (model loading) | `configs/sequence_scoring_config.json` |
| `batch_design.py` | Batch process multiple PDB files | Yes (parsing + model) | `configs/batch_design_config.json` |
| `constrained_design.py` | Design with fixed position constraints | Yes (helpers + model) | `configs/constrained_design_config.json` |
| `ca_only_design.py` | Design using carbon alpha coordinates only | Yes (CA model) | `configs/ca_only_design_config.json` |

## Installation & Setup

### Prerequisites

1. **Conda Environment**: Activate the ProteinMPNN environment
   ```bash
   source ~/miniforge3/etc/profile.d/conda.sh
   conda activate ./env
   ```

2. **Model Weights**: Ensure model weights are available in `examples/data/`
   - `vanilla_model_weights/` - Standard models
   - `ca_model_weights/` - Carbon alpha models
   - `soluble_model_weights/` - Soluble protein models

3. **ProteinMPNN Repository**: Scripts depend on `repo/ProteinMPNN/` for core functionality

## Usage

### Basic Usage

```bash
# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./env

# Run a script with absolute paths (recommended)
python scripts/simple_design.py \
  --input $(realpath examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb) \
  --output $(realpath results/simple_design) \
  --num_sequences 3

# Run with config file
python scripts/simple_design.py \
  --config configs/simple_design_config.json \
  --input $(realpath examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb)
```

### Script-Specific Examples

#### 1. Simple Design
```bash
python scripts/simple_design.py \
  --input $(realpath examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb) \
  --output results/simple_design \
  --chains "A B" \
  --num_sequences 5 \
  --temperature 0.1
```

#### 2. Sequence Scoring
```bash
# Score native sequence
python scripts/sequence_scoring.py \
  --input $(realpath examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb) \
  --output results/scoring \
  --chains "A B" \
  --save_probs

# Score custom sequences
python scripts/sequence_scoring.py \
  --input $(realpath examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb) \
  --output results/scoring \
  --fasta_sequences "MKTAYIAK.../DVFSLREM..."
```

#### 3. Batch Design
```bash
python scripts/batch_design.py \
  --input_dir $(realpath examples/data/inputs/PDB_monomers/pdbs) \
  --output results/batch \
  --num_sequences 2
```

#### 4. Constrained Design
```bash
python scripts/constrained_design.py \
  --input $(realpath examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb) \
  --output results/constrained \
  --chains_to_design "A B" \
  --fixed_positions "1 2 3 25 26, 10 11 12 15"
```

#### 5. CA-Only Design
```bash
python scripts/ca_only_design.py \
  --input $(realpath examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb) \
  --output results/ca_only \
  --chains "A B" \
  --num_sequences 3
```

## Configuration Files

All scripts support JSON configuration files in the `configs/` directory:

- `simple_design_config.json` - Basic sequence generation
- `sequence_scoring_config.json` - Likelihood calculation
- `batch_design_config.json` - Batch processing
- `constrained_design_config.json` - Fixed position constraints
- `ca_only_design_config.json` - Carbon alpha design
- `default_config.json` - Template and documentation

### Using Config Files

```bash
# Use config file with optional overrides
python scripts/simple_design.py \
  --config configs/simple_design_config.json \
  --input input.pdb \
  --temperature 0.2
```

## Output Structure

Each script creates a standardized output directory:

```
results/
├── simple_design/
│   ├── seqs/                    # FASTA sequence files
│   ├── scores/                  # Score data (if applicable)
│   ├── logs/                    # Log files
│   └── execution_metadata.json # Execution details
└── ...
```

### Output Files

- **Sequence Files**: FASTA format with ProteinMPNN headers containing scores and metadata
- **Score Files**: NPZ format with likelihood scores and probabilities
- **Metadata**: JSON with execution details, parameters, and file paths
- **Constraint Files**: JSONL format for batch and constrained design

## Shared Library

Common functions are in `scripts/lib/`:

### `lib/io.py`
- `load_fasta()`, `save_fasta()` - FASTA file operations
- `load_json()`, `save_json()` - JSON file operations
- `parse_fasta_sequences()` - Parse command-line FASTA strings

### `lib/utils.py`
- `validate_chains()` - Chain ID validation
- `get_model_path()` - Model weight path resolution
- `get_repo_path()` - Repository path resolution
- `parse_fixed_positions()` - Constraint parsing
- `extract_scores_from_fasta_header()` - Score extraction

## Dependencies

### Essential Dependencies (Imported)
- `argparse` - Command line parsing
- `subprocess` - ProteinMPNN execution
- `pathlib` - Path handling
- `json` - Configuration files
- `numpy` - Score data processing (scoring script only)

### Repository Dependencies (Required)
- `repo/ProteinMPNN/protein_mpnn_run.py` - Main ProteinMPNN script
- `repo/ProteinMPNN/helper_scripts/` - PDB parsing and constraint helpers
- Model weight files in `examples/data/`

### Inlined Functions
Simple utility functions were inlined to reduce dependencies:
- Chain validation and parsing
- Path resolution and management
- Basic file I/O operations
- Score extraction from FASTA headers

## MCP Integration

Each script exports a main function ready for MCP wrapping:

```python
# Example MCP tool wrapper
from scripts.simple_design import run_simple_design

@mcp.tool()
def protein_design(input_file: str, output_dir: str = None, num_sequences: int = 3):
    """Generate protein sequences using ProteinMPNN."""
    return run_simple_design(
        input_file=input_file,
        output_dir=output_dir,
        num_sequences=num_sequences
    )
```

### Function Signatures

```python
# Simple design
run_simple_design(input_file, output_dir=None, config=None, **kwargs) -> dict

# Sequence scoring
run_sequence_scoring(input_file, output_dir=None, config=None, **kwargs) -> dict

# Batch design
run_batch_design(input_dir, output_dir=None, config=None, **kwargs) -> dict

# Constrained design
run_constrained_design(input_file, output_dir=None, config=None, **kwargs) -> dict

# CA-only design
run_ca_only_design(input_file, output_dir=None, config=None, **kwargs) -> dict
```

## Important Notes

### Path Requirements
- **Use absolute paths** when calling scripts to avoid path resolution issues
- The `$(realpath path)` command converts relative to absolute paths
- ProteinMPNN runs from its repo directory, so relative paths may fail

### Model Availability
- **Vanilla models**: v_48_002, v_48_010, v_48_020 (recommended), v_48_030
- **CA models**: v_48_002, v_48_010, v_48_020 (no v_48_030)
- **Soluble models**: v_48_002, v_48_010, v_48_020, v_48_030

### Performance Considerations
- Single sequences: ~10-20 seconds
- Batch processing: Scales with number of PDBs
- Constraint generation adds ~5-10 seconds overhead
- Model loading is cached within each run

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Use absolute paths with `$(realpath path)`
2. **Import errors**: Ensure conda environment is activated
3. **Model not found**: Check model weights in `examples/data/`
4. **Permission errors**: Ensure write permissions for output directory

### Debug Information

Each script saves detailed metadata including:
- Full command executed
- Input/output file paths
- Configuration used
- Model paths
- Execution timing

Check `execution_metadata.json` in output directory for debugging.

## Testing

The scripts have been tested with the verified use case data:

- ✅ `simple_design.py` - Generates sequences for 3HTN.pdb
- ✅ `sequence_scoring.py` - Scores native sequences
- ✅ `batch_design.py` - Processes multiple PDB files
- ✅ `constrained_design.py` - Applies position constraints
- ✅ `ca_only_design.py` - Uses CA coordinates only

All scripts produce outputs compatible with the original use case results.