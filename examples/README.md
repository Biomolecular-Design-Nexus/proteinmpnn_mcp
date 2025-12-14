# ProteinMPNN Examples

This directory contains standalone Python scripts demonstrating different use cases of ProteinMPNN for protein sequence design.

## Use Case Scripts

### Core Functionality
1. **`use_case_1_simple_design.py`** - Basic sequence generation
   - Takes a PDB file and generates new sequences for specified chains
   - Perfect for getting started with ProteinMPNN
   - Example: `python use_case_1_simple_design.py --input data/inputs/PDB_complexes/pdbs/3HTN.pdb`

2. **`use_case_2_sequence_scoring.py`** - Likelihood calculation
   - Scores existing sequences against backbone structures
   - Evaluates sequence-structure compatibility
   - Example: `python use_case_2_sequence_scoring.py --input data/inputs/PDB_complexes/pdbs/3HTN.pdb --save_probs`

### Advanced Features
3. **`use_case_3_batch_design.py`** - Batch processing
   - Processes multiple PDB files in one run
   - Efficient for high-throughput workflows
   - Example: `python use_case_3_batch_design.py --input_dir data/inputs/PDB_monomers/pdbs/`

4. **`use_case_4_constrained_design.py`** - Constrained design
   - Fixes specific residues while optimizing others
   - Useful for preserving active sites or binding interfaces
   - Example: `python use_case_4_constrained_design.py --fixed_positions "1 2 3 4 5, 10 11 12"`

5. **`use_case_5_ca_only_design.py`** - CA-only design
   - Uses only carbon alpha atoms for design
   - Works with backbone traces or low-resolution structures
   - Example: `python use_case_5_ca_only_design.py --input data/inputs/PDB_complexes/pdbs/3HTN.pdb`

## Demo Data

### Input Structures (`data/inputs/`)

**PDB_complexes/** - Multi-chain protein complexes
- `3HTN.pdb` - Protein complex with multiple chains (A, B) - Default test structure
- `4YOW.pdb` - Alternative protein complex

**PDB_monomers/** - Single-chain proteins
- `6MRR.pdb` - Monomer protein for simple design tasks
- `5L33.pdb` - Additional monomer for batch processing examples

**PDB_homooligomers/** - Symmetric protein assemblies
- `6EHB.pdb` - Homooligomer structure for symmetry-based design
- `4GYT.pdb` - Alternative symmetric structure

### Model Weights (`data/`)

**vanilla_model_weights/** - Standard ProteinMPNN models
- `v_48_002.pt` - Basic model
- `v_48_010.pt` - Enhanced with 0.10Å noise
- `v_48_020.pt` - **Recommended default**
- `v_48_030.pt` - Advanced model (slower, more accurate)

**ca_model_weights/** - CA-only models
- `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt` - For backbone-only design

**soluble_model_weights/** - Soluble protein-specific models
- `v_48_010.pt`, `v_48_020.pt` - Optimized for soluble proteins

## Running Examples

### Prerequisites
Make sure you're in the MCP root directory and have activated the environment:
```bash
cd /path/to/proteinmpnn_mcp
mamba run -p ./env python examples/script_name.py [args]
```

### Quick Start
```bash
# Basic sequence generation
mamba run -p ./env python examples/use_case_1_simple_design.py

# Sequence scoring with probabilities
mamba run -p ./env python examples/use_case_2_sequence_scoring.py --save_probs

# Batch process all monomers
mamba run -p ./env python examples/use_case_3_batch_design.py

# Constrained design with fixed positions
mamba run -p ./env python examples/use_case_4_constrained_design.py

# CA-only design
mamba run -p ./env python examples/use_case_5_ca_only_design.py
```

### Common Parameters

All scripts support these common parameters:
- `--input` or `-i`: Input PDB file path
- `--output` or `-o`: Output directory
- `--num_sequences`: Number of sequences to generate (default: 3)
- `--temperature`: Sampling temperature (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 37)
- `--model`: Model version to use (default: v_48_020)

### Getting Help
```bash
# Show all parameters for a script
python examples/use_case_1_simple_design.py --help
```

## Output Organization

### Generated Files
When you run examples, outputs are organized as:
```
examples/outputs/
├── simple_design/          # From use_case_1
├── sequence_scoring/        # From use_case_2
├── batch_design/           # From use_case_3
├── constrained_design/     # From use_case_4
└── ca_only_design/        # From use_case_5
```

### Output File Types
- **`.fa` files**: FASTA format with generated sequences and scores
- **`.npz` files**: NumPy arrays with score data (when --save_score used)
- **`*probs*.npz` files**: Per-residue probabilities (when --save_probs used)
- **`.jsonl` files**: Intermediate parsed data (batch/constrained modes)

## Understanding Results

### Sequence Headers
Generated sequences include metadata in FASTA headers:
```
>3HTN, score=1.1705, global_score=1.2045, fixed_chains=['B'], designed_chains=['A', 'C'], model_name=v_48_020, seed=37
```

**Key Metrics:**
- `score`: Average negative log probability for designed residues (lower = better)
- `global_score`: Average over all residues including fixed ones
- `seq_recovery`: Fraction of original sequence recovered (when applicable)

### Score Interpretation
- **Lower scores = Higher likelihood = Better sequences**
- Typical ranges: 0.5-2.0 for good sequences, >3.0 may indicate problems
- Compare scores between variants to identify best designs

### Temperature Effects
- **0.1**: Conservative, likely sequences (recommended for most uses)
- **0.2-0.3**: More diverse sequences, some may be less stable
- **>0.5**: High diversity, many sequences may not fold properly

## Troubleshooting Examples

### Common Issues
1. **"No such file or directory"**: Check that you're in the MCP root directory
2. **"CUDA out of memory"**: Use smaller proteins or add `--batch_size 1`
3. **"Model not found"**: Verify `data/` directory contains model weights

### Debugging Tips
```bash
# Check if demo data exists
ls examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb

# Test environment
mamba run -p ./env python -c "import torch; print(torch.cuda.is_available())"

# Run with minimal output
python examples/use_case_1_simple_design.py --input examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb --num_sequences 1
```

## Customization

### Using Your Own Structures
```bash
# Use your own PDB file
python examples/use_case_1_simple_design.py --input /path/to/your/structure.pdb

# Specify which chains to design
python examples/use_case_1_simple_design.py --input your_structure.pdb --chains "A C E"
```

### Batch Processing Your Data
```bash
# Point to your PDB directory
python examples/use_case_3_batch_design.py --input_dir /path/to/your/pdbs/

# Increase sequences per target
python examples/use_case_3_batch_design.py --input_dir your_pdbs/ --num_sequences 5
```

### Advanced Constraints
```bash
# Fix specific residues (1-indexed positions)
python examples/use_case_4_constrained_design.py \
    --chains_to_design "A B" \
    --fixed_positions "1 2 3 10 15, 5 6 7 20 25"
    # Chain A: fix positions 1,2,3,10,15
    # Chain B: fix positions 5,6,7,20,25
```