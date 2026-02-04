#!/usr/bin/env python3
"""
Script: constrained_design.py
Description: Design proteins with constrained/fixed positions using ProteinMPNN

Original Use Case: examples/use_case_4_constrained_design.py
Dependencies Removed: Inlined constraint helper functions and simplified position handling

Usage:
    python scripts/constrained_design.py --input <input_file> --output <output_dir> --fixed_positions <positions>

Example:
    python scripts/constrained_design.py --input examples/data/inputs/PDB_complexes/3HTN.pdb --output results/constrained --fixed_positions "1 2 3, 10 11 12"
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import subprocess
import os
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json

# Local imports from shared library
from lib.io import load_fasta, save_json, load_json
from lib.utils import (
    validate_chains, format_command, check_file_exists,
    get_model_path, get_repo_path, create_output_directories,
    extract_scores_from_fasta_header, parse_fixed_positions
)

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "chains_to_design": "A B",
    "fixed_positions": "",  # Format: "1 2 3, 10 11 12" for chains A and B
    "num_sequences": 3,
    "temperature": 0.1,
    "seed": 37,
    "model": "v_48_020",
    "use_soluble": False,
    "batch_size": 1,
    "backbone_noise": 0.0,
    "omit_AAs": ""
}

VALID_MODELS = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

# ==============================================================================
# Utility Functions (inlined from repo helper scripts)
# ==============================================================================
def create_chain_assignment_file(
    pdb_file: Path,
    chains_to_design: List[str],
    output_dir: Path
) -> Path:
    """Create chain assignment file for ProteinMPNN.

    This function is inlined from repo/ProteinMPNN/helper_scripts/assign_fixed_chains.py
    to minimize dependencies on repo helper scripts.

    Args:
        pdb_file: Path to PDB file
        chains_to_design: List of chain identifiers to design
        output_dir: Directory to save assignment file

    Returns:
        Path to generated assigned_pdbs.jsonl file
    """
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    assigned_path = output_dir / "assigned_pdbs.jsonl"

    # Create assignment using repo script
    # Use sys.executable to ensure we use the same Python interpreter
    cmd = [
        sys.executable, str(repo_path / "helper_scripts" / "assign_fixed_chains.py"),
        "--input_path", str(pdb_file.parent),
        "--output_path", str(assigned_path),
        "--chain_list", " ".join(chains_to_design)
    ]

    print(f"🔗 Creating chain assignments for chains: {' '.join(chains_to_design)}")

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Chain assignment failed: {result.stderr}")

        print(f"✅ Chain assignments created successfully")
        return assigned_path

    except Exception as e:
        raise RuntimeError(f"Error creating chain assignments: {e}")


def create_fixed_positions_file(
    pdb_file: Path,
    chains_to_design: List[str],
    fixed_positions_by_chain: Dict[int, List[int]],
    output_dir: Path
) -> Path:
    """Create fixed positions file for ProteinMPNN.

    This function is inlined from repo/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py
    to minimize dependencies on repo helper scripts.

    Args:
        pdb_file: Path to PDB file
        chains_to_design: List of chain identifiers to design
        fixed_positions_by_chain: Dictionary mapping chain index to position lists
        output_dir: Directory to save fixed positions file

    Returns:
        Path to generated fixed_pdbs.jsonl file
    """
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    fixed_path = output_dir / "fixed_pdbs.jsonl"

    # Create position list argument for the helper script
    position_args = []
    for chain_idx, chain_id in enumerate(chains_to_design):
        if chain_idx in fixed_positions_by_chain:
            positions = fixed_positions_by_chain[chain_idx]
            position_str = " ".join(map(str, positions))
            position_args.extend(["--position_list", position_str])

    if not position_args:
        print("⚠️  No fixed positions specified")
        # Create empty constraints file
        with open(fixed_path, 'w') as f:
            json.dump({}, f)
        return fixed_path

    # Create fixed positions using repo script
    # Use sys.executable to ensure we use the same Python interpreter
    cmd = [
        sys.executable, str(repo_path / "helper_scripts" / "make_fixed_positions_dict.py"),
        "--input_path", str(pdb_file.parent),
        "--output_path", str(fixed_path),
        "--chain_list", " ".join(chains_to_design)
    ] + position_args

    print(f"🔒 Creating fixed position constraints:")
    for chain_idx, positions in fixed_positions_by_chain.items():
        chain_id = chains_to_design[chain_idx] if chain_idx < len(chains_to_design) else f"Chain{chain_idx}"
        print(f"  - {chain_id}: positions {positions}")

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Fixed positions creation failed: {result.stderr}")

        print(f"✅ Fixed position constraints created successfully")
        return fixed_path

    except Exception as e:
        raise RuntimeError(f"Error creating fixed positions: {e}")


def parse_pdb_single(pdb_file: Path, output_dir: Path) -> Path:
    """Parse a single PDB file for ProteinMPNN.

    Args:
        pdb_file: Path to PDB file
        output_dir: Directory to save parsed file

    Returns:
        Path to generated parsed_pdbs.jsonl file
    """
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    parsed_path = output_dir / "parsed_pdbs.jsonl"

    # Use sys.executable to ensure we use the same Python interpreter
    cmd = [
        sys.executable, str(repo_path / "helper_scripts" / "parse_multiple_chains.py"),
        "--input_path", str(pdb_file.parent),
        "--output_path", str(parsed_path)
    ]

    print(f"📋 Parsing PDB file: {pdb_file.name}")

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"PDB parsing failed: {result.stderr}")

        print(f"✅ PDB file parsed successfully")
        return parsed_path

    except Exception as e:
        raise RuntimeError(f"Error parsing PDB file: {e}")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_constrained_design(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Design proteins with constrained/fixed positions using ProteinMPNN.

    Args:
        input_file: Path to input PDB file
        output_dir: Directory to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - output_dir: Path to output directory
            - sequence_files: List of generated sequence files
            - constraint_files: List of constraint files created
            - sequences: Generated sequence data with constraints
            - metadata: Execution metadata

    Example:
        >>> result = run_constrained_design(
        ...     "input.pdb",
        ...     fixed_positions="1 2 3, 10 11 12",
        ...     chains_to_design="A B"
        ... )
        >>> print(f"Generated {len(result['sequences'])} constrained sequences")
    """
    # Setup
    input_file = check_file_exists(input_file, "PDB file")
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate configuration
    chains_list = validate_chains(config["chains_to_design"])
    if config["model"] not in VALID_MODELS:
        raise ValueError(f"Invalid model: {config['model']}. Must be one of {VALID_MODELS}")

    # Parse fixed positions
    fixed_positions_by_chain = parse_fixed_positions(config["fixed_positions"])

    # Setup output directory
    if output_dir is None:
        output_dir = Path("results") / "constrained_design"
    # Convert to absolute path to ensure correct location when subprocess runs with different cwd
    output_dir = Path(output_dir).resolve()
    output_dirs = create_output_directories(output_dir)

    # Get paths
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    model_path = get_model_path(script_dir, config["use_soluble"])

    print(f"🧬 Running ProteinMPNN Constrained Design")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_dirs['base']}")
    print(f"🔗 Chains to design: {config['chains_to_design']}")
    print(f"🔒 Fixed positions: {config['fixed_positions']}")
    print(f"🎯 Sequences: {config['num_sequences']}")
    print(f"🧠 Model: {config['model']}")
    print("-" * 50)

    # Step 1: Parse PDB file
    parsed_pdbs_file = parse_pdb_single(input_file, output_dirs['base'])

    # Step 2: Create chain assignment file
    assigned_pdbs_file = create_chain_assignment_file(
        input_file, chains_list, output_dirs['base']
    )

    # Step 3: Create fixed positions file
    fixed_pdbs_file = create_fixed_positions_file(
        input_file, chains_list, fixed_positions_by_chain, output_dirs['base']
    )

    # Prepare ProteinMPNN command
    # Use sys.executable to ensure we use the same Python interpreter (with torch installed)
    cmd = [
        sys.executable, str(repo_path / "protein_mpnn_run.py"),
        "--jsonl_path", str(parsed_pdbs_file),
        "--chain_id_jsonl", str(assigned_pdbs_file),
        "--fixed_positions_jsonl", str(fixed_pdbs_file),
        "--out_folder", str(output_dirs['base']),
        "--num_seq_per_target", str(config["num_sequences"]),
        "--sampling_temp", str(config["temperature"]),
        "--seed", str(config["seed"]),
        "--model_name", config["model"],
        "--batch_size", str(config["batch_size"]),
        "--path_to_model_weights", str(model_path)
    ]

    # Add soluble model flag if needed
    if config["use_soluble"]:
        cmd.append("--use_soluble_model")

    # Add backbone noise if specified
    if config.get("backbone_noise", 0.0) > 0:
        cmd.extend(["--backbone_noise", str(config["backbone_noise"])])

    # Add omit_AAs if specified
    if config.get("omit_AAs"):
        cmd.extend(["--omit_AAs", config["omit_AAs"]])

    # Execute ProteinMPNN
    print(f"\n🔧 Executing ProteinMPNN with constraints...")
    print(f"💻 Command: {format_command(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ProteinMPNN failed: {result.stderr}")

        print("✅ ProteinMPNN constrained design completed successfully!")

    except Exception as e:
        raise RuntimeError(f"Error running ProteinMPNN: {e}")

    # Process results
    sequence_files = list(output_dirs['base'].glob("**/*.fa"))
    constraint_files = [assigned_pdbs_file, fixed_pdbs_file, parsed_pdbs_file]

    sequences = []
    for seq_file in sequence_files:
        try:
            file_sequences = load_fasta(seq_file)
            for seq_data in file_sequences:
                scores = extract_scores_from_fasta_header(seq_data['header'])
                sequences.append({
                    'file': str(seq_file),
                    'header': seq_data['header'],
                    'sequence': seq_data['sequence'],
                    'scores': scores,
                    'length': len(seq_data['sequence']),
                    'target': seq_file.stem  # PDB name
                })
        except Exception as e:
            print(f"Warning: Could not parse sequence file {seq_file}: {e}")

    # Load constraint data for metadata
    constraint_data = {}
    for constraint_file in constraint_files:
        if constraint_file.exists():
            try:
                constraint_name = constraint_file.stem
                if constraint_file.suffix == '.jsonl':
                    # JSONL format
                    constraint_data[constraint_name] = []
                    with open(constraint_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    constraint_data[constraint_name].append(json.loads(line.strip()))
                                except json.JSONDecodeError:
                                    continue
                else:
                    # Regular JSON
                    constraint_data[constraint_name] = load_json(constraint_file)
            except Exception as e:
                print(f"Warning: Could not load constraint file {constraint_file}: {e}")

    # Save execution metadata
    metadata = {
        "input_file": str(input_file),
        "output_directory": str(output_dirs['base']),
        "config": config,
        "chains_to_design": chains_list,
        "fixed_positions_by_chain": fixed_positions_by_chain,
        "constraint_files": {
            "parsed_pdbs": str(parsed_pdbs_file),
            "assigned_pdbs": str(assigned_pdbs_file),
            "fixed_pdbs": str(fixed_pdbs_file)
        },
        "sequences_generated": len(sequences),
        "sequence_files": [str(f) for f in sequence_files],
        "model_path": str(model_path),
        "command_executed": format_command(cmd),
        "constraint_data": constraint_data
    }

    # Save metadata to output directory
    metadata_file = output_dirs['base'] / "execution_metadata.json"
    save_json(metadata, metadata_file)

    print(f"\n📊 Constrained design summary:")
    print(f"  🧬 Sequences generated: {len(sequences)}")
    print(f"  📄 Sequence files: {len(sequence_files)}")
    print(f"  🔒 Constraint files: {len(constraint_files)}")
    print(f"💾 Metadata saved to: {metadata_file}")

    # Show constraint summary
    if fixed_positions_by_chain:
        print(f"\n🔒 Applied constraints:")
        for chain_idx, positions in fixed_positions_by_chain.items():
            chain_id = chains_list[chain_idx] if chain_idx < len(chains_list) else f"Chain{chain_idx}"
            print(f"  - {chain_id}: fixed positions {positions}")

    return {
        "output_dir": str(output_dirs['base']),
        "sequence_files": [str(f) for f in sequence_files],
        "constraint_files": [str(f) for f in constraint_files],
        "sequences": sequences,
        "metadata": metadata
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Input PDB file path'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output directory path (default: results/constrained_design)'
    )
    parser.add_argument(
        '--chains_to_design',
        default=DEFAULT_CONFIG["chains_to_design"],
        help=f'Chains to design (default: {DEFAULT_CONFIG["chains_to_design"]})'
    )
    parser.add_argument(
        '--fixed_positions',
        default=DEFAULT_CONFIG["fixed_positions"],
        help='Fixed positions per chain (format: "1 2 3, 10 11 12" for chain A and B)'
    )
    parser.add_argument(
        '--num_sequences', type=int,
        default=DEFAULT_CONFIG["num_sequences"],
        help=f'Number of sequences to generate (default: {DEFAULT_CONFIG["num_sequences"]})'
    )
    parser.add_argument(
        '--temperature', type=float,
        default=DEFAULT_CONFIG["temperature"],
        help=f'Sampling temperature (default: {DEFAULT_CONFIG["temperature"]})'
    )
    parser.add_argument(
        '--seed', type=int,
        default=DEFAULT_CONFIG["seed"],
        help=f'Random seed (default: {DEFAULT_CONFIG["seed"]})'
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_CONFIG["model"],
        choices=VALID_MODELS,
        help=f'Model version (default: {DEFAULT_CONFIG["model"]})'
    )
    parser.add_argument(
        '--use_soluble', action='store_true',
        help='Use soluble protein model weights'
    )
    parser.add_argument(
        '--backbone_noise', type=float,
        default=DEFAULT_CONFIG["backbone_noise"],
        help=f'Gaussian noise std dev for backbone atoms (default: {DEFAULT_CONFIG["backbone_noise"]})'
    )
    parser.add_argument(
        '--omit_AAs',
        default=DEFAULT_CONFIG["omit_AAs"],
        help='Amino acids to exclude from design (e.g., "C" to exclude cysteine)'
    )
    parser.add_argument(
        '--config', '-c',
        help='Config file (JSON) to override defaults'
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    config.update({
        'chains_to_design': args.chains_to_design,
        'fixed_positions': args.fixed_positions,
        'num_sequences': args.num_sequences,
        'temperature': args.temperature,
        'seed': args.seed,
        'model': args.model,
        'use_soluble': args.use_soluble,
        'backbone_noise': args.backbone_noise,
        'omit_AAs': args.omit_AAs
    })

    # Run
    try:
        result = run_constrained_design(
            input_file=args.input,
            output_dir=args.output,
            config=config
        )

        print(f"\n✅ Success: {result['output_dir']}")
        print(f"📄 Generated {len(result['sequence_files'])} sequence files:")
        for seq_file in result['sequence_files']:
            print(f"  - {seq_file}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())