#!/usr/bin/env python3
"""
Script: simple_design.py
Description: Generate protein sequences for a given PDB structure using ProteinMPNN

Original Use Case: examples/use_case_1_simple_design.py
Dependencies Removed: Simplified subprocess handling and path management

Usage:
    python scripts/simple_design.py --input <input_file> --output <output_dir>

Example:
    python scripts/simple_design.py --input examples/data/inputs/PDB_complexes/3HTN.pdb --output results/simple_design
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import subprocess
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json

# Local imports from shared library
from lib.io import load_fasta, save_json
from lib.utils import (
    validate_chains, format_command, check_file_exists,
    get_model_path, get_repo_path, create_output_directories,
    extract_scores_from_fasta_header
)

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "chains": "A B",
    "num_sequences": 3,
    "temperature": 0.1,
    "seed": 37,
    "model": "v_48_020",
    "use_soluble": False,
    "batch_size": 1
}

VALID_MODELS = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_simple_design(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate protein sequences for a given PDB structure using ProteinMPNN.

    Args:
        input_file: Path to input PDB file
        output_dir: Directory to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - output_dir: Path to output directory
            - sequence_files: List of generated sequence files
            - sequences: List of sequence data with metadata
            - metadata: Execution metadata

    Example:
        >>> result = run_simple_design("input.pdb", "output/")
        >>> print(result['sequence_files'])
        >>> print(len(result['sequences']))
    """
    # Setup
    input_file = check_file_exists(input_file, "PDB file")
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate configuration
    chains_list = validate_chains(config["chains"])
    if config["model"] not in VALID_MODELS:
        raise ValueError(f"Invalid model: {config['model']}. Must be one of {VALID_MODELS}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("results") / "simple_design"
    # Convert to absolute path to ensure correct location when subprocess runs with different cwd
    output_dir = Path(output_dir).resolve()
    output_dirs = create_output_directories(output_dir)

    # Get paths
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    model_path = get_model_path(script_dir, config["use_soluble"])

    # Prepare ProteinMPNN command
    cmd = [
        "python", str(repo_path / "protein_mpnn_run.py"),
        "--pdb_path", str(input_file),
        "--pdb_path_chains", config["chains"],
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

    # Execute ProteinMPNN
    print(f"🧬 Running ProteinMPNN Simple Design")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_dirs['base']}")
    print(f"🔗 Chains: {config['chains']}")
    print(f"🎯 Sequences: {config['num_sequences']}")
    print(f"🧠 Model: {config['model']}")
    print(f"💻 Command: {format_command(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ProteinMPNN failed: {result.stderr}")

        print("✅ ProteinMPNN completed successfully!")

    except Exception as e:
        raise RuntimeError(f"Error running ProteinMPNN: {e}")

    # Process results
    sequence_files = list(output_dirs['base'].glob("**/*.fa"))
    sequences = []

    for seq_file in sequence_files:
        try:
            file_sequences = load_fasta(seq_file)
            for seq_data in file_sequences:
                # Extract scores from header
                scores = extract_scores_from_fasta_header(seq_data['header'])
                sequences.append({
                    'file': str(seq_file),
                    'header': seq_data['header'],
                    'sequence': seq_data['sequence'],
                    'scores': scores,
                    'length': len(seq_data['sequence'])
                })
        except Exception as e:
            print(f"Warning: Could not parse sequence file {seq_file}: {e}")

    # Save execution metadata
    metadata = {
        "input_file": str(input_file),
        "output_directory": str(output_dirs['base']),
        "config": config,
        "chains_designed": chains_list,
        "sequences_generated": len(sequences),
        "sequence_files": [str(f) for f in sequence_files],
        "model_path": str(model_path),
        "command_executed": format_command(cmd)
    }

    # Save metadata to output directory
    metadata_file = output_dirs['base'] / "execution_metadata.json"
    save_json(metadata, metadata_file)

    print(f"📊 Generated {len(sequences)} sequences in {len(sequence_files)} files")
    print(f"💾 Metadata saved to: {metadata_file}")

    return {
        "output_dir": str(output_dirs['base']),
        "sequence_files": [str(f) for f in sequence_files],
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
        help='Output directory path (default: results/simple_design)'
    )
    parser.add_argument(
        '--chains',
        default=DEFAULT_CONFIG["chains"],
        help=f'Chains to design (default: {DEFAULT_CONFIG["chains"]})'
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
        'chains': args.chains,
        'num_sequences': args.num_sequences,
        'temperature': args.temperature,
        'seed': args.seed,
        'model': args.model,
        'use_soluble': args.use_soluble
    })

    # Run
    try:
        result = run_simple_design(
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