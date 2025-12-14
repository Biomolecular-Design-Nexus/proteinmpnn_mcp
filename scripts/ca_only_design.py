#!/usr/bin/env python3
"""
Script: ca_only_design.py
Description: Design sequences using only carbon alpha atoms (backbone traces)

Original Use Case: examples/use_case_5_ca_only_design.py
Dependencies Removed: Simplified CA model handling and path management

Usage:
    python scripts/ca_only_design.py --input <input_file> --output <output_dir>

Example:
    python scripts/ca_only_design.py --input examples/data/inputs/PDB_complexes/3HTN.pdb --output results/ca_only
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
    "batch_size": 1
}

VALID_MODELS = ["v_48_002", "v_48_010", "v_48_020"]  # Note: CA models only have 3 versions

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_ca_only_design(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Design sequences using only carbon alpha atoms (backbone traces).

    This function uses the CA-ProteinMPNN model which is specialized for
    designing proteins from carbon alpha traces (low-resolution structures).

    Args:
        input_file: Path to input PDB file (can be full atom or CA-only)
        output_dir: Directory to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - output_dir: Path to output directory
            - sequence_files: List of generated sequence files
            - sequences: Generated sequence data
            - metadata: Execution metadata

    Example:
        >>> result = run_ca_only_design("input.pdb", "output/")
        >>> print(f"Generated {len(result['sequences'])} CA-based sequences")
    """
    # Setup
    input_file = check_file_exists(input_file, "PDB file")
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate configuration
    chains_list = validate_chains(config["chains"])
    if config["model"] not in VALID_MODELS:
        raise ValueError(f"Invalid CA model: {config['model']}. Must be one of {VALID_MODELS}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("results") / "ca_only_design"
    # Convert to absolute path to ensure correct location when subprocess runs with different cwd
    output_dir = Path(output_dir).resolve()
    output_dirs = create_output_directories(output_dir)

    # Get paths
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    # CA models are always in ca_model_weights directory
    model_path = get_model_path(script_dir, use_soluble=False, ca_only=True)

    # Prepare ProteinMPNN command with CA-specific flags
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
        "--path_to_model_weights", str(model_path),
        "--ca_only"  # Enable CA-only mode
    ]

    # Execute ProteinMPNN
    print(f"🧬 Running ProteinMPNN CA-Only Design")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_dirs['base']}")
    print(f"🔗 Chains: {config['chains']}")
    print(f"🎯 Sequences: {config['num_sequences']}")
    print(f"🧠 Model: {config['model']} (CA-only)")
    print(f"⚠️  Using carbon alpha coordinates only")
    print(f"💻 Command: {format_command(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ProteinMPNN CA-only failed: {result.stderr}")

        print("✅ ProteinMPNN CA-only design completed successfully!")

    except Exception as e:
        raise RuntimeError(f"Error running ProteinMPNN CA-only: {e}")

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
                    'length': len(seq_data['sequence']),
                    'design_method': 'ca_only'
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
        "model_type": "ca_only",
        "design_method": "CA-ProteinMPNN (carbon alpha only)",
        "command_executed": format_command(cmd),
        "notes": "This design used only carbon alpha coordinates from the backbone structure"
    }

    # Save metadata to output directory
    metadata_file = output_dirs['base'] / "execution_metadata.json"
    save_json(metadata, metadata_file)

    print(f"\n📊 CA-only design summary:")
    print(f"  🧬 Sequences generated: {len(sequences)}")
    print(f"  📄 Sequence files: {len(sequence_files)}")
    print(f"  🦴 Design method: CA-only (backbone traces)")
    print(f"💾 Metadata saved to: {metadata_file}")

    # Show sequence statistics
    if sequences:
        lengths = [seq['length'] for seq in sequences]
        scores = [seq['scores'].get('score', 0) for seq in sequences if seq['scores']]

        print(f"\n📈 Sequence statistics:")
        print(f"  - Length range: {min(lengths)}-{max(lengths)} residues")
        if scores:
            print(f"  - Score range: {min(scores):.3f}-{max(scores):.3f}")

        # Show first sequence as example
        first_seq = sequences[0]
        print(f"\n🧬 Example sequence:")
        print(f"  Header: {first_seq['header']}")
        print(f"  Length: {first_seq['length']} residues")
        if len(first_seq['sequence']) <= 60:
            print(f"  Sequence: {first_seq['sequence']}")
        else:
            print(f"  Sequence: {first_seq['sequence'][:30]}...{first_seq['sequence'][-30:]}")

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
        help='Input PDB file path (full atom or CA-only)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output directory path (default: results/ca_only_design)'
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
        help=f'CA model version (default: {DEFAULT_CONFIG["model"]})'
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
        'model': args.model
    })

    # Run
    try:
        result = run_ca_only_design(
            input_file=args.input,
            output_dir=args.output,
            config=config
        )

        print(f"\n✅ Success: {result['output_dir']}")
        print(f"📄 Generated {len(result['sequence_files'])} sequence files:")
        for seq_file in result['sequence_files']:
            print(f"  - {seq_file}")

        print(f"\n💡 Note: These sequences were designed using only carbon alpha coordinates")
        print(f"   This is useful for low-resolution structures or backbone traces.")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())