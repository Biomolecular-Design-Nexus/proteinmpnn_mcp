#!/usr/bin/env python3
"""
Script: sequence_scoring.py
Description: Score protein sequences using ProteinMPNN likelihood calculation

Original Use Case: examples/use_case_2_sequence_scoring.py
Dependencies Removed: Simplified scoring output parsing and path management

Usage:
    python scripts/sequence_scoring.py --input <input_file> --output <output_dir>

Example:
    python scripts/sequence_scoring.py --input examples/data/inputs/PDB_complexes/3HTN.pdb --output results/scoring
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
import numpy as np

# Local imports from shared library
from lib.io import load_fasta, save_json, parse_fasta_sequences
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
    "fasta_sequences": "",  # Empty means score native sequence
    "temperature": 0.1,
    "seed": 37,
    "model": "v_48_020",
    "use_soluble": False,
    "save_probs": False,
    "batch_size": 1
}

VALID_MODELS = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

# ==============================================================================
# Utility Functions (inlined from repo)
# ==============================================================================
def load_score_data(npz_file: Path) -> Dict[str, Any]:
    """Load and parse ProteinMPNN score data from NPZ file.

    Args:
        npz_file: Path to NPZ file containing scores

    Returns:
        Dictionary with parsed score data
    """
    try:
        data = np.load(npz_file, allow_pickle=True)

        # Extract available data from NPZ file
        result = {"file": str(npz_file)}

        for key in data.files:
            try:
                array_data = data[key]
                if isinstance(array_data, np.ndarray):
                    if array_data.size == 1:
                        # Single value
                        result[key] = float(array_data.item()) if array_data.dtype.kind in 'fc' else array_data.item()
                    elif array_data.ndim == 1 and array_data.size <= 10:
                        # Small array - convert to list
                        result[key] = array_data.tolist()
                    else:
                        # Large array - store shape and dtype info
                        result[key] = {
                            "shape": array_data.shape,
                            "dtype": str(array_data.dtype),
                            "size": array_data.size
                        }
                else:
                    result[key] = str(array_data)

            except Exception as e:
                result[key] = f"Error loading: {e}"

        return result

    except Exception as e:
        return {"file": str(npz_file), "error": f"Could not load NPZ file: {e}"}

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_sequence_scoring(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Score protein sequences using ProteinMPNN likelihood calculation.

    Args:
        input_file: Path to input PDB file
        output_dir: Directory to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - output_dir: Path to output directory
            - score_files: List of generated score files (NPZ format)
            - sequence_files: List of generated FASTA files
            - scores: Parsed score data
            - metadata: Execution metadata

    Example:
        >>> result = run_sequence_scoring("input.pdb", "output/")
        >>> print(result['scores'])
    """
    # Setup
    input_file = check_file_exists(input_file, "PDB file")
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate configuration
    chains_list = validate_chains(config["chains"])
    if config["model"] not in VALID_MODELS:
        raise ValueError(f"Invalid model: {config['model']}. Must be one of {VALID_MODELS}")

    # Parse custom sequences if provided
    custom_sequences = parse_fasta_sequences(config["fasta_sequences"]) if config["fasta_sequences"] else []

    # Setup output directory
    if output_dir is None:
        output_dir = Path("results") / "sequence_scoring"
    # Convert to absolute path to ensure correct location when subprocess runs with different cwd
    output_dir = Path(output_dir).resolve()
    output_dirs = create_output_directories(output_dir)

    # Get paths
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    model_path = get_model_path(script_dir, config["use_soluble"])

    # Prepare ProteinMPNN command
    # Use sys.executable to ensure we use the same Python interpreter (with torch installed)
    cmd = [
        sys.executable, str(repo_path / "protein_mpnn_run.py"),
        "--pdb_path", str(input_file),
        "--pdb_path_chains", config["chains"],
        "--out_folder", str(output_dirs['base']),
        "--score_only", "1",
        "--save_score", "1",
        "--sampling_temp", str(config["temperature"]),
        "--seed", str(config["seed"]),
        "--model_name", config["model"],
        "--batch_size", str(config["batch_size"]),
        "--path_to_model_weights", str(model_path)
    ]

    # Add custom sequences if provided
    if config["fasta_sequences"]:
        cmd.extend(["--path_to_fasta", config["fasta_sequences"]])

    # Add probability saving if requested
    if config["save_probs"]:
        cmd.extend(["--save_probs", "1"])

    # Add soluble model flag if needed
    if config["use_soluble"]:
        cmd.append("--use_soluble_model")

    # Execute ProteinMPNN
    print(f"🧬 Running ProteinMPNN Sequence Scoring")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_dirs['base']}")
    print(f"🔗 Chains: {config['chains']}")
    print(f"🧬 Custom sequences: {'Yes' if custom_sequences else 'No (native sequence)'}")
    print(f"📊 Save probabilities: {config['save_probs']}")
    print(f"🧠 Model: {config['model']}")
    print(f"💻 Command: {format_command(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ProteinMPNN failed: {result.stderr}")

        print("✅ ProteinMPNN scoring completed successfully!")

    except Exception as e:
        raise RuntimeError(f"Error running ProteinMPNN: {e}")

    # Process results
    sequence_files = list(output_dirs['base'].glob("**/*.fa"))
    score_files = list(output_dirs['base'].glob("**/*.npz"))
    prob_files = [f for f in score_files if "probs" in f.name]

    # Parse sequence results
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
                    'length': len(seq_data['sequence'])
                })
        except Exception as e:
            print(f"Warning: Could not parse sequence file {seq_file}: {e}")

    # Parse score files
    score_data = []
    for score_file in score_files:
        try:
            data = load_score_data(score_file)
            score_data.append(data)
        except Exception as e:
            print(f"Warning: Could not parse score file {score_file}: {e}")
            score_data.append({"file": str(score_file), "error": str(e)})

    # Save execution metadata
    metadata = {
        "input_file": str(input_file),
        "output_directory": str(output_dirs['base']),
        "config": config,
        "chains_scored": chains_list,
        "custom_sequences": custom_sequences,
        "sequences_processed": len(sequences),
        "score_files_generated": len(score_files),
        "probability_files_generated": len(prob_files),
        "sequence_files": [str(f) for f in sequence_files],
        "score_files": [str(f) for f in score_files],
        "model_path": str(model_path),
        "command_executed": format_command(cmd)
    }

    # Save metadata to output directory
    metadata_file = output_dirs['base'] / "execution_metadata.json"
    save_json(metadata, metadata_file)

    print(f"📊 Processed {len(sequences)} sequences")
    print(f"💾 Generated {len(score_files)} score files")
    if prob_files:
        print(f"📈 Generated {len(prob_files)} probability files")
    print(f"💾 Metadata saved to: {metadata_file}")

    return {
        "output_dir": str(output_dirs['base']),
        "score_files": [str(f) for f in score_files],
        "sequence_files": [str(f) for f in sequence_files],
        "scores": score_data,
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
        help='Output directory path (default: results/sequence_scoring)'
    )
    parser.add_argument(
        '--chains',
        default=DEFAULT_CONFIG["chains"],
        help=f'Chains to score (default: {DEFAULT_CONFIG["chains"]})'
    )
    parser.add_argument(
        '--fasta_sequences',
        default=DEFAULT_CONFIG["fasta_sequences"],
        help='Custom sequences to score (format: "SEQ1/SEQ2"). If not provided, scores native sequence'
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
        '--save_probs', action='store_true',
        help='Save per-residue probabilities to NPZ files'
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
        'fasta_sequences': args.fasta_sequences,
        'temperature': args.temperature,
        'seed': args.seed,
        'model': args.model,
        'use_soluble': args.use_soluble,
        'save_probs': args.save_probs
    })

    # Run
    try:
        result = run_sequence_scoring(
            input_file=args.input,
            output_dir=args.output,
            config=config
        )

        print(f"\n✅ Success: {result['output_dir']}")
        print(f"📄 Generated {len(result['score_files'])} score files:")
        for score_file in result['score_files']:
            print(f"  - {score_file}")

        # Display score summary
        if result['sequences']:
            print(f"\n📊 Score Summary:")
            for i, seq in enumerate(result['sequences'][:3]):  # Show first 3
                if seq['scores']:
                    print(f"  Sequence {i+1}:")
                    for score_type, value in seq['scores'].items():
                        print(f"    {score_type}: {value}")
            if len(result['sequences']) > 3:
                print(f"    ... and {len(result['sequences']) - 3} more sequences")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())