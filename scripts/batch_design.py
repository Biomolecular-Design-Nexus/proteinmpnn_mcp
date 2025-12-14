#!/usr/bin/env python3
"""
Script: batch_design.py
Description: Batch process multiple PDB files for protein sequence design

Original Use Case: examples/use_case_3_batch_design.py
Dependencies Removed: Inlined PDB parsing helper and simplified batch processing

Usage:
    python scripts/batch_design.py --input_dir <input_directory> --output <output_dir>

Example:
    python scripts/batch_design.py --input_dir examples/data/inputs/PDB_monomers/pdbs --output results/batch_design
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
import glob

# Local imports from shared library
from lib.io import load_fasta, save_json, load_json
from lib.utils import (
    validate_chains, format_command, check_file_exists,
    get_model_path, get_repo_path, create_output_directories,
    extract_scores_from_fasta_header
)

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "chains": "",  # Auto-detect chains for each PDB
    "num_sequences": 2,
    "temperature": 0.1,
    "seed": 37,
    "model": "v_48_020",
    "use_soluble": False,
    "batch_size": 1,
    "file_pattern": "*.pdb"
}

VALID_MODELS = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

# ==============================================================================
# Utility Functions (inlined from repo helper scripts)
# ==============================================================================
def parse_pdbs_in_directory(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = "*.pdb"
) -> Path:
    """Parse multiple PDB files in a directory.

    This function is inlined from repo/ProteinMPNN/helper_scripts/parse_multiple_chains.py
    to minimize dependencies on repo helper scripts.

    Args:
        input_dir: Directory containing PDB files
        output_dir: Directory to save parsed JSONL
        file_pattern: Pattern to match PDB files

    Returns:
        Path to generated parsed_pdbs.jsonl file

    Raises:
        RuntimeError: If parsing fails
    """
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    parsed_path = output_dir / "parsed_pdbs.jsonl"

    # Use the repo's parsing script
    cmd = [
        "python", str(repo_path / "helper_scripts" / "parse_multiple_chains.py"),
        "--input_path", str(input_dir),
        "--output_path", str(parsed_path)
    ]

    print(f"📁 Parsing PDB files in: {input_dir}")
    print(f"🔍 Pattern: {file_pattern}")

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"PDB parsing failed: {result.stderr}")

        print(f"✅ PDB files parsed successfully")
        return parsed_path

    except Exception as e:
        raise RuntimeError(f"Error parsing PDB files: {e}")


def find_pdb_files(input_dir: Path, file_pattern: str = "*.pdb") -> List[Path]:
    """Find PDB files in directory matching pattern.

    Args:
        input_dir: Directory to search
        file_pattern: Pattern to match files

    Returns:
        List of PDB file paths
    """
    pdb_files = list(input_dir.glob(file_pattern))

    # Also try common extensions
    if not pdb_files:
        for pattern in ["*.pdb", "*.PDB", "*.ent"]:
            pdb_files.extend(input_dir.glob(pattern))

    return sorted(pdb_files)

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_batch_design(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Batch process multiple PDB files for protein sequence design.

    Args:
        input_dir: Directory containing PDB files
        output_dir: Directory to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - output_dir: Path to output directory
            - pdb_files_processed: List of processed PDB files
            - sequence_files: List of generated sequence files
            - parsed_pdbs_file: Path to parsed PDB data file
            - sequences: Aggregated sequence data
            - metadata: Execution metadata

    Example:
        >>> result = run_batch_design("inputs/", "output/")
        >>> print(f"Processed {len(result['pdb_files_processed'])} PDB files")
        >>> print(f"Generated {len(result['sequence_files'])} sequence files")
    """
    # Setup
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate configuration
    if config["model"] not in VALID_MODELS:
        raise ValueError(f"Invalid model: {config['model']}. Must be one of {VALID_MODELS}")

    # Find PDB files
    pdb_files = find_pdb_files(input_dir, config["file_pattern"])
    if not pdb_files:
        raise ValueError(f"No PDB files found in {input_dir} matching pattern '{config['file_pattern']}'")

    print(f"🔍 Found {len(pdb_files)} PDB files to process:")
    for pdb_file in pdb_files:
        print(f"  - {pdb_file.name}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("results") / "batch_design"
    # Convert to absolute path to ensure correct location when subprocess runs with different cwd
    output_dir = Path(output_dir).resolve()
    output_dirs = create_output_directories(output_dir)

    # Get paths
    script_dir = Path(__file__).parent
    repo_path = get_repo_path(script_dir)
    model_path = get_model_path(script_dir, config["use_soluble"])

    # Parse PDB files
    print("\n🔧 Parsing PDB files...")
    parsed_pdbs_file = parse_pdbs_in_directory(input_dir, output_dirs['base'], config["file_pattern"])

    # Prepare ProteinMPNN command
    cmd = [
        "python", str(repo_path / "protein_mpnn_run.py"),
        "--jsonl_path", str(parsed_pdbs_file),
        "--out_folder", str(output_dirs['base']),
        "--num_seq_per_target", str(config["num_sequences"]),
        "--sampling_temp", str(config["temperature"]),
        "--seed", str(config["seed"]),
        "--model_name", config["model"],
        "--batch_size", str(config["batch_size"]),
        "--path_to_model_weights", str(model_path)
    ]

    # Add chains if specified (otherwise auto-detect)
    if config["chains"]:
        chains_list = validate_chains(config["chains"])
        cmd.extend(["--pdb_path_chains", config["chains"]])
    else:
        print("🔗 Auto-detecting chains for each PDB file")

    # Add soluble model flag if needed
    if config["use_soluble"]:
        cmd.append("--use_soluble_model")

    # Execute ProteinMPNN
    print(f"\n🧬 Running ProteinMPNN Batch Design")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output: {output_dirs['base']}")
    print(f"📋 PDB files: {len(pdb_files)}")
    print(f"🎯 Sequences per target: {config['num_sequences']}")
    print(f"🧠 Model: {config['model']}")
    print(f"💻 Command: {format_command(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ProteinMPNN failed: {result.stderr}")

        print("✅ ProteinMPNN batch design completed successfully!")

    except Exception as e:
        raise RuntimeError(f"Error running ProteinMPNN: {e}")

    # Process results
    sequence_files = list(output_dirs['base'].glob("**/*.fa"))
    sequences = []
    sequences_by_target = {}

    for seq_file in sequence_files:
        try:
            file_sequences = load_fasta(seq_file)
            target_name = seq_file.stem  # File name without extension

            sequences_by_target[target_name] = []

            for seq_data in file_sequences:
                scores = extract_scores_from_fasta_header(seq_data['header'])
                sequence_info = {
                    'target': target_name,
                    'file': str(seq_file),
                    'header': seq_data['header'],
                    'sequence': seq_data['sequence'],
                    'scores': scores,
                    'length': len(seq_data['sequence'])
                }
                sequences.append(sequence_info)
                sequences_by_target[target_name].append(sequence_info)

        except Exception as e:
            print(f"Warning: Could not parse sequence file {seq_file}: {e}")

    # Load parsed PDB data if available
    parsed_pdb_data = {}
    try:
        if parsed_pdbs_file.exists():
            # JSONL format - read line by line
            with open(parsed_pdbs_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            if isinstance(data, dict) and 'name' in data:
                                parsed_pdb_data[data['name']] = data
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        print(f"Warning: Could not load parsed PDB data: {e}")

    # Save execution metadata
    metadata = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dirs['base']),
        "config": config,
        "pdb_files_found": [str(f) for f in pdb_files],
        "pdb_files_processed": len(pdb_files),
        "sequences_generated_total": len(sequences),
        "sequence_files_generated": len(sequence_files),
        "sequences_by_target": {k: len(v) for k, v in sequences_by_target.items()},
        "parsed_pdbs_file": str(parsed_pdbs_file),
        "model_path": str(model_path),
        "command_executed": format_command(cmd),
        "targets_processed": list(sequences_by_target.keys())
    }

    # Save metadata to output directory
    metadata_file = output_dirs['base'] / "execution_metadata.json"
    save_json(metadata, metadata_file)

    print(f"\n📊 Batch processing summary:")
    print(f"  📋 PDB files processed: {len(pdb_files)}")
    print(f"  🎯 Targets with sequences: {len(sequences_by_target)}")
    print(f"  🧬 Total sequences generated: {len(sequences)}")
    print(f"  📄 Sequence files created: {len(sequence_files)}")
    print(f"💾 Metadata saved to: {metadata_file}")

    return {
        "output_dir": str(output_dirs['base']),
        "pdb_files_processed": [str(f) for f in pdb_files],
        "sequence_files": [str(f) for f in sequence_files],
        "parsed_pdbs_file": str(parsed_pdbs_file),
        "sequences": sequences,
        "sequences_by_target": sequences_by_target,
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
        '--input_dir', '-i', required=True,
        help='Input directory containing PDB files'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output directory path (default: results/batch_design)'
    )
    parser.add_argument(
        '--chains',
        default=DEFAULT_CONFIG["chains"],
        help='Chains to design for all PDBs (default: auto-detect per PDB)'
    )
    parser.add_argument(
        '--num_sequences', type=int,
        default=DEFAULT_CONFIG["num_sequences"],
        help=f'Number of sequences per target (default: {DEFAULT_CONFIG["num_sequences"]})'
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
        '--file_pattern',
        default=DEFAULT_CONFIG["file_pattern"],
        help=f'Pattern to match PDB files (default: {DEFAULT_CONFIG["file_pattern"]})'
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
        'use_soluble': args.use_soluble,
        'file_pattern': args.file_pattern
    })

    # Run
    try:
        result = run_batch_design(
            input_dir=args.input_dir,
            output_dir=args.output,
            config=config
        )

        print(f"\n✅ Success: {result['output_dir']}")
        print(f"📄 Generated sequence files for {len(result['sequences_by_target'])} targets:")
        for target, sequences in result['sequences_by_target'].items():
            print(f"  - {target}: {len(sequences)} sequences")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())