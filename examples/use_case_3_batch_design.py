#!/usr/bin/env python3
"""
ProteinMPNN Use Case 3: Batch Protein Design

This script processes multiple PDB files in a directory and generates sequences for all of them.
Useful for high-throughput protein design workflows.
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

# Add the ProteinMPNN repository to the path
repo_dir = Path(__file__).parent.parent / "repo" / "ProteinMPNN"
sys.path.append(str(repo_dir))

def setup_args():
    parser = argparse.ArgumentParser(
        description="Batch process multiple PDB files for protein sequence design"
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default="examples/data/inputs/PDB_monomers/pdbs",
        help="Directory containing PDB files to process"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="examples/outputs/batch_design",
        help="Output directory for generated sequences"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=2,
        help="Number of sequences to generate per target"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0.1-0.3 recommended)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=37,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="v_48_020",
        choices=["v_48_002", "v_48_010", "v_48_020", "v_48_030"],
        help="ProteinMPNN model version to use"
    )
    parser.add_argument(
        "--use_soluble",
        action="store_true",
        help="Use soluble protein model weights"
    )
    return parser.parse_args()

def parse_pdbs(input_dir, output_dir):
    """Parse multiple PDB files using ProteinMPNN helper script."""

    parsed_path = Path(output_dir) / "parsed_pdbs.jsonl"

    cmd = [
        "python", str(repo_dir / "helper_scripts" / "parse_multiple_chains.py"),
        "--input_path", input_dir,
        "--output_path", str(parsed_path)
    ]

    print(f"Parsing PDB files from {input_dir}...")
    print(" ".join(cmd))

    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Successfully parsed PDB files")
            print(f"📁 Parsed data saved to: {parsed_path}")
            return str(parsed_path)
        else:
            print("❌ PDB parsing failed!")
            print("Error output:")
            print(result.stderr)
            return None
    except Exception as e:
        print(f"❌ Error parsing PDBs: {e}")
        return None

def run_batch_proteinmpnn(args, parsed_path):
    """Run ProteinMPNN on parsed PDB files."""

    # Prepare command
    cmd = [
        "python", str(repo_dir / "protein_mpnn_run.py"),
        "--jsonl_path", parsed_path,
        "--out_folder", args.output,
        "--num_seq_per_target", str(args.num_sequences),
        "--sampling_temp", str(args.temperature),
        "--seed", str(args.seed),
        "--model_name", args.model,
        "--batch_size", "1"
    ]

    # Add model path based on type
    if args.use_soluble:
        model_path = Path(__file__).parent / "data" / "soluble_model_weights"
        cmd.extend(["--use_soluble_model", "--path_to_model_weights", str(model_path)])
    else:
        model_path = Path(__file__).parent / "data" / "vanilla_model_weights"
        cmd.extend(["--path_to_model_weights", str(model_path)])

    print(f"Running ProteinMPNN batch processing...")
    print(" ".join(cmd))
    print("-" * 50)

    # Run the command
    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ ProteinMPNN batch processing completed successfully!")
            print(f"📁 Output saved to: {args.output}")

            # Show a summary of results
            output_files = list(Path(args.output).glob("*.fa"))
            if output_files:
                print(f"\n📄 Generated {len(output_files)} sequence files:")

                # Group files by target
                targets = {}
                for file in output_files:
                    # Extract target name from filename
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        target = parts[0]
                        if target not in targets:
                            targets[target] = []
                        targets[target].append(file.name)

                for target, files in targets.items():
                    print(f"  🎯 {target}: {len(files)} files")
                    for file in sorted(files)[:3]:  # Show first 3 files
                        print(f"    - {file}")
                    if len(files) > 3:
                        print(f"    ... and {len(files) - 3} more")

        else:
            print("❌ ProteinMPNN batch processing failed!")
            print("Error output:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running ProteinMPNN: {e}")
        return False

    return True

def main():
    args = setup_args()

    print("🧬 ProteinMPNN Batch Protein Design")
    print("=" * 50)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output}")
    print(f"🎯 Number of sequences per target: {args.num_sequences}")
    print(f"🌡️ Temperature: {args.temperature}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🧠 Model: {args.model}")
    print(f"🧪 Soluble model: {args.use_soluble}")
    print()

    # Check if input directory exists
    if not Path(args.input_dir).exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1

    # Check for PDB files in input directory
    pdb_files = list(Path(args.input_dir).glob("*.pdb"))
    if not pdb_files:
        print(f"❌ No PDB files found in: {args.input_dir}")
        return 1

    print(f"🔍 Found {len(pdb_files)} PDB files to process:")
    for pdb_file in sorted(pdb_files):
        print(f"  - {pdb_file.name}")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Step 1: Parse PDB files
    parsed_path = parse_pdbs(args.input_dir, args.output)
    if not parsed_path:
        return 1

    # Step 2: Run ProteinMPNN
    success = run_batch_proteinmpnn(args, parsed_path)

    if success:
        print("\n✅ Batch processing completed!")
        print(f"📁 Check the output directory for results: {args.output}")
        return 0
    else:
        print("\n❌ Batch processing failed!")
        return 1

if __name__ == "__main__":
    exit(main())