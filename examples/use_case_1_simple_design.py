#!/usr/bin/env python3
"""
ProteinMPNN Use Case 1: Simple Protein Sequence Design

This script takes a single PDB file and generates new sequences while keeping the backbone structure fixed.
This is the most basic use case of ProteinMPNN - scaffold-based sequence generation.
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
        description="Generate protein sequences for a given PDB structure using ProteinMPNN"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb",
        help="Path to input PDB file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="examples/outputs/simple_design",
        help="Output directory for generated sequences"
    )
    parser.add_argument(
        "--chains",
        type=str,
        default="A B",
        help="Chains to design (space-separated, e.g., 'A B')"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=3,
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

def run_proteinmpnn(args):
    """Run ProteinMPNN with the specified parameters."""

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Prepare command
    cmd = [
        "python", str(repo_dir / "protein_mpnn_run.py"),
        "--pdb_path", args.input,
        "--pdb_path_chains", args.chains,
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

    print(f"Running ProteinMPNN with command:")
    print(" ".join(cmd))
    print("-" * 50)

    # Run the command
    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ ProteinMPNN completed successfully!")
            print(f"📁 Output saved to: {args.output}")

            # Show a preview of the generated sequences
            output_files = list(Path(args.output).glob("*.fa"))
            if output_files:
                print(f"\n📄 Generated {len(output_files)} sequence files:")
                for file in output_files:
                    print(f"  - {file.name}")

                # Show first few lines of the first output file
                with open(output_files[0], 'r') as f:
                    lines = f.readlines()[:10]
                    print(f"\n📋 Preview of {output_files[0].name}:")
                    for line in lines:
                        print(f"  {line.rstrip()}")
                    if len(lines) >= 10:
                        print("  ...")

        else:
            print("❌ ProteinMPNN failed!")
            print("Error output:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running ProteinMPNN: {e}")
        return False

    return True

def main():
    args = setup_args()

    print("🧬 ProteinMPNN Simple Sequence Design")
    print("=" * 50)
    print(f"📁 Input PDB: {args.input}")
    print(f"📁 Output directory: {args.output}")
    print(f"🔗 Chains to design: {args.chains}")
    print(f"🎯 Number of sequences: {args.num_sequences}")
    print(f"🌡️ Temperature: {args.temperature}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🧠 Model: {args.model}")
    print(f"🧪 Soluble model: {args.use_soluble}")
    print()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1

    # Run ProteinMPNN
    success = run_proteinmpnn(args)

    if success:
        print("\n✅ Sequence generation completed!")
        print(f"📁 Check the output directory for results: {args.output}")
        return 0
    else:
        print("\n❌ Sequence generation failed!")
        return 1

if __name__ == "__main__":
    exit(main())