#!/usr/bin/env python3
"""
ProteinMPNN Use Case 5: CA-only Protein Design

This script demonstrates ProteinMPNN's CA-only mode, which uses only carbon alpha (CA) atoms
for sequence design. This is useful when you only have backbone traces or low-resolution structures.
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
        description="Generate protein sequences using CA-only ProteinMPNN model"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="examples/data/inputs/PDB_complexes/pdbs/3HTN.pdb",
        help="Path to input PDB file (can contain full atom or CA-only structure)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="examples/outputs/ca_only_design",
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
        choices=["v_48_002", "v_48_010", "v_48_020"],
        help="CA-ProteinMPNN model version to use"
    )
    return parser.parse_args()

def run_ca_only_proteinmpnn(args):
    """Run ProteinMPNN in CA-only mode."""

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Prepare command
    cmd = [
        "python", str(repo_dir / "protein_mpnn_run.py"),
        "--pdb_path", args.input,
        "--pdb_path_chains", args.chains,
        "--out_folder", args.output,
        "--ca_only",  # Enable CA-only mode
        "--num_seq_per_target", str(args.num_sequences),
        "--sampling_temp", str(args.temperature),
        "--seed", str(args.seed),
        "--model_name", args.model,
        "--batch_size", "1"
    ]

    # CA-only models are stored in a different directory
    model_path = Path(__file__).parent / "data" / "ca_model_weights"
    cmd.extend(["--path_to_model_weights", str(model_path)])

    print(f"Running CA-only ProteinMPNN with command:")
    print(" ".join(cmd))
    print("-" * 50)

    # Run the command
    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ CA-only ProteinMPNN completed successfully!")
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
                        line = line.rstrip()
                        print(f"  {line}")
                        # Show model info from header
                        if 'CA_model_name' in line:
                            print(f"    🧠 Used CA model: {line.split('CA_model_name=')[1].split(',')[0]}")
                    if len(lines) >= 10:
                        print("  ...")

        else:
            print("❌ CA-only ProteinMPNN failed!")
            print("Error output:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running ProteinMPNN: {e}")
        return False

    return True

def validate_ca_models():
    """Check if CA-only model weights are available."""
    model_path = Path(__file__).parent / "data" / "ca_model_weights"

    if not model_path.exists():
        print(f"❌ CA model weights directory not found: {model_path}")
        return False

    # Check for common model files
    model_files = list(model_path.glob("*.pt"))
    if not model_files:
        print(f"❌ No CA model weight files (.pt) found in {model_path}")
        return False

    print(f"✅ Found {len(model_files)} CA model weight files:")
    for model_file in sorted(model_files):
        print(f"  - {model_file.name}")

    return True

def main():
    args = setup_args()

    print("🧬 ProteinMPNN CA-only Sequence Design")
    print("=" * 50)
    print("ℹ️ CA-only mode uses only carbon alpha atoms for design")
    print("  Useful for backbone traces or low-resolution structures")
    print()
    print(f"📁 Input PDB: {args.input}")
    print(f"📁 Output directory: {args.output}")
    print(f"🔗 Chains to design: {args.chains}")
    print(f"🎯 Number of sequences: {args.num_sequences}")
    print(f"🌡️ Temperature: {args.temperature}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🧠 CA Model: {args.model}")
    print()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1

    # Validate CA models are available
    if not validate_ca_models():
        return 1

    # Run CA-only ProteinMPNN
    success = run_ca_only_proteinmpnn(args)

    if success:
        print("\n✅ CA-only sequence generation completed!")
        print(f"📁 Check the output directory for results: {args.output}")
        print("\n📝 Note: CA-only models may produce different results")
        print("  compared to full-atom models due to reduced structural information")
        return 0
    else:
        print("\n❌ CA-only sequence generation failed!")
        return 1

if __name__ == "__main__":
    exit(main())