#!/usr/bin/env python3
"""
ProteinMPNN Use Case 2: Protein Sequence Scoring (Likelihood Calculation)

This script evaluates the likelihood/probability of existing protein sequences given their backbone structure.
This is useful for assessing sequence-structure compatibility and design quality.
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
        description="Score protein sequences using ProteinMPNN likelihood calculation"
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
        default="examples/outputs/sequence_scoring",
        help="Output directory for scoring results"
    )
    parser.add_argument(
        "--chains",
        type=str,
        default="A B",
        help="Chains to score (space-separated, e.g., 'A B')"
    )
    parser.add_argument(
        "--fasta_sequences",
        type=str,
        default="",
        help="Optional: FASTA format sequences to score (e.g., 'GGGGGG/PPPPS' for chains A/B). If not provided, scores native sequence."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling (affects probability calculation)"
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
    parser.add_argument(
        "--save_probs",
        action="store_true",
        help="Save per-residue probabilities to file"
    )
    return parser.parse_args()

def run_proteinmpnn_scoring(args):
    """Run ProteinMPNN in scoring mode."""

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Prepare command
    cmd = [
        "python", str(repo_dir / "protein_mpnn_run.py"),
        "--pdb_path", args.input,
        "--pdb_path_chains", args.chains,
        "--out_folder", args.output,
        "--score_only", "1",
        "--save_score", "1",
        "--sampling_temp", str(args.temperature),
        "--seed", str(args.seed),
        "--model_name", args.model,
        "--batch_size", "1"
    ]

    # Add fasta sequences if provided
    if args.fasta_sequences:
        cmd.extend(["--path_to_fasta", args.fasta_sequences])

    # Add probability saving if requested
    if args.save_probs:
        cmd.extend(["--save_probs", "1"])

    # Add model path based on type
    if args.use_soluble:
        model_path = Path(__file__).parent / "data" / "soluble_model_weights"
        cmd.extend(["--use_soluble_model", "--path_to_model_weights", str(model_path)])
    else:
        model_path = Path(__file__).parent / "data" / "vanilla_model_weights"
        cmd.extend(["--path_to_model_weights", str(model_path)])

    print(f"Running ProteinMPNN scoring with command:")
    print(" ".join(cmd))
    print("-" * 50)

    # Run the command
    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ ProteinMPNN scoring completed successfully!")
            print(f"📁 Output saved to: {args.output}")

            # Show scoring results
            output_files = list(Path(args.output).glob("*.fa"))
            score_files = list(Path(args.output).glob("*.npz"))
            prob_files = list(Path(args.output).glob("*probs*.npz"))

            if output_files:
                print(f"\n📄 Generated {len(output_files)} result files:")
                for file in output_files:
                    print(f"  - {file.name}")

                # Show scoring results from the FASTA file
                with open(output_files[0], 'r') as f:
                    lines = f.readlines()
                    print(f"\n📋 Scoring results from {output_files[0].name}:")
                    for i, line in enumerate(lines):
                        line = line.rstrip()
                        if line.startswith('>'):
                            print(f"  {line}")
                            # Print scores from the header
                            if 'score=' in line:
                                import re
                                score_match = re.search(r'score=([0-9.]+)', line)
                                global_score_match = re.search(r'global_score=([0-9.]+)', line)
                                if score_match:
                                    print(f"    🎯 Local score: {score_match.group(1)}")
                                if global_score_match:
                                    print(f"    🌍 Global score: {global_score_match.group(1)}")
                        elif i < 10:  # Show first sequence
                            print(f"  {line}")

            if score_files:
                print(f"\n💾 Generated {len(score_files)} score files (.npz format)")

            if prob_files:
                print(f"\n📊 Generated {len(prob_files)} probability files")

        else:
            print("❌ ProteinMPNN scoring failed!")
            print("Error output:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running ProteinMPNN: {e}")
        return False

    return True

def main():
    args = setup_args()

    print("🧬 ProteinMPNN Sequence Scoring & Likelihood Calculation")
    print("=" * 60)
    print(f"📁 Input PDB: {args.input}")
    print(f"📁 Output directory: {args.output}")
    print(f"🔗 Chains to score: {args.chains}")
    print(f"🧬 Custom sequences: {args.fasta_sequences if args.fasta_sequences else 'None (using native sequence)'}")
    print(f"🌡️ Temperature: {args.temperature}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🧠 Model: {args.model}")
    print(f"🧪 Soluble model: {args.use_soluble}")
    print(f"📊 Save probabilities: {args.save_probs}")
    print()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1

    # Run ProteinMPNN scoring
    success = run_proteinmpnn_scoring(args)

    if success:
        print("\n✅ Sequence scoring completed!")
        print(f"📁 Check the output directory for results: {args.output}")
        print("\n📝 Interpretation:")
        print("  - Lower scores indicate higher likelihood (better sequences)")
        print("  - Local score: average over designed residues")
        print("  - Global score: average over all residues")
        return 0
    else:
        print("\n❌ Sequence scoring failed!")
        return 1

if __name__ == "__main__":
    exit(main())