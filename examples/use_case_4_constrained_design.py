#!/usr/bin/env python3
"""
ProteinMPNN Use Case 4: Constrained Protein Design with Fixed Positions

This script demonstrates how to design proteins while keeping specific residue positions fixed.
This is useful when you want to preserve functionally important residues (e.g., active sites, binding sites).
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
        description="Design proteins with constrained/fixed positions using ProteinMPNN"
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
        default="examples/outputs/constrained_design",
        help="Output directory for generated sequences"
    )
    parser.add_argument(
        "--chains_to_design",
        type=str,
        default="A C",
        help="Chains to design (space-separated, e.g., 'A C')"
    )
    parser.add_argument(
        "--fixed_positions",
        type=str,
        default="1 2 3 4 5 6 7 8 23 25, 10 11 12 13 14 15 16 17 18 19 20 40",
        help="Fixed positions for each chain (comma-separated for different chains). Example: '1 2 3, 10 11 12' fixes positions 1,2,3 in first chain and 10,11,12 in second chain"
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

def parse_pdb_file(pdb_path, output_dir):
    """Parse a single PDB file."""

    input_dir = str(Path(pdb_path).parent)
    parsed_path = Path(output_dir) / "parsed_pdbs.jsonl"

    cmd = [
        "python", str(repo_dir / "helper_scripts" / "parse_multiple_chains.py"),
        "--input_path", input_dir,
        "--output_path", str(parsed_path)
    ]

    print(f"Parsing PDB file: {pdb_path}")

    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully parsed PDB file")
            return str(parsed_path)
        else:
            print("❌ PDB parsing failed!")
            print(result.stderr)
            return None
    except Exception as e:
        print(f"❌ Error parsing PDB: {e}")
        return None

def assign_chains(parsed_path, output_dir, chains_to_design):
    """Assign which chains to design."""

    assigned_path = Path(output_dir) / "assigned_pdbs.jsonl"

    cmd = [
        "python", str(repo_dir / "helper_scripts" / "assign_fixed_chains.py"),
        "--input_path", parsed_path,
        "--output_path", str(assigned_path),
        "--chain_list", chains_to_design
    ]

    print(f"Assigning chains to design: {chains_to_design}")

    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully assigned chains")
            return str(assigned_path)
        else:
            print("❌ Chain assignment failed!")
            print(result.stderr)
            return None
    except Exception as e:
        print(f"❌ Error assigning chains: {e}")
        return None

def create_fixed_positions(parsed_path, output_dir, chains_to_design, fixed_positions):
    """Create fixed positions dictionary."""

    fixed_path = Path(output_dir) / "fixed_pdbs.jsonl"

    cmd = [
        "python", str(repo_dir / "helper_scripts" / "make_fixed_positions_dict.py"),
        "--input_path", parsed_path,
        "--output_path", str(fixed_path),
        "--chain_list", chains_to_design,
        "--position_list", fixed_positions
    ]

    print(f"Creating fixed positions: {fixed_positions}")

    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully created fixed positions")
            return str(fixed_path)
        else:
            print("❌ Fixed positions creation failed!")
            print(result.stderr)
            return None
    except Exception as e:
        print(f"❌ Error creating fixed positions: {e}")
        return None

def run_constrained_proteinmpnn(args, parsed_path, assigned_path, fixed_path):
    """Run ProteinMPNN with constraints."""

    # Prepare command
    cmd = [
        "python", str(repo_dir / "protein_mpnn_run.py"),
        "--jsonl_path", parsed_path,
        "--chain_id_jsonl", assigned_path,
        "--fixed_positions_jsonl", fixed_path,
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

    print(f"Running constrained ProteinMPNN...")
    print(" ".join(cmd))
    print("-" * 50)

    # Run the command
    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Constrained ProteinMPNN completed successfully!")
            print(f"📁 Output saved to: {args.output}")

            # Show results
            output_files = list(Path(args.output).glob("*.fa"))
            if output_files:
                print(f"\n📄 Generated {len(output_files)} sequence files:")
                for file in output_files:
                    print(f"  - {file.name}")

                # Show first few lines of the first output file
                with open(output_files[0], 'r') as f:
                    lines = f.readlines()[:15]
                    print(f"\n📋 Preview of {output_files[0].name}:")
                    for i, line in enumerate(lines):
                        line = line.rstrip()
                        print(f"  {line}")
                        if i >= 10:
                            break
                    if len(lines) > 15:
                        print("  ...")

        else:
            print("❌ Constrained ProteinMPNN failed!")
            print("Error output:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running ProteinMPNN: {e}")
        return False

    return True

def main():
    args = setup_args()

    print("🧬 ProteinMPNN Constrained Design with Fixed Positions")
    print("=" * 60)
    print(f"📁 Input PDB: {args.input}")
    print(f"📁 Output directory: {args.output}")
    print(f"🔗 Chains to design: {args.chains_to_design}")
    print(f"🔒 Fixed positions: {args.fixed_positions}")
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

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Step 1: Parse PDB file
    parsed_path = parse_pdb_file(args.input, args.output)
    if not parsed_path:
        return 1

    # Step 2: Assign chains to design
    assigned_path = assign_chains(parsed_path, args.output, args.chains_to_design)
    if not assigned_path:
        return 1

    # Step 3: Create fixed positions dictionary
    fixed_path = create_fixed_positions(parsed_path, args.output, args.chains_to_design, args.fixed_positions)
    if not fixed_path:
        return 1

    # Step 4: Run constrained ProteinMPNN
    success = run_constrained_proteinmpnn(args, parsed_path, assigned_path, fixed_path)

    if success:
        print("\n✅ Constrained design completed!")
        print(f"📁 Check the output directory for results: {args.output}")
        print("\n📝 Note: Fixed positions are preserved in the generated sequences")
        print("  while other positions are optimized by ProteinMPNN")
        return 0
    else:
        print("\n❌ Constrained design failed!")
        return 1

if __name__ == "__main__":
    exit(main())