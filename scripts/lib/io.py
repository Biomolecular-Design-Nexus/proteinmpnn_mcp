"""
Shared I/O functions for ProteinMPNN MCP scripts.

These functions are extracted and simplified from the ProteinMPNN repository
to minimize dependencies while maintaining functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Any, Optional


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary containing JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Dictionary to save as JSON
        file_path: Path to save JSON file
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_fasta(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load FASTA file and parse sequences.

    Args:
        file_path: Path to FASTA file

    Returns:
        List of dictionaries with 'header' and 'sequence' keys

    Example:
        >>> sequences = load_fasta("output.fa")
        >>> print(sequences[0]['header'])
        >>> print(sequences[0]['sequence'])
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    sequences = []
    current_header = None
    current_sequence = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header is not None:
                    sequences.append({
                        'header': current_header,
                        'sequence': ''.join(current_sequence)
                    })

                # Start new sequence
                current_header = line[1:]  # Remove '>'
                current_sequence = []
            else:
                current_sequence.append(line)

        # Save last sequence
        if current_header is not None:
            sequences.append({
                'header': current_header,
                'sequence': ''.join(current_sequence)
            })

    return sequences


def save_fasta(sequences: List[Dict[str, str]], file_path: Union[str, Path]) -> None:
    """Save sequences to FASTA file.

    Args:
        sequences: List of dictionaries with 'header' and 'sequence' keys
        file_path: Path to save FASTA file

    Example:
        >>> sequences = [{'header': 'seq1', 'sequence': 'MKTAYIAKQR...'}]
        >>> save_fasta(sequences, "output.fa")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        for seq in sequences:
            f.write(f">{seq['header']}\n")
            f.write(f"{seq['sequence']}\n")


def parse_fasta_sequences(fasta_string: str) -> List[str]:
    """Parse FASTA format string into list of sequences.

    This function is inlined from repo/ProteinMPNN/protein_mpnn_utils.py
    to handle the custom format used by ProteinMPNN command line arguments.

    Args:
        fasta_string: FASTA format string (e.g., "GGGGGG/PPPPS")

    Returns:
        List of sequences

    Example:
        >>> sequences = parse_fasta_sequences("MKTAYIAKQR/DVFSLREMKP")
        >>> print(sequences)  # ['MKTAYIAKQR', 'DVFSLREMKP']
    """
    if not fasta_string:
        return []

    # Handle both '/' and newline separators
    sequences = []
    if '/' in fasta_string:
        sequences = fasta_string.split('/')
    else:
        # Handle multi-line FASTA format
        lines = fasta_string.strip().split('\n')
        current_seq = []
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            elif line:
                current_seq.append(line)

        if current_seq:
            sequences.append(''.join(current_seq))

    return [seq.strip() for seq in sequences if seq.strip()]