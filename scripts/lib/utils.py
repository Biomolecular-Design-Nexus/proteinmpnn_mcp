"""
Shared utility functions for ProteinMPNN MCP scripts.

These functions are extracted and simplified from the ProteinMPNN repository
to provide common functionality without deep dependencies.
"""

import os
import re
from pathlib import Path
from typing import List, Union, Optional, Dict, Any


def validate_chains(chains: str) -> List[str]:
    """Validate and normalize chain specification.

    Args:
        chains: Chain specification (e.g., "A B", "A,B", or "A")

    Returns:
        List of chain identifiers

    Raises:
        ValueError: If chain specification is invalid

    Example:
        >>> validate_chains("A B")
        ['A', 'B']
        >>> validate_chains("A,B,C")
        ['A', 'B', 'C']
    """
    if not chains or not chains.strip():
        raise ValueError("Chain specification cannot be empty")

    # Handle both space and comma separated chains
    if ',' in chains:
        chain_list = [c.strip() for c in chains.split(',')]
    else:
        chain_list = chains.split()

    # Validate chain identifiers
    validated_chains = []
    for chain in chain_list:
        chain = chain.strip()
        if not chain:
            continue
        if not re.match(r'^[A-Za-z0-9]$', chain):
            raise ValueError(f"Invalid chain identifier: {chain}. Chains must be single alphanumeric characters.")
        validated_chains.append(chain.upper())

    if not validated_chains:
        raise ValueError("No valid chains specified")

    return validated_chains


def format_command(cmd_parts: List[str]) -> str:
    """Format command parts into a readable command string.

    Args:
        cmd_parts: List of command components

    Returns:
        Formatted command string
    """
    return ' '.join(f'"{part}"' if ' ' in part else part for part in cmd_parts)


def check_file_exists(file_path: Union[str, Path], file_type: str = "file") -> Path:
    """Check if file exists and return Path object.

    Args:
        file_path: Path to file
        file_type: Description of file type for error messages

    Returns:
        Path object if file exists

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{file_type} not found: {path_obj}")
    return path_obj


def parse_fixed_positions(fixed_positions: str) -> Dict[int, List[int]]:
    """Parse fixed positions string into chain-specific position lists.

    Args:
        fixed_positions: Position specification (e.g., "1 2 3, 10 11 12")

    Returns:
        Dictionary mapping chain index to position list

    Example:
        >>> parse_fixed_positions("1 2 3, 10 11 12")
        {0: [1, 2, 3], 1: [10, 11, 12]}
    """
    if not fixed_positions or not fixed_positions.strip():
        return {}

    result = {}
    chain_positions = fixed_positions.split(',')

    for chain_idx, positions_str in enumerate(chain_positions):
        positions_str = positions_str.strip()
        if not positions_str:
            continue

        try:
            positions = [int(p.strip()) for p in positions_str.split() if p.strip()]
            if positions:
                result[chain_idx] = positions
        except ValueError as e:
            raise ValueError(f"Invalid position specification for chain {chain_idx}: {positions_str}") from e

    return result


def get_model_path(script_dir: Path, use_soluble: bool = False, ca_only: bool = False) -> Path:
    """Get model weights path based on model type.

    Args:
        script_dir: Directory containing the script (for relative path calculation)
        use_soluble: Whether to use soluble protein model weights
        ca_only: Whether to use CA-only model weights

    Returns:
        Path to model weights directory

    Raises:
        FileNotFoundError: If model weights directory doesn't exist
    """
    # Navigate to MCP root directory
    mcp_root = script_dir.parent

    if ca_only:
        model_dir = mcp_root / "examples" / "data" / "ca_model_weights"
    elif use_soluble:
        model_dir = mcp_root / "examples" / "data" / "soluble_model_weights"
    else:
        model_dir = mcp_root / "examples" / "data" / "vanilla_model_weights"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model weights directory not found: {model_dir}")

    return model_dir


def get_repo_path(script_dir: Path) -> Path:
    """Get ProteinMPNN repository path.

    Args:
        script_dir: Directory containing the script

    Returns:
        Path to ProteinMPNN repository

    Raises:
        FileNotFoundError: If repository doesn't exist
    """
    repo_path = script_dir.parent / "repo" / "ProteinMPNN"
    if not repo_path.exists():
        raise FileNotFoundError(f"ProteinMPNN repository not found: {repo_path}")
    return repo_path


def extract_scores_from_fasta_header(header: str) -> Optional[Dict[str, float]]:
    """Extract score information from ProteinMPNN FASTA header.

    Args:
        header: FASTA header line (without '>')

    Returns:
        Dictionary with extracted scores, or None if no scores found

    Example:
        >>> header = "T=0.1, sample=1, score=1.2345, global_score=1.1234"
        >>> extract_scores_from_fasta_header(header)
        {'score': 1.2345, 'global_score': 1.1234, 'temperature': 0.1, 'sample': 1}
    """
    scores = {}

    # Extract various numeric values from header
    patterns = {
        'score': r'score=([0-9.-]+)',
        'global_score': r'global_score=([0-9.-]+)',
        'temperature': r'T=([0-9.-]+)',
        'sample': r'sample=([0-9]+)',
        'seq_recovery': r'seq_recovery=([0-9.-]+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, header)
        if match:
            try:
                value = float(match.group(1))
                scores[key] = value
            except ValueError:
                continue

    return scores if scores else None


def create_output_directories(base_output_dir: Union[str, Path]) -> Dict[str, Path]:
    """Create standard output directory structure.

    Args:
        base_output_dir: Base output directory

    Returns:
        Dictionary with paths to created directories

    Example:
        >>> dirs = create_output_directories("results/uc_001")
        >>> print(dirs['seqs'])  # Path to sequences directory
    """
    base_path = Path(base_output_dir)
    directories = {
        'base': base_path,
        'seqs': base_path / 'seqs',
        'scores': base_path / 'scores',
        'logs': base_path / 'logs'
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories