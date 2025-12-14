"""
Shared utilities for ProteinMPNN MCP scripts.

This module contains common functions extracted and simplified from the ProteinMPNN repository
to minimize dependencies and create self-contained MCP tools.
"""

__version__ = "1.0.0"
__description__ = "ProteinMPNN MCP Scripts Shared Library"

from .io import load_json, save_json, load_fasta, save_fasta
from .utils import validate_chains, format_command, check_file_exists

__all__ = [
    "load_json", "save_json", "load_fasta", "save_fasta",
    "validate_chains", "format_command", "check_file_exists"
]