"""MCP Server for ProteinMPNN

Provides both synchronous and asynchronous (submit) APIs for all tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import os

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("ProteinMPNN")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def simple_design(
    input_file: str,
    chains: Optional[str] = None,
    num_sequences: int = 3,
    temperature: float = 0.1,
    output_dir: Optional[str] = None
) -> dict:
    """
    Generate protein sequences for a given PDB structure using ProteinMPNN.

    Fast operation that completes in ~10 seconds. Use this for single structure design.

    Args:
        input_file: Path to input PDB file
        chains: Space-separated chain IDs to design (e.g., "A B")
        num_sequences: Number of sequences to generate (default: 3)
        temperature: Sampling temperature (default: 0.1)
        output_dir: Optional directory to save output files

    Returns:
        Dictionary with generated sequences and metadata
    """
    try:
        from simple_design import run_simple_design

        result = run_simple_design(
            input_file=input_file,
            chains=chains,
            num_sequences=num_sequences,
            temperature=temperature,
            output_dir=output_dir
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"simple_design failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def sequence_scoring(
    input_file: str,
    fasta_sequences: Optional[str] = None,
    save_probs: bool = False,
    output_dir: Optional[str] = None
) -> dict:
    """
    Score protein sequences using ProteinMPNN likelihood calculation.

    Fast operation that completes in ~8 seconds. Use this for sequence evaluation.

    Args:
        input_file: Path to input PDB structure file
        fasta_sequences: Custom sequences to score (format: SEQ1/SEQ2)
        save_probs: Save per-residue probabilities
        output_dir: Optional directory to save output files

    Returns:
        Dictionary with sequence scores and analysis
    """
    try:
        from sequence_scoring import run_sequence_scoring

        result = run_sequence_scoring(
            input_file=input_file,
            fasta_sequences=fasta_sequences,
            save_probs=save_probs,
            output_dir=output_dir
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"sequence_scoring failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def constrained_design(
    input_file: str,
    chains_to_design: Optional[str] = None,
    fixed_positions: Optional[str] = None,
    num_sequences: int = 3,
    output_dir: Optional[str] = None
) -> dict:
    """
    Design proteins with constrained/fixed positions using ProteinMPNN.

    Fast operation for constrained design. Use this for position-specific constraints.

    Args:
        input_file: Path to input PDB file
        chains_to_design: Chains to design (e.g., "A B")
        fixed_positions: Fixed positions per chain (format: '1 2 3, 10 11 12')
        num_sequences: Number of sequences to generate
        output_dir: Optional directory to save output files

    Returns:
        Dictionary with constrained sequences and metadata
    """
    try:
        from constrained_design import run_constrained_design

        result = run_constrained_design(
            input_file=input_file,
            chains_to_design=chains_to_design,
            fixed_positions=fixed_positions,
            num_sequences=num_sequences,
            output_dir=output_dir
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"constrained_design failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def ca_only_design(
    input_file: str,
    chains: Optional[str] = None,
    model: str = "v_48_020",
    num_sequences: int = 3,
    output_dir: Optional[str] = None
) -> dict:
    """
    Design sequences using only carbon alpha atoms (backbone traces).

    Fast operation for CA-only design. Use this for backbone-only structures.

    Args:
        input_file: Path to input PDB file (full-atom or CA-only)
        chains: Chains to design using CA coordinates
        model: CA model version (v_48_002, v_48_010, v_48_020)
        num_sequences: Number of sequences to generate
        output_dir: Optional directory to save output files

    Returns:
        Dictionary with CA-designed sequences and metadata
    """
    try:
        from ca_only_design import run_ca_only_design

        result = run_ca_only_design(
            input_file=input_file,
            chains=chains,
            model=model,
            num_sequences=num_sequences,
            output_dir=output_dir
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"ca_only_design failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_batch_design(
    input_dir: str,
    file_pattern: str = "*.pdb",
    chains: Optional[str] = None,
    num_sequences: int = 2,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch processing for multiple PDB files.

    This operation may take >10 minutes for large batches. Returns a job_id for tracking.

    Args:
        input_dir: Directory containing PDB files to process
        file_pattern: Pattern to match PDB files (default: "*.pdb")
        chains: Chains to design (empty = auto-detect)
        num_sequences: Number of sequences per structure
        output_dir: Directory for outputs
        job_name: Optional name for tracking

    Returns:
        Dictionary with job_id. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results
        - get_job_log(job_id) to see logs
    """
    script_path = str(SCRIPTS_DIR / "batch_design.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input_dir": input_dir,
            "file_pattern": file_pattern,
            "chains": chains,
            "num_sequences": num_sequences,
            "output": output_dir
        },
        job_name=job_name or f"batch_design_{Path(input_dir).name}"
    )

@mcp.tool()
def submit_large_design(
    input_file: str,
    chains: Optional[str] = None,
    num_sequences: int = 50,
    temperature: float = 0.1,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit large-scale sequence design for background processing.

    Use this for generating many sequences (>10) which may take longer.

    Args:
        input_file: Path to input PDB file
        chains: Space-separated chain IDs to design
        num_sequences: Number of sequences to generate (large number)
        temperature: Sampling temperature
        output_dir: Directory for outputs
        job_name: Optional name for tracking

    Returns:
        Dictionary with job_id for tracking the design job
    """
    script_path = str(SCRIPTS_DIR / "simple_design.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "chains": chains,
            "num_sequences": num_sequences,
            "temperature": temperature,
            "output": output_dir
        },
        job_name=job_name or f"large_design_{num_sequences}_seqs"
    )

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_pdb_structure(input_file: str) -> dict:
    """
    Validate a PDB structure for ProteinMPNN compatibility.

    Quick validation of PDB file format and chains.

    Args:
        input_file: Path to PDB file to validate

    Returns:
        Dictionary with validation results and structure info
    """
    try:
        from lib.utils import check_file_exists, validate_chains
        from lib.io import load_fasta

        # Check file exists
        if not check_file_exists(input_file):
            return {"status": "error", "error": "PDB file not found"}

        # Basic PDB parsing to get chain info
        chains = []
        with open(input_file) as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain_id = line[21:22].strip()
                    if chain_id and chain_id not in chains:
                        chains.append(chain_id)

        return {
            "status": "success",
            "file_path": str(Path(input_file).resolve()),
            "file_size": os.path.getsize(input_file),
            "chains_found": chains,
            "num_chains": len(chains),
            "valid": len(chains) > 0
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def list_example_structures() -> dict:
    """
    List available example PDB structures for testing.

    Returns paths to example structures included with ProteinMPNN.

    Returns:
        Dictionary with example structure paths and descriptions
    """
    try:
        examples_dir = MCP_ROOT / "examples" / "data" / "inputs"
        structures = []

        for pdb_dir in examples_dir.rglob("pdbs"):
            if pdb_dir.is_dir():
                for pdb_file in pdb_dir.glob("*.pdb"):
                    structures.append({
                        "name": pdb_file.name,
                        "path": str(pdb_file.resolve()),
                        "category": pdb_dir.parent.name,
                        "size": os.path.getsize(pdb_file)
                    })

        return {
            "status": "success",
            "examples_dir": str(examples_dir),
            "structures": structures,
            "total_count": len(structures)
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()