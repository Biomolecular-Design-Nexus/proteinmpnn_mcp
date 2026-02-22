# ProteinMPNN MCP

**Protein sequence design using ProteinMPNN via Docker**

An MCP (Model Context Protocol) server for protein design with 7 core tools:
- Design protein sequences from PDB structures
- Score protein sequences for likelihood
- Constrained design with fixed/redesigned positions
- CA-only design using backbone coordinates
- Submit large-scale batch design jobs with async job tracking
- Monitor and retrieve design results
- List available example structures

## Quick Start with Docker

### Approach 1: Pull Pre-built Image from GitHub

The fastest way to get started. A pre-built Docker image is automatically published to GitHub Container Registry on every release.

```bash
# Pull the latest image
docker pull ghcr.io/macromnex/proteinmpnn_mcp:latest

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add proteinmpnn -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` ghcr.io/macromnex/proteinmpnn_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support (`nvidia-docker` or Docker with NVIDIA runtime)
- Claude Code installed

That's it! The ProteinMPNN MCP server is now available in Claude Code.

---

### Approach 2: Build Docker Image Locally

Build the image yourself and install it into Claude Code. Useful for customization or offline environments.

```bash
# Clone the repository
git clone https://github.com/MacromNex/proteinmpnn_mcp.git
cd proteinmpnn_mcp

# Build the Docker image
docker build -t proteinmpnn_mcp:latest .

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add proteinmpnn -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` proteinmpnn_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support
- Claude Code installed
- Git (to clone the repository)

**About the Docker Flags:**
- `-i` — Interactive mode for Claude Code
- `--rm` — Automatically remove container after exit
- `` --user `id -u`:`id -g` `` — Runs the container as your current user, so output files are owned by you (not root)
- `--gpus all` — Grants access to all available GPUs
- `--ipc=host` — Uses host IPC namespace for PyTorch shared memory
- `-v` — Mounts your project directory so the container can access your data

---

## Verify Installation

After adding the MCP server, you can verify it's working:

```bash
# List registered MCP servers
claude mcp list

# You should see 'proteinmpnn' in the output
```

In Claude Code, you can now use all 7 ProteinMPNN tools:
- `simple_design`
- `sequence_scoring`
- `constrained_design`
- `ca_only_design`
- `submit_batch_design`
- `get_job_status`
- `get_job_result`

---

## Next Steps

- **Detailed documentation**: See [detail.md](detail.md) for comprehensive guides on:
  - Available MCP tools and parameters
  - Local Python environment setup (alternative to Docker)
  - Example workflows and use cases
  - Configuration file options
  - Troubleshooting

---

## Usage Examples

Once registered, you can use the ProteinMPNN tools directly in Claude Code. Here are some common workflows:

### Example 1: Basic Protein Sequence Design

```
I have a protein structure at /path/to/3HTN.pdb with chains A and B. Can you use simple_design to generate 5 diverse sequences with temperature 0.1 and save results to /path/to/results/?
```

### Example 2: Constrained Design with Fixed Positions

```
I want to redesign /path/to/protein.pdb while keeping key residues fixed at positions 1, 2, 3 in chain A and positions 10, 11 in chain B. Can you use constrained_design and generate 3 sequences?
```

### Example 3: Batch Design for Multiple Structures

```
I have many PDB files in /path/to/structures/ directory. Can you submit a batch design job using submit_batch_design for all .pdb files, generate 5 sequences each, and save results to /path/to/results/? Monitor the job until it finishes.
```

---

## Troubleshooting

**Docker not found?**
```bash
docker --version  # Install Docker if missing
```

**GPU not accessible?**
- Ensure NVIDIA Docker runtime is installed
- Check with: `docker run --gpus all ubuntu nvidia-smi`

**Claude Code not found?**
```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

---

## License

MIT — Based on [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) by Dauparas et al.
