# RNA-FM MCP

> RNA sequence analysis tools powered by RNA Foundation Models, accessible via Model Context Protocol (MCP)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

This MCP server provides access to RNA Foundation Model (RNA-FM) capabilities for analyzing RNA sequences. It offers both synchronous tools for quick analyses and asynchronous (submit) APIs for long-running tasks, with comprehensive fallback modes for testing and development.

### Features
- **RNA Sequence Embeddings**: Extract 640-dimensional contextual representations using RNA-FM
- **Secondary Structure Prediction**: Predict base-pairing patterns with probability matrices
- **RNA Classification**: Functional clustering and family classification using embeddings + K-means
- **Job Management**: Submit long-running tasks with progress tracking and log monitoring
- **Batch Processing**: Process multiple files simultaneously with comprehensive analysis
- **Mock Mode**: Fast fallback generation for testing without GPU requirements

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Main MCP environment (Python 3.10)
├── env_py38/              # Legacy environment for RNA-FM (Python 3.8)
├── src/
│   └── server.py           # MCP server with 11 tools
├── scripts/
│   ├── rna_embeddings.py      # Extract RNA sequence embeddings
│   ├── secondary_structure.py # Predict RNA secondary structure
│   ├── rna_classification.py  # Classify and cluster RNA sequences
│   └── lib/                   # Shared utilities
├── examples/
│   └── data/               # Demo FASTA files and configs
├── configs/                # Configuration templates
│   ├── rna_embeddings_config.json
│   ├── secondary_structure_config.json
│   └── rna_classification_config.json
└── repo/                   # Original RNA-FM repository
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+ for MCP server
- Python 3.8.11 for RNA-FM models (optional, has fallbacks)

### Create Environment
Please strictly follow the dual environment setup procedure from `reports/step3_environment.md`. An example workflow is shown below.

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnafm_mcp

# Create main MCP environment (Python 3.10 for MCP server)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate main environment
mamba activate ./env
# or: conda activate ./env

# Install MCP dependencies
pip install fastmcp loguru pandas numpy tqdm

# Optional: Create legacy environment for real RNA-FM models (Python 3.8)
mamba create -p ./env_py38 python=3.8.11 pytorch=1.9.0 cudatoolkit=11.1.1 -c pytorch -c conda-forge -y
mamba activate ./env_py38
pip install biopython scikit-learn matplotlib
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/rna_embeddings.py` | Extract 640-dimensional embeddings from RNA sequences | See below |
| `scripts/secondary_structure.py` | Predict base-pairing patterns and secondary structures | See below |
| `scripts/rna_classification.py` | Classify and cluster RNA sequences by functional families | See below |

### Script Examples

#### RNA Embeddings Extraction

```bash
# Activate environment (main env is sufficient for mock mode)
mamba activate ./env

# Run with mock embeddings (fast, no GPU required)
python scripts/rna_embeddings.py \
  --input examples/data/example.fasta \
  --output results/embeddings \
  --use-mock

# Run with real RNA-FM model (requires env_py38 + GPU)
mamba activate ./env_py38
python scripts/rna_embeddings.py \
  --input examples/data/example.fasta \
  --output results/embeddings \
  --config configs/rna_embeddings_config.json
```

**Parameters:**
- `--input, -i`: Path to FASTA file with RNA sequences (required)
- `--output, -o`: Output directory to save embeddings (default: results/)
- `--use-mock`: Use deterministic mock embeddings instead of RNA-FM model
- `--config, -c`: Configuration file path (optional)

#### Secondary Structure Prediction

```bash
# Run with mock predictions (fast)
python scripts/secondary_structure.py \
  --input examples/data/example.fasta \
  --output results/structures \
  --use-mock \
  --threshold 0.5

# Run with real RNA-FM ResNet model
mamba activate ./env_py38
python scripts/secondary_structure.py \
  --input examples/data/RF00005.fasta \
  --output results/structures \
  --threshold 0.6 \
  --formats npy txt ct
```

**Parameters:**
- `--input, -i`: Path to FASTA file with RNA sequences (required)
- `--output, -o`: Output directory (default: results/)
- `--threshold, -t`: Base-pair probability threshold (default: 0.5)
- `--use-mock`: Use mock structure predictions
- `--formats`: Output formats: npy, txt, ct (default: all)

#### RNA Classification and Clustering

```bash
# Classify tRNA sequences into functional families
python scripts/rna_classification.py \
  --input examples/data/RF00005.fasta \
  --output results/classification \
  --clusters 3 \
  --use-mock \
  --pca-components 50
```

**Parameters:**
- `--input, -i`: Path to FASTA file with RNA sequences (required)
- `--output, -o`: Output directory (default: results/)
- `--clusters, -c`: Number of clusters for K-means (default: 3)
- `--use-mock`: Use mock embeddings for classification
- `--pca-components`: PCA dimensions for visualization (default: 50)

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
mamba activate ./env
fastmcp install src/server.py --name RNA-FM
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
mamba activate ./env
claude mcp add RNA-FM -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "RNA-FM": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnafm_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnafm_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What RNA analysis tools are available from RNA-FM?
```

#### Basic RNA Embeddings
```
Extract RNA embeddings from @examples/data/example.fasta using mock mode
```

#### Secondary Structure with Configuration
```
Predict secondary structure for @examples/data/RF00001.fasta with threshold 0.6 using @configs/secondary_structure_config.json
```

#### RNA Classification
```
Classify RNA sequences in @examples/data/RF00005.fasta into 3 functional clusters using mock embeddings
```

#### Long-Running Tasks (Submit API)
```
Submit RNA classification for @examples/data/format_rnacentral_active.100.sample-Max50.fasta with 5 clusters
Then check the job status
```

#### Batch Processing
```
Process these files in batch with all analysis types:
- @examples/data/RF00001.fasta
- @examples/data/RF00005.fasta
- @examples/data/RF00010.fasta
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/example.fasta` | Basic RNA sequences for testing |
| `@examples/data/RF00005.fasta` | tRNA family sequences |
| `@configs/rna_embeddings_config.json` | Embeddings configuration |
| `@results/` | Output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "RNA-FM": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnafm_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnafm_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (similar to Claude Code)
> What RNA tools are available?
> Extract embeddings from examples/data/example.fasta
> Classify RNA sequences in examples/data/RF00005.fasta
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `extract_rna_embeddings` | Extract 640-dim embeddings using RNA-FM | `input_file`, `output_file`, `use_mock`, `config_file` |
| `predict_secondary_structure` | Predict RNA base-pairing patterns | `input_file`, `output_file`, `threshold`, `use_mock`, `formats` |
| `classify_rna_sequences` | Cluster RNA sequences by functional families | `input_file`, `output_file`, `num_clusters`, `use_mock`, `pca_components` |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_rna_embeddings` | Large-scale embeddings extraction | `input_file`, `output_dir`, `use_mock`, `job_name` |
| `submit_secondary_structure` | Large-scale structure prediction | `input_file`, `output_dir`, `threshold`, `use_mock`, `job_name` |
| `submit_rna_classification` | Large-scale RNA classification | `input_file`, `output_dir`, `num_clusters`, `use_mock`, `job_name` |
| `submit_batch_rna_analysis` | Process multiple files with all analyses | `input_files`, `analysis_type`, `output_dir`, `use_mock`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and current status |
| `get_job_result` | Get results when job completed |
| `get_job_log` | View execution logs with tail option |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs with status filter |

---

## Examples

### Example 1: RNA Sequence Embeddings

**Goal:** Extract contextual embeddings from RNA sequences for downstream analysis

**Using Script:**
```bash
mamba activate ./env
python scripts/rna_embeddings.py \
  --input examples/data/example.fasta \
  --output results/example1/ \
  --use-mock
```

**Using MCP (in Claude Code):**
```
Extract RNA embeddings from @examples/data/example.fasta using mock mode and save to results/example1/
```

**Expected Output:**
- `3ktw_C_embeddings.npy`: 640-dim embeddings (96 x 640)
- `2der_D_embeddings.npy`: 640-dim embeddings (72 x 640)
- `1p6v_B_embeddings.npy`: 640-dim embeddings (45 x 640)
- `embeddings_summary.txt`: Processing statistics

### Example 2: Secondary Structure Prediction

**Goal:** Predict RNA base-pairing patterns and secondary structures

**Using Script:**
```bash
python scripts/secondary_structure.py \
  --input examples/data/RF00001.fasta \
  --output results/example2/ \
  --use-mock \
  --threshold 0.5
```

**Using MCP (in Claude Code):**
```
Predict secondary structure for @examples/data/RF00001.fasta with threshold 0.5 using mock mode
```

**Expected Output:**
- `*_probability_matrix.npy`: Base-pair probability matrices
- `*_structure.txt`: Human-readable structures with base pairs
- `*_structure.ct`: Connect Table format files
- `structure_summary.txt`: Processing summary

### Example 3: RNA Functional Classification

**Goal:** Classify RNA sequences into functional families

**Using Script:**
```bash
python scripts/rna_classification.py \
  --input examples/data/RF00005.fasta \
  --output results/example3/ \
  --clusters 3 \
  --use-mock
```

**Using MCP (in Claude Code):**
```
Classify tRNA sequences in @examples/data/RF00005.fasta into 3 functional clusters using mock embeddings
```

**Expected Output:**
- `cluster_assignments.json`: Sequence-to-cluster mappings
- `embeddings.npy`: Full embedding matrix (954 x 640)
- `embeddings_pca.npy`: PCA-reduced embeddings (954 x 50)
- `cluster_analysis.json`: Cluster statistics and silhouette scores
- `classification_summary.txt`: Processing summary

### Example 4: Batch Processing

**Goal:** Process multiple RNA files with comprehensive analysis

**Using Script:**
```bash
for f in examples/data/RF*.fasta; do
  python scripts/rna_embeddings.py --input "$f" --output results/batch/ --use-mock
  python scripts/secondary_structure.py --input "$f" --output results/batch/ --use-mock
  python scripts/rna_classification.py --input "$f" --output results/batch/ --use-mock
done
```

**Using MCP (in Claude Code):**
```
Submit batch RNA analysis for all files:
- @examples/data/RF00001.fasta
- @examples/data/RF00005.fasta
- @examples/data/RF00010.fasta

Process with all analysis types using mock mode
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Sequences | Use With |
|------|-------------|-----------|----------|
| `example.fasta` | Basic RNA sequences for testing | 3 | All tools |
| `RF00001.fasta` | 5S ribosomal RNA family | Many | Structure prediction, classification |
| `RF00005.fasta` | tRNA family sequences | 954 | Classification, embeddings |
| `RF00010.fasta` | RNase P RNA sequences | Many | Structure prediction |
| `format_rnacentral_active.100.sample-Max50.fasta` | Large RNA dataset | 100 | Batch processing |
| `extract_embedding.yml` | RNA-FM embedding config | - | Configuration reference |
| `ss_prediction.yml` | Structure prediction config | - | Configuration reference |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Key Parameters |
|--------|-------------|----------------|
| `rna_embeddings_config.json` | Embeddings extraction settings | `embedding_dim: 640`, `device: cuda`, `batch_size: 8` |
| `secondary_structure_config.json` | Structure prediction settings | `threshold: 0.5`, `formats: [npy, txt, ct]` |
| `rna_classification_config.json` | Classification settings | `num_clusters: 3`, `pca_components: 50` |

### Config Example

```json
{
  "model": {
    "type": "rna-fm",
    "embedding_dim": 640,
    "device": "cuda"
  },
  "processing": {
    "batch_size": 8,
    "use_mock": false
  },
  "mock_generation": {
    "nucleotide_bias": {"A": 0.1, "U": 0.2, "G": 0.3, "C": 0.4},
    "positional_encoding_strength": 0.5
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate main environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install fastmcp loguru pandas numpy tqdm
```

**Problem:** Import errors in scripts
```bash
# Verify environment activation
which python
python -c "import numpy, pathlib, json; print('Core imports OK')"
```

**Problem:** Legacy environment for real models
```bash
# Create legacy environment for RNA-FM
mamba create -p ./env_py38 python=3.8.11 pytorch=1.9.0 cudatoolkit=11.1.1 -c pytorch -y
mamba activate ./env_py38
pip install biopython scikit-learn
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove RNA-FM
mamba activate ./env
claude mcp add RNA-FM -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
mamba activate ./env
python src/server.py --test
```

**Problem:** Path issues in server
```bash
# Verify paths exist
ls -la src/server.py
ls -la scripts/
ls -la examples/data/
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/
```

**Problem:** Job failed with error
```
Use get_job_log with job_id "<job_id>" and tail 100 to see detailed error logs
```

**Problem:** Out of memory errors
```bash
# Use smaller batch sizes or mock mode
python scripts/rna_embeddings.py --input file.fasta --use-mock
```

### Script Issues

**Problem:** FASTA parsing errors
```bash
# Validate FASTA format
head -10 examples/data/example.fasta
# Check for proper headers (>) and valid RNA nucleotides (A, U, G, C)
```

**Problem:** Mock mode not working
```bash
# Verify NumPy installation
python -c "import numpy; print(numpy.__version__)"
```

**Problem:** Real model loading fails
```bash
# Fall back to mock mode
python scripts/rna_embeddings.py --input file.fasta --use-mock
# Or check if repo/RNA-FM/ exists and is properly configured
```

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test individual scripts
python scripts/rna_embeddings.py --input examples/data/example.fasta --use-mock --output /tmp/test1
python scripts/secondary_structure.py --input examples/data/example.fasta --use-mock --output /tmp/test2
python scripts/rna_classification.py --input examples/data/RF00005.fasta --use-mock --output /tmp/test3
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
mamba activate ./env
fastmcp dev src/server.py

# Or run directly
python src/server.py
```

### Performance Benchmarks

**Mock Mode Performance** (on example.fasta, 3 sequences):
- `rna_embeddings.py`: ~0.8s, outputs 4 files (1.09MB)
- `secondary_structure.py`: ~0.5s, outputs 7 files (160KB)
- `rna_classification.py` on RF00005.fasta (954 sequences): ~2.1s, outputs 5 files (21.2MB)

---

## License

This MCP server is based on the [RNA-FM](https://github.com/ml4bio/RNA-FM) foundation model repository and provides an MCP interface for convenient access to RNA analysis capabilities.

## Credits

- **Original RNA-FM Model**: [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM)
- **MCP Framework**: [Anthropic Model Context Protocol](https://github.com/anthropics/mcp)
- **FastMCP**: [FastMCP framework](https://github.com/jlowin/fastmcp)