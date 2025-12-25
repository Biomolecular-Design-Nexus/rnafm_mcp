# RNA-FM Examples

This directory contains standalone Python scripts demonstrating the key use cases of RNA-FM foundation models. All scripts are designed to work with the legacy environment (`./env_py38`).

## Quick Start

```bash
# Activate the RNA-FM environment
mamba activate ./env_py38

# Run working example (mock version due to model download issues)
python use_case_1_rna_embeddings_mock.py --input data/example.fasta --output ../results/embeddings
```

## ⚠️ Important Notice

**Model Download Issue**: The RNA-FM pretrained model (~1.1GB) download currently fails due to network/server issues. This affects all use cases. A working mock version is available for embedding extraction that demonstrates the workflow without requiring the model download.

## Use Case Scripts

### 1. RNA Sequence Embeddings (`use_case_1_rna_embeddings.py`)
Extract 640-dimensional embeddings from RNA sequences using RNA-FM or mRNA-FM.

```bash
python use_case_1_rna_embeddings.py --input data/example.fasta --output ../results/embeddings
python use_case_1_rna_embeddings.py --input data/example.fasta --model mrna-fm --output ../results/mrna_embeddings
```

### 2. Secondary Structure Prediction (`use_case_2_secondary_structure.py`)
Predict RNA secondary structure with base-pair probability matrices.

```bash
python use_case_2_secondary_structure.py --input data/example.fasta --output ../results/structures
python use_case_2_secondary_structure.py --input data/RF00001.fasta --threshold 0.6 --output ../results/5s_structures
```

### 3. RNA Family Classification (`use_case_3_rna_classification.py`)
Cluster and classify RNA sequences by functional families.

```bash
python use_case_3_rna_classification.py --input data/RF00001.fasta --output ../results/classification
python use_case_3_rna_classification.py --input data/format_rnacentral_active.100.sample-Max50.fasta --clusters 10 --output ../results/large_classification
```

### 4. mRNA Translation Analysis (`use_case_4_mrna_expression.py`)
Analyze mRNA coding sequences for translation efficiency features.

```bash
python use_case_4_mrna_expression.py --input data/example.fasta --output ../results/mrna_analysis
```

### 5. UTR Function Prediction (`use_case_5_utr_function.py`)
Predict regulatory elements in 5' and 3' untranslated regions.

```bash
python use_case_5_utr_function.py --input data/example.fasta --type 5UTR --output ../results/5utr_analysis
python use_case_5_utr_function.py --input data/example.fasta --type 3UTR --output ../results/3utr_analysis
```

## Demo Data

### Basic Test Data
- `data/example.fasta` - Small set of RNA sequences for quick testing

### RNA Family Data
- `data/RF00001.fasta` - 5S ribosomal RNA sequences
- `data/RF00005.fasta` - tRNA sequences
- `data/RF00010.fasta` - RNase P RNA sequences

### Large Dataset
- `data/format_rnacentral_active.100.sample-Max50.fasta` - 100 diverse RNA sequences for clustering

### Configuration Files
- `data/extract_embedding.yml` - RNA-FM embedding extraction config
- `data/ss_prediction.yml` - Secondary structure prediction config

## Common Workflows

### Basic RNA Analysis Pipeline
```bash
# 1. Extract embeddings
python use_case_1_rna_embeddings.py --input data/RF00001.fasta --output ../results/5s_embeddings

# 2. Predict secondary structure
python use_case_2_secondary_structure.py --input data/RF00001.fasta --output ../results/5s_structures

# 3. Classify RNA families
python use_case_3_rna_classification.py --input data/RF00001.fasta --output ../results/5s_classification
```

### mRNA-specific Analysis
```bash
# For mRNA sequences (length must be divisible by 3)
python use_case_1_rna_embeddings.py --input data/mrna_sequences.fasta --model mrna-fm --output ../results/mrna_embeddings
python use_case_4_mrna_expression.py --input data/mrna_sequences.fasta --output ../results/mrna_features
```

### UTR Regulatory Analysis
```bash
# Analyze 5' UTRs
python use_case_5_utr_function.py --input data/5utr_sequences.fasta --type 5UTR --output ../results/5utr_regulatory

# Analyze 3' UTRs
python use_case_5_utr_function.py --input data/3utr_sequences.fasta --type 3UTR --output ../results/3utr_regulatory
```

## Output Formats

### Embeddings
- `.npy` files: NumPy arrays with sequence embeddings
- `.npz` files: Compressed multiple arrays
- `.txt` files: Human-readable summaries

### Secondary Structure
- `.npy` files: Probability matrices and contact maps
- `.txt` files: Dot-bracket notation
- `.ct` files: Connectivity table format

### Classification
- `.txt` files: Cluster assignments and summaries
- `.csv` files: Feature matrices for machine learning
- `.png` files: Visualization plots (if matplotlib available)

### Analysis Reports
- `.txt` files: Detailed analysis reports
- `.csv` files: Machine-readable feature tables

## Verified Working Examples

The following examples have been tested and verified to work:

### Mock RNA Embedding Extraction
```bash
# Activate environment
mamba activate ./env_py38

# Run mock embedding extraction (works without model download)
python use_case_1_rna_embeddings_mock.py \
    --input data/example.fasta \
    --output ../results/embeddings_mock

# Expected output:
# - ../results/embeddings_mock/*.npy (embeddings for each sequence)
# - ../results/embeddings_mock/embeddings_summary.txt (processing summary)
```

**Output**: Successfully processes 3 sequences from example.fasta:
- `3ktw_C`: 96 nucleotides → 96×640 embeddings
- `2der_D`: 72 nucleotides → 72×640 embeddings
- `1p6v_B`: 45 nucleotides → 45×640 embeddings

### Available Test Data
- `data/example.fasta` - 3 RNA sequences (45-96 nucleotides) ✅ Verified
- `data/RF00001.fasta` - 5S ribosomal RNA family ✅ Available
- `data/RF00005.fasta` - tRNA family ✅ Available
- `data/RF00010.fasta` - RNase P RNA ✅ Available
- `data/format_rnacentral_active.100.sample-Max50.fasta` - 100 sequences ✅ Available

## Troubleshooting

### Issue: Model Download Fails
**Symptoms**:
```
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
```

**Solutions**:
1. **Use Mock Version**: Use `use_case_1_rna_embeddings_mock.py` for testing workflows
2. **Alternative Download**: Try downloading the model manually from a different network
3. **Contact Authors**: RNA-FM model server may have temporary issues

### Issue: Import Errors
**Symptoms**:
```
ImportError: No module named 'fm'
```

**Solution**: Ensure correct environment and paths:
```bash
# Check environment
mamba activate ./env_py38
python -c "import torch; print('PyTorch:', torch.__version__)"

# Verify path fix was applied
grep "repo/RNA-FM'" examples/use_case_*.py
```

### Issue: Wrong Environment
**Symptoms**: Missing packages or Python version errors

**Solution**: Use the legacy Python 3.8 environment:
```bash
mamba activate ./env_py38  # NOT ./env
python --version  # Should show Python 3.8.11
```

## Tips

1. **Memory Management**: For large datasets, consider processing in smaller batches
2. **GPU Usage**: Set `CUDA_VISIBLE_DEVICES=""` to force CPU-only mode if needed
3. **Sequence Requirements**:
   - mRNA-FM requires sequences divisible by 3
   - Minimum sequence length ~20 nucleotides recommended
4. **Output Organization**: Use descriptive output directory names to organize results
5. **Mock vs Real**: Mock versions demonstrate workflow; real versions require model download