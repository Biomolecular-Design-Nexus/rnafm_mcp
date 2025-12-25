# Step 3: Use Cases Report

## Scan Information
- **Scan Date**: 2025-12-24
- **Filter Applied**: RNA sequence embedding, ncRNA function prediction, RNA secondary structure prediction, mRNA translation efficiency prediction
- **Python Version**: 3.8.11
- **Environment Strategy**: dual

## Use Cases

### UC-001: RNA Sequence Embedding Extraction
- **Description**: Extract deep learning embeddings from RNA sequences using RNA-FM foundation model
- **Script Path**: `examples/use_case_1_rna_embeddings.py`
- **Complexity**: medium
- **Priority**: high
- **Environment**: `./env_py38`
- **Source**: `repo/RNA-FM/redevelop/pretrained/extract_embedding.yml`, README.md embedding section

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_fasta | file | Input FASTA file with RNA sequences | --input, -i |
| model_type | choice | Model variant (rna-fm, mrna-fm) | --model |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| embeddings | npy files | 640-dimensional embeddings per sequence |
| summary | txt file | Processing summary and statistics |

**Example Usage:**
```bash
mamba activate ./env_py38
python examples/use_case_1_rna_embeddings.py --input examples/data/example.fasta --output results/embeddings
```

**Example Data**: `examples/data/example.fasta`, `examples/data/RF00001.fasta`

---

### UC-002: RNA Secondary Structure Prediction
- **Description**: Predict RNA secondary structure (base-pairing patterns) using RNA-FM + ResNet predictor
- **Script Path**: `examples/use_case_2_secondary_structure.py`
- **Complexity**: medium
- **Priority**: high
- **Environment**: `./env_py38`
- **Source**: `repo/RNA-FM/redevelop/pretrained/ss_prediction.yml`, `tutorials/secondary-structure-prediction/Secondary-Structure-Prediction.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_fasta | file | Input FASTA file with RNA sequences | --input, -i |
| threshold | float | Base-pair probability threshold | --threshold, -t |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| prob_matrix | npy files | Base-pair probability matrices |
| contact_map | npy files | Binary contact maps |
| structure | txt files | Dot-bracket notation structures |
| ct_format | ct files | Connectivity table format |

**Example Usage:**
```bash
mamba activate ./env_py38
python examples/use_case_2_secondary_structure.py --input examples/data/example.fasta --output results/structures --threshold 0.5
```

**Example Data**: `examples/data/example.fasta`

---

### UC-003: RNA Family Clustering and Classification
- **Description**: Perform RNA family clustering and classification using RNA-FM embeddings for functional analysis
- **Script Path**: `examples/use_case_3_rna_classification.py`
- **Complexity**: complex
- **Priority**: high
- **Environment**: `./env_py38`
- **Source**: `tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_fasta | file | Input FASTA file with RNA sequences | --input, -i |
| n_clusters | int | Number of clusters for K-means | --clusters, -c |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| embeddings | npy file | RNA sequence embeddings matrix |
| cluster_assignments | txt file | Cluster assignments per sequence |
| cluster_summary | txt file | Summary statistics per cluster |
| visualization | png file | PCA/t-SNE clustering plots |

**Example Usage:**
```bash
mamba activate ./env_py38
python examples/use_case_3_rna_classification.py --input examples/data/format_rnacentral_active.100.sample-Max50.fasta --output results/classification --clusters 5
```

**Example Data**: `examples/data/format_rnacentral_active.100.sample-Max50.fasta`, `examples/data/RF00001.fasta`

---

### UC-004: mRNA Translation Efficiency Prediction
- **Description**: Analyze mRNA sequences for translation efficiency using mRNA-FM with codon-aware features
- **Script Path**: `examples/use_case_4_mrna_expression.py`
- **Complexity**: complex
- **Priority**: medium
- **Environment**: `./env_py38`
- **Source**: `tutorials/mRNA_expression/mrnafm-tutorial-code.ipynb`, README.md mRNA-FM section

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_fasta | file | Input FASTA file with mRNA CDS | --input, -i |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| embeddings | npz file | mRNA-specific codon-level embeddings |
| analysis | txt file | Translation efficiency analysis |
| codon_usage | txt file | Codon usage statistics |
| features | csv file | Machine-readable feature matrix |

**Example Usage:**
```bash
mamba activate ./env_py38
python examples/use_case_4_mrna_expression.py --input examples/data/example.fasta --output results/mrna_analysis
```

**Example Data**: `examples/data/example.fasta` (note: requires sequences divisible by 3)

---

### UC-005: UTR Function Prediction
- **Description**: Predict functional properties of untranslated regions (5' and 3' UTRs) using RNA-FM
- **Script Path**: `examples/use_case_5_utr_function.py`
- **Complexity**: complex
- **Priority**: medium
- **Environment**: `./env_py38`
- **Source**: `tutorials/utr-function-prediction/UTR-Function-Prediction.ipynb`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input_fasta | file | Input FASTA file with UTR sequences | --input, -i |
| region_type | choice | UTR type (5UTR, 3UTR) | --type, -t |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| embeddings | npz file | UTR-specific embeddings |
| analysis | txt file | Detailed UTR functional analysis |
| features | csv file | UTR feature matrix |
| regulatory_elements | txt file | Predicted regulatory elements |

**Example Usage:**
```bash
mamba activate ./env_py38
python examples/use_case_5_utr_function.py --input examples/data/example.fasta --output results/utr_analysis --type 5UTR
```

**Example Data**: `examples/data/example.fasta`

---

## Summary

| Metric | Count |
|--------|-------|
| Total Found | 5 |
| Scripts Created | 5 |
| High Priority | 3 |
| Medium Priority | 2 |
| Low Priority | 0 |
| Demo Data Copied | ✅ |

## Demo Data Index

| Source | Destination | Description |
|--------|-------------|-------------|
| `repo/RNA-FM/redevelop/data/examples/example.fasta` | `examples/data/example.fasta` | Basic RNA sequences for testing |
| `repo/RNA-FM/tutorials/tutorial_data/RF00001.fasta` | `examples/data/RF00001.fasta` | 5S ribosomal RNA family sequences |
| `repo/RNA-FM/tutorials/tutorial_data/RF00005.fasta` | `examples/data/RF00005.fasta` | tRNA family sequences |
| `repo/RNA-FM/tutorials/tutorial_data/RF00010.fasta` | `examples/data/RF00010.fasta` | RNase P RNA sequences |
| `repo/RNA-FM/tutorials/tutorial_data/format_rnacentral_active.100.sample-Max50.fasta` | `examples/data/format_rnacentral_active.100.sample-Max50.fasta` | Large RNA sequence dataset for clustering |
| `repo/RNA-FM/redevelop/pretrained/extract_embedding.yml` | `examples/data/extract_embedding.yml` | Configuration for embedding extraction |
| `repo/RNA-FM/redevelop/pretrained/ss_prediction.yml` | `examples/data/ss_prediction.yml` | Configuration for structure prediction |

## Key Features Identified

### RNA-FM Capabilities:
1. **RNA sequence embeddings** - 640-dimensional contextual representations
2. **Secondary structure prediction** - Base-pair probability matrices
3. **RNA family classification** - Functional clustering analysis
4. **mRNA-specific analysis** - Codon-aware translation features
5. **UTR regulatory analysis** - 5' and 3' UTR functional elements

### Model Variants:
- **RNA-FM**: 12-layer model for ncRNA sequences (23.7M training sequences)
- **mRNA-FM**: 12-layer model for mRNA coding sequences (45M training sequences, codon tokenization)

### Compatible Use Case Filters:
- ✅ RNA sequence embedding
- ✅ ncRNA function prediction
- ✅ RNA secondary structure prediction
- ✅ mRNA translation efficiency prediction