#!/usr/bin/env python3
"""
Use Case 4: mRNA Translation Efficiency Prediction

This script demonstrates using mRNA-FM for predicting translation efficiency and
expression levels from mRNA coding sequences (CDS). It shows how the mRNA-specific
model can capture features related to codon usage and translation.

Usage:
    python examples/use_case_4_mrna_expression.py --input examples/data/example.fasta --output results/mrna_analysis

Environment: ./env_py38 (Python 3.8 with PyTorch and RNA-FM dependencies)
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the repo path for imports
sys.path.append('repo/RNA-FM/redevelop')

def analyze_mrna_sequences(input_fasta, output_dir):
    """
    Analyze mRNA sequences for translation-related features using mRNA-FM

    Args:
        input_fasta: Path to input FASTA file with mRNA coding sequences
        output_dir: Directory to save analysis results
    """
    try:
        import fm
        print("Successfully imported RNA-FM module")
    except ImportError as e:
        print(f"Error importing RNA-FM: {e}")
        print("Please ensure you're running in the legacy environment: mamba activate ./env_py38")
        return False

    # Load mRNA-FM model
    print("Loading mRNA-FM model...")
    try:
        model, alphabet = fm.pretrained.mrna_fm_t12()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        print("mRNA-FM model loaded successfully")
    except Exception as e:
        print(f"Error loading mRNA-FM model: {e}")
        return False

    # Read FASTA file and validate sequences
    print(f"Reading mRNA sequences from {input_fasta}")
    data = []
    sequence_info = []

    try:
        with open(input_fasta, 'r') as f:
            lines = f.readlines()

        seq_name = None
        seq = ""
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if seq_name is not None:
                    # Process the previous sequence
                    process_sequence(seq_name, seq, data, sequence_info)
                seq_name = line[1:]
                seq = ""
            else:
                seq += line.upper().replace('T', 'U')  # Convert DNA to RNA

        # Process the last sequence
        if seq_name is not None:
            process_sequence(seq_name, seq, data, sequence_info)

    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return False

    if len(data) == 0:
        print("Error: No valid mRNA sequences found (sequences must be divisible by 3)")
        return False

    print(f"Loaded {len(data)} valid mRNA sequences")

    # Extract mRNA-specific embeddings
    print("Extracting mRNA embeddings...")
    all_embeddings = []
    codon_features = []

    # Process sequences in batches
    batch_size = 5  # Smaller batch size for mRNA-FM
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])

        embeddings = results["representations"][12]

        # Extract embeddings for this batch
        for j, seq_str in enumerate(batch_strs):
            # Use codon-level embeddings (mRNA-FM uses codon tokenization)
            seq_embedding = embeddings[j, 1:len(seq_str)//3+1]  # Exclude BOS/EOS, codon-level

            # Compute various pooling strategies
            mean_embedding = seq_embedding.mean(dim=0)
            max_embedding = seq_embedding.max(dim=0)[0]
            std_embedding = seq_embedding.std(dim=0)

            all_embeddings.append({
                'sequence': seq_embedding.numpy(),
                'mean': mean_embedding.numpy(),
                'max': max_embedding.numpy(),
                'std': std_embedding.numpy()
            })

            # Analyze codon usage
            sequence = batch_strs[j]
            codon_analysis = analyze_codon_usage(sequence)
            codon_features.append(codon_analysis)

        print(f"Processed {min(i+batch_size, len(data))}/{len(data)} sequences")

    # Analyze translation efficiency features
    print("Analyzing translation efficiency features...")
    os.makedirs(output_dir, exist_ok=True)

    # Compile results
    results = []
    for i, info in enumerate(sequence_info):
        result = {
            'name': info['name'],
            'length': info['length'],
            'sequence': info['sequence'],
            'num_codons': info['length'] // 3,
            'embeddings': all_embeddings[i],
            'codon_features': codon_features[i]
        }
        results.append(result)

    # Save detailed analysis
    save_mrna_analysis(results, output_dir)

    print(f"mRNA analysis completed! Results saved to {output_dir}")
    return True

def process_sequence(seq_name, seq, data, sequence_info):
    """Process and validate a single mRNA sequence"""
    # Remove whitespace
    seq = seq.replace(' ', '').replace('\\n', '')

    # Check if sequence is valid for mRNA-FM (length divisible by 3)
    if len(seq) % 3 != 0:
        print(f"Warning: Skipping {seq_name} - length {len(seq)} not divisible by 3")
        return

    # Check for valid nucleotides
    valid_nucs = set('ACGU')
    if not all(nuc in valid_nucs for nuc in seq):
        print(f"Warning: Skipping {seq_name} - contains invalid nucleotides")
        return

    # Check minimum length
    if len(seq) < 21:  # At least 7 codons
        print(f"Warning: Skipping {seq_name} - too short ({len(seq)} nucleotides)")
        return

    data.append((seq_name, seq))
    sequence_info.append({
        'name': seq_name,
        'length': len(seq),
        'sequence': seq
    })

def analyze_codon_usage(sequence):
    """Analyze codon usage patterns in the mRNA sequence"""
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]

    # Standard genetic code
    genetic_code = {
        'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
        'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
        'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    # Count codons and amino acids
    codon_counts = {}
    aa_counts = {}

    for codon in codons:
        codon_counts[codon] = codon_counts.get(codon, 0) + 1
        if codon in genetic_code:
            aa = genetic_code[codon]
            aa_counts[aa] = aa_counts.get(aa, 0) + 1

    # Calculate features
    gc_content = sum(1 for nuc in sequence if nuc in 'GC') / len(sequence)

    # Codon adaptation index (simplified)
    rare_codons = ['CGA', 'CGG', 'AGA', 'AGG', 'CUA', 'CUG', 'UUA', 'UUG']
    rare_codon_count = sum(codon_counts.get(codon, 0) for codon in rare_codons)
    rare_codon_frequency = rare_codon_count / len(codons) if len(codons) > 0 else 0

    return {
        'total_codons': len(codons),
        'gc_content': gc_content,
        'rare_codon_frequency': rare_codon_frequency,
        'unique_codons': len(codon_counts),
        'stop_codons': codon_counts.get('UAA', 0) + codon_counts.get('UAG', 0) + codon_counts.get('UGA', 0),
        'start_codons': codon_counts.get('AUG', 0),
        'codon_counts': codon_counts,
        'aa_counts': aa_counts
    }

def save_mrna_analysis(results, output_dir):
    """Save comprehensive mRNA analysis results"""

    # Save embeddings
    embeddings_file = os.path.join(output_dir, "mrna_embeddings.npz")
    embeddings_data = {}
    for i, result in enumerate(results):
        embeddings_data[f"seq_{i}_mean"] = result['embeddings']['mean']
        embeddings_data[f"seq_{i}_max"] = result['embeddings']['max']
        embeddings_data[f"seq_{i}_std"] = result['embeddings']['std']
    np.savez(embeddings_file, **embeddings_data)

    # Save detailed analysis
    analysis_file = os.path.join(output_dir, "mrna_analysis.txt")
    with open(analysis_file, 'w') as f:
        f.write("mRNA Translation Efficiency Analysis\\n")
        f.write("===================================\\n\\n")
        f.write(f"Number of sequences analyzed: {len(results)}\\n")
        f.write(f"Model: mRNA-FM (codon-aware)\\n\\n")

        for result in results:
            f.write(f"Sequence: {result['name']}\\n")
            f.write(f"Length: {result['length']} nucleotides ({result['num_codons']} codons)\\n")
            f.write(f"GC content: {result['codon_features']['gc_content']:.3f}\\n")
            f.write(f"Rare codon frequency: {result['codon_features']['rare_codon_frequency']:.3f}\\n")
            f.write(f"Unique codons: {result['codon_features']['unique_codons']}\\n")
            f.write(f"Start codons (AUG): {result['codon_features']['start_codons']}\\n")
            f.write(f"Stop codons: {result['codon_features']['stop_codons']}\\n")

            # Embedding statistics
            mean_emb = result['embeddings']['mean']
            f.write(f"Embedding statistics:\\n")
            f.write(f"  Mean embedding norm: {np.linalg.norm(mean_emb):.3f}\\n")
            f.write(f"  Embedding mean: {np.mean(mean_emb):.3f}\\n")
            f.write(f"  Embedding std: {np.std(mean_emb):.3f}\\n")
            f.write("-" * 60 + "\\n")

    # Save codon usage summary
    codon_summary_file = os.path.join(output_dir, "codon_usage_summary.txt")
    with open(codon_summary_file, 'w') as f:
        f.write("Codon Usage Summary\\n")
        f.write("==================\\n\\n")

        # Aggregate codon usage across all sequences
        total_codon_counts = {}
        for result in results:
            for codon, count in result['codon_features']['codon_counts'].items():
                total_codon_counts[codon] = total_codon_counts.get(codon, 0) + count

        total_codons = sum(total_codon_counts.values())

        f.write(f"Total codons analyzed: {total_codons}\\n")
        f.write(f"Unique codons found: {len(total_codon_counts)}\\n\\n")
        f.write("Codon frequencies:\\n")

        for codon in sorted(total_codon_counts.keys()):
            frequency = total_codon_counts[codon] / total_codons
            f.write(f"{codon}: {total_codon_counts[codon]:6d} ({frequency:.4f})\\n")

    # Save machine-readable data
    csv_file = os.path.join(output_dir, "mrna_features.csv")
    with open(csv_file, 'w') as f:
        f.write("sequence_name,length,num_codons,gc_content,rare_codon_freq,unique_codons,start_codons,stop_codons,embedding_norm\\n")

        for result in results:
            codon_features = result['codon_features']
            embedding_norm = np.linalg.norm(result['embeddings']['mean'])

            f.write(f"{result['name']},{result['length']},{result['num_codons']},")
            f.write(f"{codon_features['gc_content']:.4f},{codon_features['rare_codon_frequency']:.4f},")
            f.write(f"{codon_features['unique_codons']},{codon_features['start_codons']},")
            f.write(f"{codon_features['stop_codons']},{embedding_norm:.4f}\\n")

    print(f"Analysis files saved:")
    print(f"  - Embeddings: {embeddings_file}")
    print(f"  - Detailed analysis: {analysis_file}")
    print(f"  - Codon usage: {codon_summary_file}")
    print(f"  - CSV features: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="mRNA translation efficiency analysis using mRNA-FM")
    parser.add_argument("--input", "-i", required=True,
                       help="Input FASTA file with mRNA coding sequences")
    parser.add_argument("--output", "-o", default="results/mrna_analysis",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1

    print("mRNA Translation Efficiency Analysis")
    print("====================================")
    print("Note: This analysis requires mRNA coding sequences with length divisible by 3")
    print("The mRNA-FM model uses codon-level tokenization for translation-specific features\\n")

    # Check environment
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Error: PyTorch not found. Please activate the env_py38 environment.")
        return 1

    # Run mRNA analysis
    success = analyze_mrna_sequences(args.input, args.output)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())