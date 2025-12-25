#!/usr/bin/env python3
"""
Use Case 1: RNA Sequence Embedding Extraction (Mock Version)

This is a mock version that demonstrates the RNA embedding workflow without requiring
the large pretrained model download. It generates synthetic embeddings for testing.

Usage:
    python examples/use_case_1_rna_embeddings_mock.py --input examples/data/example.fasta --output results/embeddings

Environment: ./env_py38 (Python 3.8 with PyTorch)
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

def extract_mock_embeddings(input_fasta, output_dir):
    """
    Extract mock RNA embeddings (for testing without model download)

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Directory to save embeddings
    """
    print(f"Mock RNA-FM Embedding Extraction")
    print(f"================================")
    print(f"Note: This is a mock version that generates synthetic embeddings")
    print(f"for testing purposes. The real version requires downloading")
    print(f"the RNA-FM pretrained model (~1.1GB).")
    print()

    # Read FASTA file
    print(f"Reading sequences from {input_fasta}")
    data = []
    try:
        with open(input_fasta, 'r') as f:
            lines = f.readlines()

        seq_name = None
        seq = ""
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if seq_name is not None:
                    data.append((seq_name, seq))
                seq_name = line[1:]  # remove '>'
                seq = ""
            else:
                seq += line.upper().replace('T', 'U')  # Convert to RNA

        # Add the last sequence
        if seq_name is not None:
            data.append((seq_name, seq))

    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return False

    print(f"Loaded {len(data)} sequences")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate mock embeddings for each sequence
    embedding_dim = 640  # RNA-FM embedding dimension

    for seq_name, seq_str in data:
        # Create synthetic embeddings based on sequence properties
        seq_len = len(seq_str)

        # Generate deterministic "embeddings" based on sequence composition
        np.random.seed(hash(seq_str) % (2**32))  # Deterministic seed

        # Mock embedding: position-wise features based on nucleotide composition
        embeddings = []
        for i, nucleotide in enumerate(seq_str):
            # Create position and composition-aware features
            base_embedding = np.random.normal(0, 1, embedding_dim)

            # Add nucleotide-specific bias
            nucleotide_bias = {
                'A': 0.1, 'U': 0.2, 'G': 0.3, 'C': 0.4
            }.get(nucleotide, 0.0)
            base_embedding[:10] += nucleotide_bias

            # Add positional encoding
            pos_encoding = np.sin(i / seq_len * np.pi)
            base_embedding[10:20] += pos_encoding * 0.5

            # Add GC content bias
            gc_content = (seq_str.count('G') + seq_str.count('C')) / len(seq_str)
            base_embedding[20:30] += gc_content * 0.3

            embeddings.append(base_embedding)

        embeddings = np.array(embeddings)

        # Save embeddings
        output_file = os.path.join(output_dir, f"{seq_name}_embeddings.npy")
        np.save(output_file, embeddings)
        print(f"Generated mock embeddings for {seq_name}: {embeddings.shape} -> {output_file}")

    # Save summary
    summary_file = os.path.join(output_dir, "embeddings_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"RNA Embedding Extraction Summary (Mock Version)\n")
        f.write(f"===============================================\n")
        f.write(f"Model: RNA-FM (Mock)\n")
        f.write(f"Input file: {input_fasta}\n")
        f.write(f"Number of sequences: {len(data)}\n")
        f.write(f"Embedding dimension: {embedding_dim}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        f.write(f"Note: These are synthetic embeddings for testing.\n")
        f.write(f"Real embeddings require the RNA-FM pretrained model.\n\n")

        for seq_name, seq_str in data:
            f.write(f"{seq_name}: length={len(seq_str)}, embeddings_shape=({len(seq_str)}, {embedding_dim})\n")

        # Add sequence composition analysis
        f.write(f"\nSequence Analysis:\n")
        f.write(f"==================\n")
        for seq_name, seq_str in data:
            a_count = seq_str.count('A')
            u_count = seq_str.count('U') + seq_str.count('T')  # Count both U and T
            g_count = seq_str.count('G')
            c_count = seq_str.count('C')
            gc_content = (g_count + c_count) / len(seq_str) * 100 if len(seq_str) > 0 else 0
            f.write(f"{seq_name}: A={a_count}, U={u_count}, G={g_count}, C={c_count}, GC%={gc_content:.1f}\n")

    print(f"\nMock embedding extraction completed!")
    print(f"Results saved to {output_dir}")
    print(f"Summary saved to {summary_file}")
    print(f"\nTo use real RNA-FM embeddings, ensure the pretrained model is downloaded")
    print(f"and use the original use_case_1_rna_embeddings.py script.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract mock RNA sequence embeddings")
    parser.add_argument("--input", "-i", required=True,
                       help="Input FASTA file with RNA sequences")
    parser.add_argument("--output", "-o", default="results/embeddings",
                       help="Output directory for embeddings")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1

    # Run mock embedding extraction
    success = extract_mock_embeddings(args.input, args.output)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())