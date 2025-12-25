#!/usr/bin/env python3
"""
Use Case 1: RNA Sequence Embedding Extraction

This script extracts deep learning embeddings from RNA sequences using RNA-FM.
The embeddings capture structural and functional information from RNA sequences.

Usage:
    python examples/use_case_1_rna_embeddings.py --input examples/data/example.fasta --output results/embeddings

Environment: ./env_py38 (Python 3.8 with PyTorch and RNA-FM dependencies)
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add the repo path for imports
sys.path.append('repo/RNA-FM')

def extract_embeddings(input_fasta, output_dir, model_type="rna-fm", use_mRNA=False):
    """
    Extract RNA embeddings using RNA-FM or mRNA-FM

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Directory to save embeddings
        model_type: Model type ("rna-fm" or "mrna-fm")
        use_mRNA: Whether to use mRNA-FM variant (for coding sequences)
    """
    try:
        import fm
        print("Successfully imported RNA-FM module")
    except ImportError as e:
        print(f"Error importing RNA-FM: {e}")
        print("Please ensure you're running in the legacy environment: mamba activate ./env_py38")
        return False

    # Load the appropriate model
    if use_mRNA:
        print("Loading mRNA-FM model...")
        model, alphabet = fm.pretrained.mrna_fm_t12()
        print("Note: mRNA-FM requires sequences with length divisible by 3 (codons)")
    else:
        print("Loading RNA-FM model...")
        model, alphabet = fm.pretrained.rna_fm_t12()

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disable dropout for deterministic results

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
                    # Check sequence length for mRNA-FM
                    if use_mRNA and len(seq) % 3 != 0:
                        print(f"Warning: Skipping {seq_name} - mRNA-FM requires sequence length divisible by 3")
                    else:
                        data.append((seq_name, seq))
                seq_name = line[1:]  # remove '>'
                seq = ""
            else:
                seq += line

        # Add the last sequence
        if seq_name is not None:
            if use_mRNA and len(seq) % 3 != 0:
                print(f"Warning: Skipping {seq_name} - mRNA-FM requires sequence length divisible by 3")
            else:
                data.append((seq_name, seq))

    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return False

    print(f"Loaded {len(data)} sequences")

    # Convert to batch format
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract embeddings
    print("Extracting embeddings...")
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])

    # Get embeddings from last layer
    embeddings = results["representations"][12]
    print(f"Extracted embeddings shape: {embeddings.shape}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as numpy arrays
    import numpy as np
    embeddings_np = embeddings.numpy()

    for i, (seq_name, seq_str) in enumerate(zip(batch_labels, batch_strs)):
        seq_embeddings = embeddings_np[i, 1:len(seq_str)+1]  # Remove BOS/EOS tokens
        output_file = os.path.join(output_dir, f"{seq_name}_embeddings.npy")
        np.save(output_file, seq_embeddings)
        print(f"Saved embeddings for {seq_name}: {seq_embeddings.shape} -> {output_file}")

    # Save summary
    summary_file = os.path.join(output_dir, "embeddings_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"RNA Embedding Extraction Summary\n")
        f.write(f"================================\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Input file: {input_fasta}\n")
        f.write(f"Number of sequences: {len(data)}\n")
        f.write(f"Embedding dimension: {embeddings.shape[-1]}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        for i, (seq_name, seq_str) in enumerate(zip(batch_labels, batch_strs)):
            f.write(f"{seq_name}: length={len(seq_str)}, embeddings_shape=({len(seq_str)}, {embeddings.shape[-1]})\n")

    print(f"Embedding extraction completed! Results saved to {output_dir}")
    print(f"Summary saved to {summary_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract RNA sequence embeddings using RNA-FM")
    parser.add_argument("--input", "-i", required=True,
                       help="Input FASTA file with RNA sequences")
    parser.add_argument("--output", "-o", default="results/embeddings",
                       help="Output directory for embeddings")
    parser.add_argument("--model", choices=["rna-fm", "mrna-fm"], default="rna-fm",
                       help="Model type to use (rna-fm for ncRNA, mrna-fm for coding sequences)")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1

    # Check environment
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("Error: PyTorch not found. Please activate the env_py38 environment.")
        return 1

    # Run embedding extraction
    use_mRNA = (args.model == "mrna-fm")
    success = extract_embeddings(args.input, args.output, args.model, use_mRNA)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())