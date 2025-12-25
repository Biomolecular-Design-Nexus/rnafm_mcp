#!/usr/bin/env python3
"""
Use Case 2: RNA Secondary Structure Prediction

This script predicts RNA secondary structure (base-pairing patterns) using RNA-FM
with the ResNet predictor. Outputs base-pair probability matrices and secondary structures.

Usage:
    python examples/use_case_2_secondary_structure.py --input examples/data/example.fasta --output results/structures

Environment: ./env_py38 (Python 3.8 with PyTorch and RNA-FM dependencies)
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the repo path for imports
sys.path.append('repo/RNA-FM')

def predict_secondary_structure(input_fasta, output_dir, threshold=0.5):
    """
    Predict RNA secondary structure using RNA-FM + ResNet

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Directory to save predictions
        threshold: Threshold for base-pair probability (default: 0.5)
    """
    try:
        import fm
        print("Successfully imported RNA-FM module")
    except ImportError as e:
        print(f"Error importing RNA-FM: {e}")
        print("Please ensure you're running in the legacy environment: mamba activate ./env_py38")
        return False

    # Load RNA-FM model with secondary structure predictor
    print("Loading RNA-FM model with secondary structure predictor...")
    try:
        model, alphabet = fm.downstream.build_rnafm_resnet(type="ss")
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disable dropout for deterministic results
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: This requires the pre-trained weights for secondary structure prediction")
        return False

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
                seq += line.upper().replace('T', 'U')  # Convert DNA to RNA

        # Add the last sequence
        if seq_name is not None:
            data.append((seq_name, seq))

    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return False

    print(f"Loaded {len(data)} sequences")

    # Process each sequence
    os.makedirs(output_dir, exist_ok=True)

    for seq_name, seq_str in data:
        print(f"Processing {seq_name} (length: {len(seq_str)})")

        # Convert single sequence to batch format
        batch_labels, batch_strs, batch_tokens = batch_converter([(seq_name, seq_str)])

        # Prepare input
        input_data = {
            "description": batch_labels,
            "token": batch_tokens
        }

        # Predict secondary structure
        with torch.no_grad():
            results = model(input_data)

        # Get base-pair probability matrix
        ss_prob_map = results["r-ss"]  # Shape: [batch_size, seq_len, seq_len]
        ss_prob_map = ss_prob_map[0].numpy()  # Remove batch dimension

        print(f"Secondary structure probability matrix shape: {ss_prob_map.shape}")

        # Save probability matrix
        prob_matrix_file = os.path.join(output_dir, f"{seq_name}_prob_matrix.npy")
        np.save(prob_matrix_file, ss_prob_map)

        # Convert to contact map using threshold
        contact_map = (ss_prob_map > threshold).astype(int)

        # Save contact map
        contact_map_file = os.path.join(output_dir, f"{seq_name}_contact_map.npy")
        np.save(contact_map_file, contact_map)

        # Generate dot-bracket notation
        dot_bracket = predict_dot_bracket(ss_prob_map, threshold)

        # Save results in text format
        results_file = os.path.join(output_dir, f"{seq_name}_structure.txt")
        with open(results_file, 'w') as f:
            f.write(f">{seq_name}\n")
            f.write(f"Sequence: {seq_str}\n")
            f.write(f"Structure: {dot_bracket}\n")
            f.write(f"Length: {len(seq_str)}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Number of predicted base pairs: {np.sum(contact_map) // 2}\n\n")

        # Save in CT format (connectivity table)
        ct_file = os.path.join(output_dir, f"{seq_name}_structure.ct")
        save_ct_format(seq_str, contact_map, ct_file, seq_name)

        print(f"Saved results for {seq_name}:")
        print(f"  - Probability matrix: {prob_matrix_file}")
        print(f"  - Contact map: {contact_map_file}")
        print(f"  - Structure text: {results_file}")
        print(f"  - CT format: {ct_file}")
        print(f"  - Predicted structure: {dot_bracket}")

    # Save summary
    summary_file = os.path.join(output_dir, "structure_prediction_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"RNA Secondary Structure Prediction Summary\n")
        f.write(f"==========================================\n")
        f.write(f"Model: RNA-FM + ResNet\n")
        f.write(f"Input file: {input_fasta}\n")
        f.write(f"Number of sequences: {len(data)}\n")
        f.write(f"Probability threshold: {threshold}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        for seq_name, seq_str in data:
            f.write(f"{seq_name}: length={len(seq_str)}\n")

    print(f"Secondary structure prediction completed! Results saved to {output_dir}")
    print(f"Summary saved to {summary_file}")
    return True

def predict_dot_bracket(prob_matrix, threshold=0.5):
    """Convert probability matrix to dot-bracket notation"""
    seq_len = prob_matrix.shape[0]
    structure = ['.'] * seq_len

    # Find base pairs above threshold
    pairs = []
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if prob_matrix[i, j] > threshold:
                pairs.append((i, j, prob_matrix[i, j]))

    # Sort by probability and select non-conflicting pairs
    pairs.sort(key=lambda x: x[2], reverse=True)
    used_positions = set()

    for i, j, prob in pairs:
        if i not in used_positions and j not in used_positions:
            structure[i] = '('
            structure[j] = ')'
            used_positions.add(i)
            used_positions.add(j)

    return ''.join(structure)

def save_ct_format(sequence, contact_map, output_file, seq_name):
    """Save secondary structure in CT (connectivity table) format"""
    seq_len = len(sequence)

    # Find base pairs
    pairs = {}
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if contact_map[i, j] == 1:
                pairs[i] = j
                pairs[j] = i

    with open(output_file, 'w') as f:
        f.write(f"{seq_len} {seq_name}\n")
        for i in range(seq_len):
            pair_partner = pairs.get(i, 0)
            f.write(f"{i+1} {sequence[i]} {i} {i+2} {pair_partner+1 if pair_partner > 0 else 0} {i+1}\n")

def main():
    parser = argparse.ArgumentParser(description="Predict RNA secondary structure using RNA-FM")
    parser.add_argument("--input", "-i", required=True,
                       help="Input FASTA file with RNA sequences")
    parser.add_argument("--output", "-o", default="results/structures",
                       help="Output directory for structure predictions")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Threshold for base-pair probability (default: 0.5)")

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

    # Run secondary structure prediction
    success = predict_secondary_structure(args.input, args.output, args.threshold)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())