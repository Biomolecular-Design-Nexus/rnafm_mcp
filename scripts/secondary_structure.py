#!/usr/bin/env python3
"""
Script: secondary_structure.py
Description: Predict RNA secondary structure using RNA-FM with ResNet predictor

Original Use Case: examples/use_case_2_secondary_structure.py
Dependencies Removed: Inlined FASTA parsing, structure output generation

Usage:
    python scripts/secondary_structure.py --input <input_file> --output <output_file>

Example:
    python scripts/secondary_structure.py --input examples/data/example.fasta --output results/structures
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
import json
import os
import sys

# Essential scientific packages
import numpy as np

# PyTorch (lazy loaded for real predictions)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Local utilities
from lib.io import load_fasta, ensure_output_dir, write_summary_file
from lib.utils import lazy_import_fm, sequence_stats, merge_configs, create_deterministic_seed

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "threshold": 0.5,        # Threshold for base-pair probability
    "use_mock": False,       # Use mock predictions if model unavailable
    "device": "cuda",        # "cuda" or "cpu"
    "output_formats": ["npy", "txt", "ct"],  # Output formats
}

# ==============================================================================
# Core Functions (extracted from use case)
# ==============================================================================
def generate_mock_structure_predictions(sequences: List[Tuple[str, str]],
                                       threshold: float = 0.5) -> Dict[str, Dict[str, Any]]:
    """Generate mock secondary structure predictions for testing.

    Args:
        sequences: List of (sequence_name, sequence) tuples
        threshold: Probability threshold for base pairing

    Returns:
        Dict mapping sequence names to prediction data
    """
    predictions = {}

    for seq_name, seq_str in sequences:
        seq_len = len(seq_str)

        # Generate mock probability matrix (symmetric)
        np.random.seed(create_deterministic_seed(seq_str))
        prob_matrix = np.random.random((seq_len, seq_len)) * 0.3  # Low baseline probabilities

        # Make matrix symmetric (required for base pairing)
        prob_matrix = (prob_matrix + prob_matrix.T) / 2

        # Set diagonal and near-diagonal to zero (no self-pairing)
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                prob_matrix[i, j] = 0

        # Add some realistic base-pair probabilities based on sequence
        for i in range(seq_len - 3):
            for j in range(i + 3, seq_len):
                # Check for Watson-Crick and wobble pairs
                base1, base2 = seq_str[i], seq_str[j]
                if (base1, base2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]:
                    prob_matrix[i, j] = prob_matrix[j, i] = np.random.uniform(0.3, 0.8)
                elif (base1, base2) in [('G', 'U'), ('U', 'G')]:  # Wobble pairs
                    prob_matrix[i, j] = prob_matrix[j, i] = np.random.uniform(0.2, 0.6)

        # Generate structure string based on threshold
        structure = ['.' for _ in range(seq_len)]
        pairs = []

        for i in range(seq_len):
            for j in range(i + 3, seq_len):
                if prob_matrix[i, j] > threshold and structure[i] == '.' and structure[j] == '.':
                    structure[i] = '('
                    structure[j] = ')'
                    pairs.append((i, j))

        predictions[seq_name] = {
            "probability_matrix": prob_matrix,
            "structure_string": ''.join(structure),
            "base_pairs": pairs,
            "threshold": threshold,
            "prediction_type": "mock"
        }

    return predictions


def predict_real_secondary_structure(sequences: List[Tuple[str, str]],
                                    threshold: float = 0.5,
                                    device: str = "cuda") -> Optional[Dict[str, Dict[str, Any]]]:
    """Predict real RNA secondary structure using RNA-FM + ResNet.

    Extracted from examples/use_case_2_secondary_structure.py with lazy loading

    Args:
        sequences: List of (sequence_name, sequence) tuples
        threshold: Probability threshold for base pairing
        device: "cuda" or "cpu"

    Returns:
        Dict mapping sequence names to prediction data, or None if failed
    """
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot predict real secondary structure.")
        return None

    # Lazy import RNA-FM
    fm = lazy_import_fm()
    if fm is None:
        return None

    try:
        # Load RNA-FM model with secondary structure predictor
        print("Loading RNA-FM model with secondary structure predictor...")
        model, alphabet = fm.downstream.build_rnafm_resnet(type="ss")
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # Disable dropout for deterministic results

        # Move model to device
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        elif device == "cpu":
            model = model.cpu()

        print(f"Model loaded successfully on {device}")

    except Exception as e:
        print(f"Error loading RNA-FM model: {e}")
        print("Note: This requires the pre-trained weights for secondary structure prediction")
        return None

    # Predict structures
    predictions = {}

    try:
        for seq_name, seq_str in sequences:
            seq_len = len(seq_str)

            # Convert sequence to model input format
            _, _, tokens = batch_converter([(seq_name, seq_str)])

            if device == "cuda" and torch.cuda.is_available():
                tokens = tokens.cuda()

            # Predict secondary structure
            with torch.no_grad():
                prob_matrix = model(tokens)[0, 1:seq_len+1, 1:seq_len+1].cpu().numpy()

            # Generate structure string based on threshold
            structure = ['.' for _ in range(seq_len)]
            pairs = []

            for i in range(seq_len):
                for j in range(i + 3, seq_len):
                    if prob_matrix[i, j] > threshold and structure[i] == '.' and structure[j] == '.':
                        structure[i] = '('
                        structure[j] = ')'
                        pairs.append((i, j))

            predictions[seq_name] = {
                "probability_matrix": prob_matrix,
                "structure_string": ''.join(structure),
                "base_pairs": pairs,
                "threshold": threshold,
                "prediction_type": "RNA-FM"
            }

            print(f"Predicted structure for {seq_name}: {len(pairs)} base pairs")

    except Exception as e:
        print(f"Error during structure prediction: {e}")
        return None

    return predictions


def save_structure_outputs(predictions: Dict[str, Dict[str, Any]],
                          sequences: List[Tuple[str, str]],
                          output_dir: Path,
                          output_formats: List[str]) -> Dict[str, List[str]]:
    """Save structure predictions in multiple formats.

    Args:
        predictions: Prediction results
        sequences: Original sequences
        output_dir: Output directory
        output_formats: List of formats to save ("npy", "txt", "ct")

    Returns:
        Dict mapping format to list of saved files
    """
    saved_files = {fmt: [] for fmt in output_formats}
    sequence_dict = {name: seq for name, seq in sequences}

    for seq_name, pred_data in predictions.items():
        seq_str = sequence_dict[seq_name]
        prob_matrix = pred_data["probability_matrix"]
        structure = pred_data["structure_string"]
        base_pairs = pred_data["base_pairs"]

        # Save probability matrix (.npy)
        if "npy" in output_formats:
            prob_file = output_dir / f"{seq_name}_probability_matrix.npy"
            np.save(prob_file, prob_matrix)
            saved_files["npy"].append(str(prob_file))

        # Save structure string (.txt)
        if "txt" in output_formats:
            txt_file = output_dir / f"{seq_name}_structure.txt"
            with open(txt_file, 'w') as f:
                f.write(f">{seq_name}\n")
                f.write(f"Sequence: {seq_str}\n")
                f.write(f"Structure: {structure}\n")
                f.write(f"Base pairs ({len(base_pairs)} total):\n")
                for i, j in base_pairs:
                    f.write(f"  {i+1}-{j+1}: {seq_str[i]}-{seq_str[j]} (prob: {prob_matrix[i,j]:.3f})\n")
            saved_files["txt"].append(str(txt_file))

        # Save CT format (Connect Table)
        if "ct" in output_formats:
            ct_file = output_dir / f"{seq_name}_structure.ct"
            with open(ct_file, 'w') as f:
                f.write(f"{len(seq_str)} {seq_name}\n")

                # Create pairing array
                pairing = [0] * len(seq_str)
                for i, j in base_pairs:
                    pairing[i] = j + 1  # CT format is 1-indexed
                    pairing[j] = i + 1

                # Write CT format
                for i, base in enumerate(seq_str):
                    f.write(f"{i+1} {base} {i} {i+2} {pairing[i]} {i+1}\n")
            saved_files["ct"].append(str(ct_file))

    return saved_files


def run_secondary_structure(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for RNA secondary structure prediction.

    Args:
        input_file: Path to input FASTA file
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - predictions: Dict of sequence predictions
            - output_dir: Path to output directory (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_secondary_structure("input.fasta", "output_dir")
        >>> print(result['metadata']['num_sequences'])
    """
    # Setup
    input_file = Path(input_file)
    config = merge_configs(DEFAULT_CONFIG, config, **kwargs)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"RNA Secondary Structure Prediction")
    print(f"==================================")
    print(f"Input: {input_file}")
    print(f"Threshold: {config['threshold']}")
    print(f"Mock mode: {config['use_mock']}")
    print()

    # Load sequences
    sequences = load_fasta(input_file)
    print(f"Loaded {len(sequences)} sequences")

    # Predict secondary structures
    if config['use_mock']:
        print("Using mock structure prediction...")
        predictions = generate_mock_structure_predictions(sequences, config['threshold'])
        prediction_type = "Mock"
    else:
        print("Attempting real structure prediction...")
        predictions = predict_real_secondary_structure(
            sequences,
            config['threshold'],
            config['device']
        )

        if predictions is None:
            print("Real structure prediction failed. Falling back to mock generation...")
            predictions = generate_mock_structure_predictions(sequences, config['threshold'])
            prediction_type = "Mock (fallback)"
        else:
            prediction_type = "RNA-FM + ResNet"

    # Save output if requested
    output_dir = None
    saved_files = {}
    if output_file:
        output_dir = ensure_output_dir(output_file)

        # Save structure predictions in multiple formats
        saved_files = save_structure_outputs(
            predictions,
            sequences,
            output_dir,
            config['output_formats']
        )

        # Count total base pairs
        total_pairs = sum(len(pred["base_pairs"]) for pred in predictions.values())

        # Print saved files summary
        for fmt, files in saved_files.items():
            print(f"Saved {len(files)} {fmt.upper()} files")

        # Save summary
        stats = sequence_stats(sequences)
        metadata = {
            "Predictor": f"RNA-FM ResNet ({prediction_type})",
            "Input file": str(input_file),
            "Number of sequences": len(sequences),
            "Total base pairs predicted": total_pairs,
            "Probability threshold": config['threshold'],
            "Total sequence length": stats.get('total_length', 0),
            "GC content %": f"{stats.get('gc_content_percent', 0):.1f}",
            "Output directory": str(output_dir),
            "Output formats": config['output_formats'],
            "Config": config
        }

        # Add per-sequence details
        for seq_name, pred_data in predictions.items():
            metadata[f"{seq_name}_pairs"] = len(pred_data["base_pairs"])
            metadata[f"{seq_name}_structure"] = pred_data["structure_string"]

        summary_file = write_summary_file(
            output_dir,
            "structure_summary.txt",
            f"RNA Secondary Structure Prediction Summary ({prediction_type})",
            metadata,
            sequences
        )
        print(f"Summary saved to {summary_file}")

    return {
        "predictions": predictions,
        "output_dir": str(output_dir) if output_dir else None,
        "metadata": {
            "input_file": str(input_file),
            "num_sequences": len(sequences),
            "prediction_type": prediction_type,
            "total_base_pairs": sum(len(pred["base_pairs"]) for pred in predictions.values()),
            "config": config,
            "saved_files": saved_files
        }
    }


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file path')
    parser.add_argument('--output', '-o', help='Output directory path')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Probability threshold for base pairing (default: 0.5)')
    parser.add_argument('--use-mock', action='store_true',
                       help='Force use of mock structure prediction')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--formats', nargs='+', choices=['npy', 'txt', 'ct'],
                       default=['npy', 'txt'], help='Output formats (default: npy txt)')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    overrides = {
        'threshold': args.threshold,
        'use_mock': args.use_mock,
        'device': args.device,
        'output_formats': args.formats
    }

    # Run
    try:
        result = run_secondary_structure(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        print(f"\n✅ Success!")
        print(f"Processed {result['metadata']['num_sequences']} sequences")
        print(f"Predicted {result['metadata']['total_base_pairs']} total base pairs")
        if result['output_dir']:
            print(f"Results saved to: {result['output_dir']}")
        else:
            print("Results returned (no output directory specified)")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())