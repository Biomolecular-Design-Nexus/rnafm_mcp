#!/usr/bin/env python3
"""
Script: rna_embeddings.py
Description: Extract RNA sequence embeddings using RNA-FM or mock generation

Original Use Case: examples/use_case_1_rna_embeddings.py
Dependencies Removed: Inlined FASTA parsing, output generation

Usage:
    python scripts/rna_embeddings.py --input <input_file> --output <output_file>

Example:
    python scripts/rna_embeddings.py --input examples/data/example.fasta --output results/embeddings
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

# PyTorch (lazy loaded for real embeddings)
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
    "model_type": "rna-fm",  # "rna-fm" or "mrna-fm"
    "embedding_dim": 640,    # RNA-FM embedding dimension
    "use_mock": False,       # Use mock embeddings if model unavailable
    "batch_size": 8,         # Batch size for processing
    "device": "cuda",        # "cuda" or "cpu"
    "normalize": False,      # Normalize embeddings
}

# ==============================================================================
# Core Functions (extracted from use case)
# ==============================================================================
def generate_mock_embeddings(sequences: List[Tuple[str, str]],
                           embedding_dim: int = 640) -> Dict[str, np.ndarray]:
    """Generate mock embeddings for testing without model download.

    Simplified from examples/use_case_1_rna_embeddings_mock.py

    Args:
        sequences: List of (sequence_name, sequence) tuples
        embedding_dim: Dimension of embeddings

    Returns:
        Dict mapping sequence names to embedding arrays
    """
    embeddings = {}

    for seq_name, seq_str in sequences:
        seq_len = len(seq_str)

        # Generate deterministic embeddings based on sequence composition
        np.random.seed(create_deterministic_seed(seq_str))

        # Mock embedding: position-wise features based on nucleotide composition
        seq_embeddings = []
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

            seq_embeddings.append(base_embedding)

        embeddings[seq_name] = np.array(seq_embeddings)

    return embeddings


def extract_real_embeddings(sequences: List[Tuple[str, str]],
                          model_type: str = "rna-fm",
                          batch_size: int = 8,
                          device: str = "cuda") -> Optional[Dict[str, np.ndarray]]:
    """Extract real RNA embeddings using RNA-FM model.

    Extracted from examples/use_case_1_rna_embeddings.py with lazy loading

    Args:
        sequences: List of (sequence_name, sequence) tuples
        model_type: "rna-fm" or "mrna-fm"
        batch_size: Batch size for processing
        device: "cuda" or "cpu"

    Returns:
        Dict mapping sequence names to embedding arrays, or None if failed
    """
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot extract real embeddings.")
        return None

    # Lazy import RNA-FM
    fm = lazy_import_fm()
    if fm is None:
        return None

    try:
        # Load the appropriate model
        if model_type == "mrna-fm":
            print("Loading mRNA-FM model...")
            model, alphabet = fm.pretrained.mrna_fm_t12()
            print("Note: mRNA-FM requires sequences with length divisible by 3 (codons)")
        else:
            print("Loading RNA-FM model...")
            model, alphabet = fm.pretrained.rna_fm_t12()

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
        return None

    # Extract embeddings
    embeddings = {}

    try:
        # Process sequences in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_data = [(name, seq) for name, seq in batch_sequences]

            # Convert to model input format
            _, _, batch_tokens = batch_converter(batch_data)

            if device == "cuda" and torch.cuda.is_available():
                batch_tokens = batch_tokens.cuda()

            # Extract embeddings
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
                batch_embeddings = results["representations"][33]

            # Store embeddings for each sequence
            for j, (seq_name, seq_str) in enumerate(batch_sequences):
                seq_len = len(seq_str)
                # Extract embeddings for actual sequence length (excluding special tokens)
                seq_embedding = batch_embeddings[j, 1:seq_len+1].cpu().numpy()
                embeddings[seq_name] = seq_embedding

            print(f"Processed batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")

    except Exception as e:
        print(f"Error during embedding extraction: {e}")
        return None

    return embeddings


def run_rna_embeddings(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for RNA embedding extraction.

    Args:
        input_file: Path to input FASTA file
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - embeddings: Dict of sequence embeddings
            - output_dir: Path to output directory (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_rna_embeddings("input.fasta", "output_dir")
        >>> print(result['metadata']['num_sequences'])
    """
    # Setup
    input_file = Path(input_file)
    config = merge_configs(DEFAULT_CONFIG, config, **kwargs)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"RNA-FM Embedding Extraction")
    print(f"===========================")
    print(f"Input: {input_file}")
    print(f"Model: {config['model_type']}")
    print(f"Mock mode: {config['use_mock']}")
    print()

    # Load sequences
    sequences = load_fasta(input_file)
    print(f"Loaded {len(sequences)} sequences")

    # Extract embeddings
    if config['use_mock']:
        print("Using mock embedding generation...")
        embeddings = generate_mock_embeddings(sequences, config['embedding_dim'])
        embedding_type = "Mock"
    else:
        print("Attempting real embedding extraction...")
        embeddings = extract_real_embeddings(
            sequences,
            config['model_type'],
            config['batch_size'],
            config['device']
        )

        if embeddings is None:
            print("Real embedding extraction failed. Falling back to mock generation...")
            embeddings = generate_mock_embeddings(sequences, config['embedding_dim'])
            embedding_type = "Mock (fallback)"
        else:
            embedding_type = "Real RNA-FM"

    # Normalize embeddings if requested
    if config.get('normalize', False):
        for seq_name in embeddings:
            embeddings[seq_name] = embeddings[seq_name] / np.linalg.norm(embeddings[seq_name], axis=1, keepdims=True)

    # Save output if requested
    output_dir = None
    if output_file:
        output_dir = ensure_output_dir(output_file)

        # Save individual embedding files
        for seq_name, embedding in embeddings.items():
            embedding_file = output_dir / f"{seq_name}_embeddings.npy"
            np.save(embedding_file, embedding)
            print(f"Saved embeddings for {seq_name}: {embedding.shape} -> {embedding_file}")

        # Save summary
        stats = sequence_stats(sequences)
        metadata = {
            "Model": f"{config['model_type']} ({embedding_type})",
            "Input file": str(input_file),
            "Number of sequences": len(sequences),
            "Embedding dimension": config['embedding_dim'],
            "Total sequence length": stats.get('total_length', 0),
            "GC content %": f"{stats.get('gc_content_percent', 0):.1f}",
            "Output directory": str(output_dir),
            "Config": config
        }

        summary_file = write_summary_file(
            output_dir,
            "embeddings_summary.txt",
            f"RNA Embedding Extraction Summary ({embedding_type})",
            metadata,
            sequences
        )
        print(f"Summary saved to {summary_file}")

    return {
        "embeddings": embeddings,
        "output_dir": str(output_dir) if output_dir else None,
        "metadata": {
            "input_file": str(input_file),
            "num_sequences": len(sequences),
            "embedding_type": embedding_type,
            "config": config,
            "embedding_shapes": {name: emb.shape for name, emb in embeddings.items()}
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
    parser.add_argument('--model-type', choices=['rna-fm', 'mrna-fm'],
                       default='rna-fm', help='Model type (default: rna-fm)')
    parser.add_argument('--use-mock', action='store_true',
                       help='Force use of mock embeddings')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize embeddings')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    overrides = {
        'model_type': args.model_type,
        'use_mock': args.use_mock,
        'batch_size': args.batch_size,
        'device': args.device,
        'normalize': args.normalize
    }

    # Run
    try:
        result = run_rna_embeddings(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        print(f"\n✅ Success!")
        print(f"Processed {result['metadata']['num_sequences']} sequences")
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