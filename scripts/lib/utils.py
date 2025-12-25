"""General utilities for RNA-FM MCP scripts.

Common functions extracted from use cases to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Dict, Any, Optional
import sys
import importlib.util
import hashlib


def get_script_root() -> Path:
    """Get the root directory of the MCP project."""
    # This file is in scripts/lib/, so go up 2 levels
    return Path(__file__).parent.parent.parent


def get_repo_path() -> Path:
    """Get the path to the RNA-FM repository."""
    return get_script_root() / "repo" / "RNA-FM"


def lazy_import_fm():
    """Lazy import of RNA-FM module to minimize startup time.

    Returns:
        fm module or None if import fails
    """
    repo_path = str(get_repo_path())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        import fm
        return fm
    except ImportError as e:
        print(f"Error importing RNA-FM: {e}")
        print(f"Please ensure RNA-FM repo is at: {repo_path}")
        print(f"And you're using the correct environment: mamba activate ./env_py38")
        return None


def validate_rna_sequence(sequence: str) -> bool:
    """Validate that a sequence contains only valid RNA nucleotides.

    Args:
        sequence: RNA sequence string

    Returns:
        True if valid, False otherwise
    """
    valid_chars = set('AUCG')
    return all(char in valid_chars for char in sequence.upper())


def sequence_stats(sequences: list) -> Dict[str, Any]:
    """Calculate statistics for a list of sequences.

    Args:
        sequences: List of (name, sequence) tuples

    Returns:
        Dict with statistics
    """
    if not sequences:
        return {}

    lengths = [len(seq) for _, seq in sequences]
    total_length = sum(lengths)

    # Count nucleotides across all sequences
    total_counts = {'A': 0, 'U': 0, 'G': 0, 'C': 0}
    for _, seq in sequences:
        for base in seq:
            if base in total_counts:
                total_counts[base] += 1

    gc_content = (total_counts['G'] + total_counts['C']) / total_length * 100 if total_length > 0 else 0

    return {
        'num_sequences': len(sequences),
        'total_length': total_length,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'avg_length': total_length / len(lengths) if lengths else 0,
        'nucleotide_counts': total_counts,
        'gc_content_percent': gc_content
    }


def create_deterministic_seed(text: str) -> int:
    """Create a deterministic seed from text for reproducible results.

    Args:
        text: Input text

    Returns:
        Integer seed
    """
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**32)


def merge_configs(default_config: Dict[str, Any],
                  user_config: Optional[Dict[str, Any]] = None,
                  **kwargs) -> Dict[str, Any]:
    """Merge configuration dictionaries with override precedence.

    Args:
        default_config: Default configuration
        user_config: User-provided config (optional)
        **kwargs: Individual parameter overrides

    Returns:
        Merged configuration
    """
    config = default_config.copy()
    if user_config:
        config.update(user_config)
    config.update(kwargs)
    return config