"""Shared I/O functions for RNA-FM MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any
import json
import os


def load_fasta(file_path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Load sequences from a FASTA file.

    Inlined from repo parsing logic with improvements.

    Args:
        file_path: Path to FASTA file

    Returns:
        List of (sequence_name, sequence) tuples

    Example:
        >>> sequences = load_fasta("examples/data/example.fasta")
        >>> print(len(sequences))
        3
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    sequences = []
    seq_name = None
    seq = ""

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if seq_name is not None:
                    sequences.append((seq_name, seq))
                # Start new sequence
                seq_name = line[1:]  # Remove '>'
                seq = ""
            else:
                seq += line.upper().replace('T', 'U')  # Convert DNA to RNA

    # Add the last sequence
    if seq_name is not None:
        sequences.append((seq_name, seq))

    if not sequences:
        raise ValueError(f"No sequences found in FASTA file: {file_path}")

    return sequences


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def ensure_output_dir(output_path: Union[str, Path]) -> Path:
    """Ensure output directory exists and return Path object."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_summary_file(
    output_dir: Union[str, Path],
    filename: str,
    title: str,
    metadata: Dict[str, Any],
    sequences: List[Tuple[str, str]] = None
) -> Path:
    """Write a summary file with processing results.

    Args:
        output_dir: Directory to write file
        filename: Name of summary file
        title: Title for the summary
        metadata: Processing metadata
        sequences: Optional sequence data for analysis

    Returns:
        Path to written summary file
    """
    summary_path = Path(output_dir) / filename

    with open(summary_path, 'w') as f:
        f.write(f"{title}\n")
        f.write(f"{'=' * len(title)}\n\n")

        # Write metadata
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Write sequence analysis if provided
        if sequences:
            f.write("Sequence Analysis:\n")
            f.write("==================\n")
            for seq_name, seq_str in sequences:
                a_count = seq_str.count('A')
                u_count = seq_str.count('U') + seq_str.count('T')
                g_count = seq_str.count('G')
                c_count = seq_str.count('C')
                gc_content = (g_count + c_count) / len(seq_str) * 100 if seq_str else 0
                f.write(f"{seq_name}: length={len(seq_str)}, A={a_count}, U={u_count}, G={g_count}, C={c_count}, GC%={gc_content:.1f}\n")

    return summary_path