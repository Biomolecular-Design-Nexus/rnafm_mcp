#!/usr/bin/env python3
"""
Script: rna_classification.py
Description: Classify and cluster RNA sequences by functional families

Original Use Case: examples/use_case_3_rna_classification.py
Dependencies Removed: Inlined clustering logic, simplified analysis

Usage:
    python scripts/rna_classification.py --input <input_file> --output <output_file>

Example:
    python scripts/rna_classification.py --input examples/data/RF00005.fasta --output results/classification
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

# Optional scientific packages for clustering (with fallbacks)
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Local utilities
from lib.io import load_fasta, ensure_output_dir, write_summary_file
from lib.utils import lazy_import_fm, sequence_stats, merge_configs, create_deterministic_seed

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "num_clusters": 3,       # Number of clusters for K-means
    "use_mock": False,       # Use mock embeddings if model unavailable
    "device": "cuda",        # "cuda" or "cpu"
    "embedding_dim": 640,    # RNA-FM embedding dimension
    "pca_components": 50,    # PCA components for dimensionality reduction
    "clustering_method": "kmeans",  # Clustering method
}

# ==============================================================================
# Simple clustering fallback (if sklearn unavailable)
# ==============================================================================
def simple_kmeans(embeddings: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
    """Simple K-means implementation without sklearn dependency.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum iterations

    Returns:
        Array of cluster assignments
    """
    n_samples, n_features = embeddings.shape

    # Initialize centroids randomly
    np.random.seed(42)
    centroids = embeddings[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((embeddings - centroids[:, np.newaxis])**2).sum(axis=2))
        assignments = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([embeddings[assignments == i].mean(axis=0) for i in range(k)])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return assignments


# ==============================================================================
# Core Functions (extracted from use case)
# ==============================================================================
def extract_embeddings_for_classification(sequences: List[Tuple[str, str]],
                                         use_mock: bool = False,
                                         device: str = "cuda") -> np.ndarray:
    """Extract embeddings suitable for classification.

    Args:
        sequences: List of (sequence_name, sequence) tuples
        use_mock: Whether to use mock embeddings
        device: Device for real embeddings

    Returns:
        Embeddings array of shape (n_sequences, embedding_dim)
    """
    if use_mock or not TORCH_AVAILABLE:
        if not use_mock:
            print("PyTorch unavailable, using mock embeddings...")

        # Generate mock embeddings (sequence-level, not position-wise)
        embeddings = []
        for seq_name, seq_str in sequences:
            # Create deterministic embedding based on sequence properties
            np.random.seed(create_deterministic_seed(seq_str))

            # Base embedding
            embedding = np.random.normal(0, 1, 640)

            # Add sequence composition features
            seq_len = len(seq_str)
            a_frac = seq_str.count('A') / seq_len
            u_frac = seq_str.count('U') / seq_len
            g_frac = seq_str.count('G') / seq_len
            c_frac = seq_str.count('C') / seq_len
            gc_content = g_frac + c_frac

            # Modify embedding based on composition
            embedding[0:10] += a_frac * 2
            embedding[10:20] += u_frac * 2
            embedding[20:30] += g_frac * 2
            embedding[30:40] += c_frac * 2
            embedding[40:50] += gc_content * 2

            # Add length-based features
            length_norm = min(seq_len / 100.0, 1.0)
            embedding[50:60] += length_norm

            embeddings.append(embedding)

        return np.array(embeddings)

    else:
        # Use real RNA-FM embeddings
        fm = lazy_import_fm()
        if fm is None:
            print("RNA-FM unavailable, using mock embeddings...")
            return extract_embeddings_for_classification(sequences, use_mock=True)

        try:
            # Load RNA-FM model
            print("Loading RNA-FM model...")
            model, alphabet = fm.pretrained.rna_fm_t12()
            batch_converter = alphabet.get_batch_converter()
            model.eval()

            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()

            # Extract embeddings and pool to sequence level
            embeddings = []
            for seq_name, seq_str in sequences:
                _, _, tokens = batch_converter([(seq_name, seq_str)])
                if device == "cuda" and torch.cuda.is_available():
                    tokens = tokens.cuda()

                with torch.no_grad():
                    results = model(tokens, repr_layers=[33])
                    # Pool embeddings (mean over sequence length)
                    seq_embedding = results["representations"][33][0, 1:len(seq_str)+1].mean(dim=0)
                    embeddings.append(seq_embedding.cpu().numpy())

            return np.array(embeddings)

        except Exception as e:
            print(f"Error extracting real embeddings: {e}")
            print("Falling back to mock embeddings...")
            return extract_embeddings_for_classification(sequences, use_mock=True)


def perform_clustering(embeddings: np.ndarray,
                      num_clusters: int = 3,
                      pca_components: int = 50) -> Dict[str, Any]:
    """Perform clustering on RNA embeddings.

    Args:
        embeddings: Embeddings array
        num_clusters: Number of clusters
        pca_components: PCA components for dimensionality reduction

    Returns:
        Dict with clustering results
    """
    n_samples = embeddings.shape[0]

    # Apply PCA for dimensionality reduction if sklearn available
    if SKLEARN_AVAILABLE and embeddings.shape[1] > pca_components:
        pca = PCA(n_components=min(pca_components, n_samples-1))
        embeddings_reduced = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_ratio_.sum()
    else:
        embeddings_reduced = embeddings
        explained_variance = 1.0
        pca = None

    # Perform clustering
    if SKLEARN_AVAILABLE:
        # Use sklearn K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_reduced)

        # Calculate silhouette score if possible
        if n_samples > num_clusters:
            try:
                silhouette = silhouette_score(embeddings_reduced, cluster_labels)
            except:
                silhouette = -1.0
        else:
            silhouette = -1.0

        cluster_centers = kmeans.cluster_centers_

    else:
        # Use simple implementation
        print("scikit-learn unavailable, using simple clustering...")
        cluster_labels = simple_kmeans(embeddings_reduced, num_clusters)
        silhouette = -1.0  # Cannot compute without sklearn

        # Calculate cluster centers manually
        cluster_centers = np.array([
            embeddings_reduced[cluster_labels == i].mean(axis=0)
            for i in range(num_clusters)
        ])

    return {
        "cluster_labels": cluster_labels,
        "cluster_centers": cluster_centers,
        "embeddings_reduced": embeddings_reduced,
        "pca_explained_variance": explained_variance,
        "silhouette_score": silhouette,
        "pca_model": pca
    }


def analyze_clusters(sequences: List[Tuple[str, str]],
                    cluster_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze cluster characteristics.

    Args:
        sequences: Original sequences
        cluster_results: Clustering results

    Returns:
        Dict with cluster analysis
    """
    cluster_labels = cluster_results["cluster_labels"]
    num_clusters = len(set(cluster_labels))

    cluster_info = {}
    for cluster_id in range(num_clusters):
        # Get sequences in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_sequences = [(name, seq) for (name, seq), mask in zip(sequences, cluster_mask) if mask]

        # Calculate cluster statistics
        stats = sequence_stats(cluster_sequences)

        cluster_info[f"cluster_{cluster_id}"] = {
            "size": len(cluster_sequences),
            "sequence_names": [name for name, _ in cluster_sequences],
            "avg_length": stats.get('avg_length', 0),
            "gc_content": stats.get('gc_content_percent', 0),
            "length_range": [stats.get('min_length', 0), stats.get('max_length', 0)]
        }

    return {
        "num_clusters": num_clusters,
        "cluster_info": cluster_info,
        "silhouette_score": cluster_results.get("silhouette_score", -1)
    }


def run_rna_classification(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for RNA classification and clustering.

    Args:
        input_file: Path to input FASTA file
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - classifications: Cluster assignments
            - analysis: Cluster analysis results
            - output_dir: Path to output directory (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_rna_classification("input.fasta", "output_dir")
        >>> print(result['metadata']['num_clusters'])
    """
    # Setup
    input_file = Path(input_file)
    config = merge_configs(DEFAULT_CONFIG, config, **kwargs)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"RNA Family Classification")
    print(f"========================")
    print(f"Input: {input_file}")
    print(f"Number of clusters: {config['num_clusters']}")
    print(f"Mock mode: {config['use_mock']}")
    print()

    # Load sequences
    sequences = load_fasta(input_file)
    print(f"Loaded {len(sequences)} sequences")

    if len(sequences) < config['num_clusters']:
        raise ValueError(f"Cannot create {config['num_clusters']} clusters from {len(sequences)} sequences")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings_for_classification(
        sequences,
        config['use_mock'],
        config['device']
    )
    print(f"Extracted embeddings: {embeddings.shape}")

    # Perform clustering
    print("Performing clustering...")
    cluster_results = perform_clustering(
        embeddings,
        config['num_clusters'],
        config['pca_components']
    )

    # Analyze clusters
    print("Analyzing clusters...")
    analysis = analyze_clusters(sequences, cluster_results)

    # Create classification results
    classifications = {}
    for i, (seq_name, _) in enumerate(sequences):
        classifications[seq_name] = {
            "cluster": int(cluster_results["cluster_labels"][i]),
            "embedding": embeddings[i].tolist()  # For JSON serialization
        }

    # Save output if requested
    output_dir = None
    if output_file:
        output_dir = ensure_output_dir(output_file)

        # Save cluster assignments
        assignments_file = output_dir / "cluster_assignments.json"
        with open(assignments_file, 'w') as f:
            json.dump(classifications, f, indent=2)

        # Save embeddings
        embeddings_file = output_dir / "embeddings.npy"
        np.save(embeddings_file, embeddings)

        # Save reduced embeddings if PCA was used
        if cluster_results.get("pca_model") is not None:
            reduced_file = output_dir / "embeddings_pca.npy"
            np.save(reduced_file, cluster_results["embeddings_reduced"])

        # Save cluster analysis
        analysis_file = output_dir / "cluster_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Print cluster summary
        print("\nCluster Summary:")
        for cluster_id in range(config['num_clusters']):
            cluster_info = analysis["cluster_info"][f"cluster_{cluster_id}"]
            print(f"  Cluster {cluster_id}: {cluster_info['size']} sequences, "
                  f"avg_length={cluster_info['avg_length']:.1f}, "
                  f"GC%={cluster_info['gc_content']:.1f}")

        # Save summary
        stats = sequence_stats(sequences)
        metadata = {
            "Classification method": "RNA-FM + K-means clustering",
            "Input file": str(input_file),
            "Number of sequences": len(sequences),
            "Number of clusters": config['num_clusters'],
            "Embedding dimension": embeddings.shape[1],
            "PCA explained variance": f"{cluster_results.get('pca_explained_variance', 0):.3f}",
            "Silhouette score": f"{analysis.get('silhouette_score', -1):.3f}",
            "Total sequence length": stats.get('total_length', 0),
            "GC content %": f"{stats.get('gc_content_percent', 0):.1f}",
            "Output directory": str(output_dir),
            "Config": config
        }

        summary_file = write_summary_file(
            output_dir,
            "classification_summary.txt",
            f"RNA Classification Summary",
            metadata,
            sequences
        )
        print(f"Summary saved to {summary_file}")

    return {
        "classifications": classifications,
        "analysis": analysis,
        "embeddings": embeddings,
        "cluster_results": cluster_results,
        "output_dir": str(output_dir) if output_dir else None,
        "metadata": {
            "input_file": str(input_file),
            "num_sequences": len(sequences),
            "num_clusters": config['num_clusters'],
            "silhouette_score": analysis.get('silhouette_score', -1),
            "config": config
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
    parser.add_argument('--clusters', '-k', type=int, default=3,
                       help='Number of clusters (default: 3)')
    parser.add_argument('--use-mock', action='store_true',
                       help='Force use of mock embeddings')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--pca-components', type=int, default=50,
                       help='PCA components for dimensionality reduction (default: 50)')

    args = parser.parse_args()

    # Check for scikit-learn
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not available. Using simple clustering implementation.")

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    overrides = {
        'num_clusters': args.clusters,
        'use_mock': args.use_mock,
        'device': args.device,
        'pca_components': args.pca_components
    }

    # Run
    try:
        result = run_rna_classification(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        print(f"\n✅ Success!")
        print(f"Classified {result['metadata']['num_sequences']} sequences into {result['metadata']['num_clusters']} clusters")
        if result['metadata']['silhouette_score'] > 0:
            print(f"Silhouette score: {result['metadata']['silhouette_score']:.3f}")
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