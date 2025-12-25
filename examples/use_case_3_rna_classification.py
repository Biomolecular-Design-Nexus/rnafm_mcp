#!/usr/bin/env python3
"""
Use Case 3: RNA Family Clustering and Classification

This script performs RNA family clustering and classification using RNA-FM embeddings.
It demonstrates how to use pre-trained embeddings for downstream classification tasks.

Usage:
    python examples/use_case_3_rna_classification.py --input examples/data/format_rnacentral_active.100.sample-Max50.fasta --output results/classification

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

def classify_rna_families(input_fasta, output_dir, n_clusters=5):
    """
    Perform RNA family clustering using RNA-FM embeddings

    Args:
        input_fasta: Path to input FASTA file with RNA sequences
        output_dir: Directory to save classification results
        n_clusters: Number of clusters for K-means clustering
    """
    try:
        import fm
        print("Successfully imported RNA-FM module")
    except ImportError as e:
        print(f"Error importing RNA-FM: {e}")
        print("Please ensure you're running in the legacy environment: mamba activate ./env_py38")
        return False

    # Load RNA-FM model
    print("Loading RNA-FM model...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # Read FASTA file
    print(f"Reading sequences from {input_fasta}")
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
                    data.append((seq_name, seq))
                    sequence_info.append({'name': seq_name, 'length': len(seq), 'sequence': seq})
                seq_name = line[1:]
                seq = ""
            else:
                seq += line

        # Add the last sequence
        if seq_name is not None:
            data.append((seq_name, seq))
            sequence_info.append({'name': seq_name, 'length': len(seq), 'sequence': seq})

    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return False

    print(f"Loaded {len(data)} sequences")

    # Limit to reasonable number for clustering demo
    if len(data) > 100:
        print(f"Limiting to first 100 sequences for demonstration")
        data = data[:100]
        sequence_info = sequence_info[:100]

    # Extract embeddings in batches
    print("Extracting embeddings...")
    all_embeddings = []
    batch_size = 10

    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])

        embeddings = results["representations"][12]

        # Use mean pooling over sequence length (excluding special tokens)
        for j, seq_str in enumerate(batch_strs):
            seq_embedding = embeddings[j, 1:len(seq_str)+1].mean(dim=0)  # Mean over sequence
            all_embeddings.append(seq_embedding.numpy())

        print(f"Processed {min(i+batch_size, len(data))}/{len(data)} sequences")

    # Stack embeddings
    embeddings_matrix = np.stack(all_embeddings)
    print(f"Final embeddings matrix shape: {embeddings_matrix.shape}")

    # Perform clustering and classification
    os.makedirs(output_dir, exist_ok=True)

    # K-means clustering
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt

        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)

        # PCA for visualization
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings_scaled)

        # t-SNE for visualization (if not too many sequences)
        if len(embeddings_matrix) <= 50:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_matrix)-1))
            embeddings_tsne = tsne.fit_transform(embeddings_scaled)
        else:
            embeddings_tsne = None

        print("Clustering analysis completed successfully")

    except ImportError:
        print("Warning: scikit-learn or matplotlib not available. Skipping clustering analysis.")
        print("Install with: pip install scikit-learn matplotlib")
        cluster_labels = np.zeros(len(data))
        embeddings_pca = None
        embeddings_tsne = None

    # Save results
    print("Saving results...")

    # Save embeddings
    embeddings_file = os.path.join(output_dir, "rna_embeddings.npy")
    np.save(embeddings_file, embeddings_matrix)

    # Save cluster assignments
    clusters_file = os.path.join(output_dir, "cluster_assignments.txt")
    with open(clusters_file, 'w') as f:
        f.write("Sequence_Name\\tCluster\\tLength\\n")
        for i, info in enumerate(sequence_info):
            f.write(f"{info['name']}\\t{cluster_labels[i]}\\t{info['length']}\\n")

    # Generate cluster summary
    cluster_summary_file = os.path.join(output_dir, "cluster_summary.txt")
    with open(cluster_summary_file, 'w') as f:
        f.write("RNA Family Clustering Summary\\n")
        f.write("============================\\n")
        f.write(f"Number of sequences: {len(data)}\\n")
        f.write(f"Number of clusters: {n_clusters}\\n")
        f.write(f"Embedding dimension: {embeddings_matrix.shape[1]}\\n\\n")

        for cluster_id in range(n_clusters):
            cluster_seqs = [info for i, info in enumerate(sequence_info) if cluster_labels[i] == cluster_id]
            f.write(f"Cluster {cluster_id}: {len(cluster_seqs)} sequences\\n")

            if len(cluster_seqs) > 0:
                avg_length = np.mean([info['length'] for info in cluster_seqs])
                f.write(f"  Average length: {avg_length:.1f}\\n")

                # Show a few example sequences
                f.write("  Example sequences:\\n")
                for info in cluster_seqs[:3]:
                    f.write(f"    {info['name']} (length: {info['length']})\\n")
                if len(cluster_seqs) > 3:
                    f.write(f"    ... and {len(cluster_seqs)-3} more\\n")
                f.write("\\n")

    # Create visualizations if possible
    if embeddings_pca is not None:
        try:
            # PCA plot
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=cluster_labels, cmap='tab10')
            plt.title('RNA Sequences in PCA Space')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.colorbar(scatter, label='Cluster')

            if embeddings_tsne is not None:
                plt.subplot(1, 2, 2)
                scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=cluster_labels, cmap='tab10')
                plt.title('RNA Sequences in t-SNE Space')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.colorbar(scatter, label='Cluster')

            plt.tight_layout()
            plot_file = os.path.join(output_dir, "clustering_visualization.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Visualization saved to {plot_file}")

        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")

    # Save detailed sequence analysis
    detailed_file = os.path.join(output_dir, "detailed_analysis.txt")
    with open(detailed_file, 'w') as f:
        f.write("Detailed RNA Sequence Analysis\\n")
        f.write("==============================\\n\\n")

        for i, info in enumerate(sequence_info):
            f.write(f"Sequence: {info['name']}\\n")
            f.write(f"Cluster: {cluster_labels[i]}\\n")
            f.write(f"Length: {info['length']}\\n")
            f.write(f"Sequence: {info['sequence'][:50]}{'...' if len(info['sequence']) > 50 else ''}\\n")
            f.write("-" * 50 + "\\n")

    print(f"RNA classification completed! Results saved to {output_dir}")
    print(f"Files created:")
    print(f"  - Embeddings: {embeddings_file}")
    print(f"  - Cluster assignments: {clusters_file}")
    print(f"  - Cluster summary: {cluster_summary_file}")
    print(f"  - Detailed analysis: {detailed_file}")

    return True

def main():
    parser = argparse.ArgumentParser(description="RNA family clustering and classification using RNA-FM")
    parser.add_argument("--input", "-i", required=True,
                       help="Input FASTA file with RNA sequences")
    parser.add_argument("--output", "-o", default="results/classification",
                       help="Output directory for classification results")
    parser.add_argument("--clusters", "-c", type=int, default=5,
                       help="Number of clusters for K-means (default: 5)")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1

    # Check environment
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Error: PyTorch not found. Please activate the env_py38 environment.")
        return 1

    # Run classification
    success = classify_rna_families(args.input, args.output, args.clusters)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())