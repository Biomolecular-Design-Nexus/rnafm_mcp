#!/usr/bin/env python3
"""
Use Case 5: UTR Function Prediction

This script demonstrates using RNA-FM for predicting functional properties of
untranslated regions (UTRs) in mRNAs. UTRs play crucial roles in gene regulation,
affecting mRNA stability, translation efficiency, and localization.

Usage:
    python examples/use_case_5_utr_function.py --input examples/data/example.fasta --output results/utr_analysis

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

def analyze_utr_function(input_fasta, output_dir, region_type="5UTR"):
    """
    Analyze UTR sequences for functional features using RNA-FM

    Args:
        input_fasta: Path to input FASTA file with UTR sequences
        output_dir: Directory to save analysis results
        region_type: Type of UTR region ("5UTR" or "3UTR")
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
    print(f"Reading {region_type} sequences from {input_fasta}")
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
                    # Process sequence
                    seq_clean = seq.upper().replace('T', 'U').replace(' ', '')
                    # Check for valid nucleotides
                    if all(nuc in 'ACGU' for nuc in seq_clean) and len(seq_clean) > 10:
                        data.append((seq_name, seq_clean))
                        sequence_info.append({
                            'name': seq_name,
                            'length': len(seq_clean),
                            'sequence': seq_clean,
                            'type': region_type
                        })
                    else:
                        print(f"Warning: Skipping {seq_name} - invalid sequence or too short")

                seq_name = line[1:]
                seq = ""
            else:
                seq += line

        # Process the last sequence
        if seq_name is not None:
            seq_clean = seq.upper().replace('T', 'U').replace(' ', '')
            if all(nuc in 'ACGU' for nuc in seq_clean) and len(seq_clean) > 10:
                data.append((seq_name, seq_clean))
                sequence_info.append({
                    'name': seq_name,
                    'length': len(seq_clean),
                    'sequence': seq_clean,
                    'type': region_type
                })

    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return False

    print(f"Loaded {len(data)} valid sequences")

    if len(data) == 0:
        print("No valid sequences found")
        return False

    # Extract embeddings and analyze UTR features
    print("Extracting embeddings and analyzing UTR features...")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    batch_size = 10

    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

        with torch.no_grad():
            batch_results = model(batch_tokens, repr_layers=[12])

        embeddings = batch_results["representations"][12]

        # Process each sequence in the batch
        for j, (seq_name, seq_str) in enumerate(zip(batch_labels, batch_strs)):
            # Extract sequence-level embedding (mean pooling)
            seq_embedding = embeddings[j, 1:len(seq_str)+1]  # Remove BOS/EOS tokens
            mean_embedding = seq_embedding.mean(dim=0).numpy()
            max_embedding = seq_embedding.max(dim=0)[0].numpy()

            # Analyze UTR-specific features
            utr_features = analyze_utr_features(seq_str, region_type)

            # Predict regulatory elements
            regulatory_elements = predict_regulatory_elements(seq_str, region_type)

            result = {
                'name': seq_name,
                'sequence': seq_str,
                'length': len(seq_str),
                'type': region_type,
                'embeddings': {
                    'mean': mean_embedding,
                    'max': max_embedding,
                    'full': seq_embedding.numpy()
                },
                'utr_features': utr_features,
                'regulatory_elements': regulatory_elements
            }
            results.append(result)

        print(f"Processed {min(i+batch_size, len(data))}/{len(data)} sequences")

    # Save comprehensive analysis
    save_utr_analysis(results, output_dir, region_type)

    print(f"UTR function analysis completed! Results saved to {output_dir}")
    return True

def analyze_utr_features(sequence, region_type):
    """Analyze UTR-specific sequence features"""
    features = {}

    # Basic composition
    features['length'] = len(sequence)
    features['gc_content'] = (sequence.count('G') + sequence.count('C')) / len(sequence)
    features['au_content'] = (sequence.count('A') + sequence.count('U')) / len(sequence)

    # Nucleotide frequencies
    for nuc in 'ACGU':
        features[f'{nuc.lower()}_content'] = sequence.count(nuc) / len(sequence)

    # Dinucleotide frequencies
    dinucs = ['AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'CU',
              'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU']
    for dinuc in dinucs:
        count = 0
        for i in range(len(sequence) - 1):
            if sequence[i:i+2] == dinuc:
                count += 1
        features[f'dinuc_{dinuc.lower()}'] = count / max(1, len(sequence) - 1)

    # Region-specific features
    if region_type == "5UTR":
        # 5' UTR specific features
        features['has_kozak'] = check_kozak_sequence(sequence)
        features['uorf_count'] = count_upstream_orfs(sequence)
        features['riboswitch_like'] = check_riboswitch_motifs(sequence)

    elif region_type == "3UTR":
        # 3' UTR specific features
        features['polya_signals'] = count_polya_signals(sequence)
        features['au_rich_elements'] = count_au_rich_elements(sequence)
        features['mirna_sites'] = count_potential_mirna_sites(sequence)

    # Secondary structure propensity (simplified)
    features['stem_propensity'] = calculate_stem_propensity(sequence)

    return features

def predict_regulatory_elements(sequence, region_type):
    """Predict regulatory elements in UTR sequences"""
    elements = {}

    if region_type == "5UTR":
        # 5' UTR regulatory elements
        elements['ribosome_binding_sites'] = find_ribosome_binding_sites(sequence)
        elements['iron_responsive_elements'] = find_ire_motifs(sequence)
        elements['internal_ribosome_entry_sites'] = find_ires_like_sequences(sequence)

    elif region_type == "3UTR":
        # 3' UTR regulatory elements
        elements['au_rich_elements'] = find_au_rich_elements(sequence)
        elements['polyadenylation_signals'] = find_polya_signals(sequence)
        elements['cytoplasmic_polyadenylation_elements'] = find_cpe_motifs(sequence)

    # Common elements in both UTRs
    elements['hairpin_loops'] = predict_hairpin_loops(sequence)
    elements['g_quadruplexes'] = predict_g_quadruplexes(sequence)

    return elements

def check_kozak_sequence(sequence):
    """Check for Kozak consensus sequence around AUG"""
    kozak_pattern = r'[AG]CC[AG]CCAUGG'
    import re
    return bool(re.search(kozak_pattern, sequence))

def count_upstream_orfs(sequence):
    """Count potential upstream open reading frames"""
    count = 0
    start_codons = ['AUG']

    for start_codon in start_codons:
        pos = sequence.find(start_codon)
        while pos != -1:
            # Check if there's a stop codon downstream
            for i in range(pos + 3, len(sequence) - 2, 3):
                codon = sequence[i:i+3]
                if codon in ['UAG', 'UAA', 'UGA']:
                    if i - pos >= 30:  # Minimum ORF length
                        count += 1
                    break
            pos = sequence.find(start_codon, pos + 1)

    return count

def check_riboswitch_motifs(sequence):
    """Check for riboswitch-like structural motifs"""
    # Simplified check for purine riboswitch-like sequences
    purine_motifs = ['GGACC', 'GGUCC', 'GAACC']
    return any(motif in sequence for motif in purine_motifs)

def count_polya_signals(sequence):
    """Count polyadenylation signals"""
    polya_signals = ['AAUAAA', 'AUUAAA', 'AAUAGA', 'AAUACA']
    count = 0
    for signal in polya_signals:
        count += sequence.count(signal)
    return count

def count_au_rich_elements(sequence):
    """Count AU-rich elements (AREs)"""
    are_patterns = ['AUUUA', 'UAUUUAU', 'UUAUUUAUU']
    count = 0
    for pattern in are_patterns:
        count += sequence.count(pattern)
    return count

def count_potential_mirna_sites(sequence):
    """Count potential miRNA binding sites (simplified)"""
    # Look for seed sequence complementarity (simplified)
    seed_complements = ['UCAGAGG', 'UCUAGAG', 'UCCUAGG']
    count = 0
    for seed in seed_complements:
        count += sequence.count(seed)
    return count

def calculate_stem_propensity(sequence):
    """Calculate propensity for stem formation (simplified)"""
    gc_pairs = sequence.count('GC') + sequence.count('CG')
    au_pairs = sequence.count('AU') + sequence.count('UA')
    total_pairs = gc_pairs + au_pairs
    return total_pairs / max(1, len(sequence) // 2)

def find_ribosome_binding_sites(sequence):
    """Find potential ribosome binding sites"""
    # Simplified Shine-Dalgarno like sequences
    rbs_motifs = ['AGGAGG', 'AGGA', 'GGAG']
    sites = []
    for motif in rbs_motifs:
        pos = sequence.find(motif)
        while pos != -1:
            sites.append({'motif': motif, 'position': pos})
            pos = sequence.find(motif, pos + 1)
    return sites

def find_ire_motifs(sequence):
    """Find Iron Responsive Element motifs"""
    # Simplified IRE motif
    ire_like = ['CAGUGN']  # N = any nucleotide
    sites = []
    for i in range(len(sequence) - 6):
        if sequence[i:i+5] == 'CAGUG':
            sites.append({'motif': 'IRE-like', 'position': i})
    return sites

def find_ires_like_sequences(sequence):
    """Find IRES-like sequences"""
    # Very simplified IRES detection
    ires_motifs = ['GNRA', 'UNCG']  # Simplified tetraloop motifs
    sites = []
    # This is a placeholder - real IRES detection is much more complex
    if 'GGAA' in sequence or 'UUCG' in sequence:
        sites.append({'motif': 'IRES-like', 'evidence': 'tetraloop'})
    return sites

def find_au_rich_elements(sequence):
    """Find AU-rich elements with positions"""
    are_motifs = ['AUUUA', 'UAUUUAU', 'UUAUUUAUU']
    sites = []
    for motif in are_motifs:
        pos = sequence.find(motif)
        while pos != -1:
            sites.append({'motif': motif, 'position': pos, 'type': 'ARE'})
            pos = sequence.find(motif, pos + 1)
    return sites

def find_polya_signals(sequence):
    """Find polyadenylation signals with positions"""
    polya_motifs = ['AAUAAA', 'AUUAAA', 'AAUAGA']
    sites = []
    for motif in polya_motifs:
        pos = sequence.find(motif)
        while pos != -1:
            sites.append({'motif': motif, 'position': pos, 'type': 'PolyA'})
            pos = sequence.find(motif, pos + 1)
    return sites

def find_cpe_motifs(sequence):
    """Find Cytoplasmic Polyadenylation Elements"""
    cpe_motifs = ['UUUUUAU', 'UUUUAU']
    sites = []
    for motif in cpe_motifs:
        pos = sequence.find(motif)
        while pos != -1:
            sites.append({'motif': motif, 'position': pos, 'type': 'CPE'})
            pos = sequence.find(motif, pos + 1)
    return sites

def predict_hairpin_loops(sequence):
    """Predict potential hairpin loops (simplified)"""
    # Very simplified hairpin detection
    hairpins = []
    min_stem = 4
    max_loop = 10

    for i in range(len(sequence) - min_stem * 2 - 3):
        for loop_size in range(3, max_loop):
            if i + min_stem + loop_size + min_stem > len(sequence):
                break

            stem1 = sequence[i:i+min_stem]
            stem2 = sequence[i+min_stem+loop_size:i+min_stem+loop_size+min_stem]
            stem2_rev = stem2[::-1].replace('A', 't').replace('U', 'a').replace('G', 'c').replace('C', 'g').upper().replace('T', 'U')

            if stem1 == stem2_rev:
                hairpins.append({
                    'position': i,
                    'stem_length': min_stem,
                    'loop_size': loop_size,
                    'stability': 'medium'
                })

    return hairpins[:5]  # Return top 5

def predict_g_quadruplexes(sequence):
    """Predict G-quadruplex forming sequences"""
    # Look for G-rich regions that could form quadruplexes
    g_quad_sites = []
    pattern = 'GGGG'

    pos = sequence.find(pattern)
    while pos != -1:
        g_quad_sites.append({'position': pos, 'motif': pattern, 'type': 'G4'})
        pos = sequence.find(pattern, pos + 1)

    return g_quad_sites

def save_utr_analysis(results, output_dir, region_type):
    """Save comprehensive UTR analysis results"""

    # Save embeddings
    embeddings_file = os.path.join(output_dir, f"{region_type.lower()}_embeddings.npz")
    embeddings_data = {}
    for i, result in enumerate(results):
        embeddings_data[f"seq_{i}_mean"] = result['embeddings']['mean']
        embeddings_data[f"seq_{i}_max"] = result['embeddings']['max']
    np.savez(embeddings_file, **embeddings_data)

    # Save detailed analysis
    analysis_file = os.path.join(output_dir, f"{region_type.lower()}_analysis.txt")
    with open(analysis_file, 'w') as f:
        f.write(f"{region_type} Function Analysis\\n")
        f.write("=" * (len(region_type) + 17) + "\\n\\n")
        f.write(f"Number of sequences analyzed: {len(results)}\\n")
        f.write(f"Region type: {region_type}\\n\\n")

        for result in results:
            f.write(f"Sequence: {result['name']}\\n")
            f.write(f"Length: {result['length']} nucleotides\\n")
            f.write(f"GC content: {result['utr_features']['gc_content']:.3f}\\n")

            # Features specific to UTR type
            if region_type == "5UTR":
                f.write(f"Kozak sequence: {result['utr_features']['has_kozak']}\\n")
                f.write(f"Upstream ORFs: {result['utr_features']['uorf_count']}\\n")
                f.write(f"Riboswitch-like: {result['utr_features']['riboswitch_like']}\\n")
            elif region_type == "3UTR":
                f.write(f"PolyA signals: {result['utr_features']['polya_signals']}\\n")
                f.write(f"AU-rich elements: {result['utr_features']['au_rich_elements']}\\n")
                f.write(f"Potential miRNA sites: {result['utr_features']['mirna_sites']}\\n")

            f.write(f"Stem propensity: {result['utr_features']['stem_propensity']:.3f}\\n")

            # Regulatory elements
            reg_elements = result['regulatory_elements']
            total_elements = sum(len(v) if isinstance(v, list) else (1 if v else 0) for v in reg_elements.values())
            f.write(f"Predicted regulatory elements: {total_elements}\\n")

            f.write("-" * 60 + "\\n")

    # Save feature matrix for machine learning
    features_file = os.path.join(output_dir, f"{region_type.lower()}_features.csv")
    with open(features_file, 'w') as f:
        # Write header
        if results:
            sample_features = results[0]['utr_features']
            header = ['sequence_name', 'length'] + list(sample_features.keys()) + ['embedding_norm']
            f.write(','.join(header) + '\\n')

            # Write data
            for result in results:
                row = [result['name'], str(result['length'])]
                for key in sample_features.keys():
                    value = result['utr_features'][key]
                    if isinstance(value, bool):
                        row.append('1' if value else '0')
                    else:
                        row.append(f"{value:.6f}" if isinstance(value, float) else str(value))

                embedding_norm = np.linalg.norm(result['embeddings']['mean'])
                row.append(f"{embedding_norm:.6f}")
                f.write(','.join(row) + '\\n')

    # Save regulatory elements summary
    elements_file = os.path.join(output_dir, f"{region_type.lower()}_regulatory_elements.txt")
    with open(elements_file, 'w') as f:
        f.write(f"Regulatory Elements Summary - {region_type}\\n")
        f.write("=" * (len(region_type) + 35) + "\\n\\n")

        for result in results:
            f.write(f"Sequence: {result['name']}\\n")
            reg_elements = result['regulatory_elements']

            for element_type, elements in reg_elements.items():
                if isinstance(elements, list) and len(elements) > 0:
                    f.write(f"  {element_type}: {len(elements)} found\\n")
                    for elem in elements[:3]:  # Show first 3
                        if isinstance(elem, dict):
                            f.write(f"    - {elem}\\n")
                elif elements:
                    f.write(f"  {element_type}: Present\\n")

            f.write("\\n")

    print(f"Analysis files saved:")
    print(f"  - Embeddings: {embeddings_file}")
    print(f"  - Detailed analysis: {analysis_file}")
    print(f"  - Feature matrix: {features_file}")
    print(f"  - Regulatory elements: {elements_file}")

def main():
    parser = argparse.ArgumentParser(description="UTR function prediction using RNA-FM")
    parser.add_argument("--input", "-i", required=True,
                       help="Input FASTA file with UTR sequences")
    parser.add_argument("--output", "-o", default="results/utr_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--type", "-t", choices=["5UTR", "3UTR"], default="5UTR",
                       help="Type of UTR region (default: 5UTR)")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1

    print(f"UTR Function Analysis - {args.type}")
    print("=" * (len(args.type) + 25))
    print(f"This analysis predicts regulatory elements and functional features in {args.type} sequences")
    print()

    # Check environment
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Error: PyTorch not found. Please activate the env_py38 environment.")
        return 1

    # Run UTR analysis
    success = analyze_utr_function(args.input, args.output, args.type)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())