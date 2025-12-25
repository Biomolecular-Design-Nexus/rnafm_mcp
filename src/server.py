#!/usr/bin/env python3
"""
RNA-FM MCP Server

Provides both synchronous and asynchronous (submit) APIs for RNA analysis tools.

Features:
- RNA sequence embedding extraction
- Secondary structure prediction
- RNA sequence classification and clustering
- Job management for long-running tasks
- Batch processing support
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import os

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("RNA-FM")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)


@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)


@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)


@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)


# ==============================================================================
# RNA Embeddings Tools
# ==============================================================================

@mcp.tool()
def extract_rna_embeddings(
    input_file: str,
    output_file: Optional[str] = None,
    use_mock: bool = False,
    config_file: Optional[str] = None
) -> dict:
    """
    Extract RNA sequence embeddings using RNA-FM (fast operation for small datasets).

    This is suitable for extracting embeddings from small RNA sequence files.
    For large datasets or when using real RNA-FM models, consider using
    submit_rna_embeddings for background processing.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_file: Optional output directory to save embeddings
        use_mock: Use mock embeddings instead of RNA-FM model (faster, for testing)
        config_file: Optional config file path

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - embeddings_shape: Shape of extracted embeddings
        - num_sequences: Number of sequences processed
        - output_dir: Output directory path (if output_file provided)
        - processing_time: Time taken
        - error: Error message (if status is "error")

    Example:
        extract_rna_embeddings("examples/data/example.fasta", "results/embeddings")
    """
    from rna_embeddings import run_rna_embeddings

    try:
        result = run_rna_embeddings(
            input_file=input_file,
            output_file=output_file,
            use_mock=use_mock,
            config_file=config_file
        )

        # Extract key information for MCP response
        embeddings = result.get('embeddings', {})
        metadata = result.get('metadata', {})

        # Calculate shape of embeddings
        if embeddings:
            # Get the first embedding to determine shape
            first_key = next(iter(embeddings.keys()))
            embedding_shape = list(embeddings[first_key].shape) if hasattr(embeddings[first_key], 'shape') else [0, 0]
        else:
            embedding_shape = [0, 0]

        return {
            "status": "success",
            "num_sequences": metadata.get('num_sequences', 0),
            "embeddings_shape": embedding_shape,
            "output_dir": result.get('output_dir'),
            "processing_time": metadata.get('processing_time', 0),
            "metadata": metadata
        }

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"RNA embeddings extraction failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def submit_rna_embeddings(
    input_file: str,
    output_dir: Optional[str] = None,
    use_mock: bool = False,
    config_file: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit RNA embedding extraction for background processing (large datasets).

    This operation extracts embeddings for large datasets that may take time.
    Suitable for hundreds of sequences or when using real RNA-FM models.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_dir: Directory to save outputs
        use_mock: Use mock embeddings (faster, for testing)
        config_file: Optional config file path
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "rna_embeddings.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "use_mock": use_mock,
            "config": config_file
        },
        job_name=job_name or "rna_embeddings"
    )


# ==============================================================================
# Secondary Structure Tools
# ==============================================================================

@mcp.tool()
def predict_secondary_structure(
    input_file: str,
    output_file: Optional[str] = None,
    threshold: float = 0.5,
    use_mock: bool = False,
    formats: Optional[List[str]] = None,
    config_file: Optional[str] = None
) -> dict:
    """
    Predict RNA secondary structure (fast operation for small datasets).

    Predicts base-pairing patterns for RNA sequences using RNA-FM + ResNet
    or mock generation for testing.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_file: Optional output directory to save predictions
        threshold: Base-pair probability threshold (default: 0.5)
        use_mock: Use mock predictions instead of RNA-FM model (faster, for testing)
        formats: List of output formats: ["npy", "txt", "ct"] (default: all)
        config_file: Optional config file path

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - num_sequences: Number of sequences processed
        - total_base_pairs: Total predicted base pairs
        - output_dir: Output directory path (if output_file provided)
        - processing_time: Time taken
        - error: Error message (if status is "error")

    Example:
        predict_secondary_structure("examples/data/example.fasta", "results/structures")
    """
    from secondary_structure import run_secondary_structure

    try:
        result = run_secondary_structure(
            input_file=input_file,
            output_file=output_file,
            threshold=threshold,
            use_mock=use_mock,
            formats=formats or ["npy", "txt", "ct"],
            config_file=config_file
        )

        # Extract key information for MCP response
        metadata = result.get('metadata', {})

        return {
            "status": "success",
            "num_sequences": metadata.get('num_sequences', 0),
            "total_base_pairs": result.get('statistics', {}).get('total_base_pairs', 0),
            "output_dir": result.get('output_dir'),
            "processing_time": metadata.get('processing_time', 0),
            "metadata": metadata
        }

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Secondary structure prediction failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def submit_secondary_structure(
    input_file: str,
    output_dir: Optional[str] = None,
    threshold: float = 0.5,
    use_mock: bool = False,
    formats: Optional[List[str]] = None,
    config_file: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit secondary structure prediction for background processing.

    This operation predicts secondary structures for large datasets or when
    using real RNA-FM models that may take significant time.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_dir: Directory to save outputs
        threshold: Base-pair probability threshold (default: 0.5)
        use_mock: Use mock predictions (faster, for testing)
        formats: List of output formats: ["npy", "txt", "ct"] (default: all)
        config_file: Optional config file path
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the prediction job
    """
    script_path = str(SCRIPTS_DIR / "secondary_structure.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "threshold": threshold,
            "use_mock": use_mock,
            "formats": ",".join(formats) if formats else "npy,txt,ct",
            "config": config_file
        },
        job_name=job_name or "secondary_structure"
    )


# ==============================================================================
# RNA Classification Tools
# ==============================================================================

@mcp.tool()
def classify_rna_sequences(
    input_file: str,
    output_file: Optional[str] = None,
    num_clusters: int = 3,
    use_mock: bool = False,
    pca_components: int = 50,
    config_file: Optional[str] = None
) -> dict:
    """
    Classify and cluster RNA sequences by functional families (fast operation).

    Uses RNA-FM embeddings and K-means clustering to group RNA sequences
    by functional similarity. Suitable for moderate-sized datasets.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_file: Optional output directory to save classifications
        num_clusters: Number of clusters for K-means (default: 3)
        use_mock: Use mock embeddings instead of RNA-FM (faster, for testing)
        pca_components: Number of PCA components for dimensionality reduction
        config_file: Optional config file path

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - num_sequences: Number of sequences processed
        - num_clusters: Number of clusters created
        - cluster_sizes: List of cluster sizes
        - silhouette_score: Clustering quality score
        - output_dir: Output directory path (if output_file provided)
        - processing_time: Time taken
        - error: Error message (if status is "error")

    Example:
        classify_rna_sequences("examples/data/RF00005.fasta", "results/classification", num_clusters=2)
    """
    from rna_classification import run_rna_classification

    try:
        result = run_rna_classification(
            input_file=input_file,
            output_file=output_file,
            num_clusters=num_clusters,
            use_mock=use_mock,
            pca_components=pca_components,
            config_file=config_file
        )

        # Extract key information for MCP response
        metadata = result.get('metadata', {})
        clustering = result.get('clustering', {})

        return {
            "status": "success",
            "num_sequences": metadata.get('num_sequences', 0),
            "num_clusters": clustering.get('n_clusters', num_clusters),
            "cluster_sizes": clustering.get('cluster_sizes', []),
            "silhouette_score": clustering.get('silhouette_score', 0.0),
            "output_dir": result.get('output_dir'),
            "processing_time": metadata.get('processing_time', 0),
            "metadata": metadata
        }

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"RNA classification failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def submit_rna_classification(
    input_file: str,
    output_dir: Optional[str] = None,
    num_clusters: int = 3,
    use_mock: bool = False,
    pca_components: int = 50,
    config_file: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit RNA classification for background processing (large datasets).

    This operation classifies large RNA datasets that may take significant time,
    especially when using real RNA-FM models for embedding extraction.

    Args:
        input_file: Path to FASTA file with RNA sequences
        output_dir: Directory to save outputs
        num_clusters: Number of clusters for K-means (default: 3)
        use_mock: Use mock embeddings (faster, for testing)
        pca_components: Number of PCA components for dimensionality reduction
        config_file: Optional config file path
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the classification job
    """
    script_path = str(SCRIPTS_DIR / "rna_classification.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "clusters": num_clusters,
            "use_mock": use_mock,
            "pca_components": pca_components,
            "config": config_file
        },
        job_name=job_name or "rna_classification"
    )


# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_batch_rna_analysis(
    input_files: List[str],
    analysis_type: str = "all",
    output_dir: Optional[str] = None,
    use_mock: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch RNA analysis (embeddings + structure + classification) for multiple files.

    Processes multiple RNA sequence files with comprehensive analysis.
    Runs embeddings extraction, structure prediction, and classification for each file.

    Args:
        input_files: List of FASTA files to analyze
        analysis_type: Type of analysis - "embeddings", "structure", "classification", or "all"
        output_dir: Directory to save all outputs
        use_mock: Use mock models for faster processing (recommended for testing)
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch analysis
    """
    if not input_files:
        return {"status": "error", "error": "No input files provided"}

    if analysis_type not in ["embeddings", "structure", "classification", "all"]:
        return {"status": "error", "error": "Invalid analysis_type. Must be 'embeddings', 'structure', 'classification', or 'all'"}

    # Create a batch script that processes all files
    script_content = f"""
#!/usr/bin/env python3
import sys
import os
import subprocess
from pathlib import Path
import json

def run_script(script_name, input_file, output_dir, use_mock=True):
    \"\"\"Run a script as subprocess and return success status.\"\"\"
    try:
        scripts_dir = Path(__file__).parent.parent / 'scripts'
        script_path = scripts_dir / script_name

        cmd = ["python", str(script_path), "--input", input_file, "--output", str(output_dir)]
        if use_mock:
            cmd.append("--use-mock")

        print(f"Running: {{' '.join(cmd)}}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {{"status": "success", "output": result.stdout}}

    except subprocess.CalledProcessError as e:
        return {{"status": "error", "error": str(e), "stderr": e.stderr}}
    except Exception as e:
        return {{"status": "error", "error": str(e)}}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--analysis-type', default='all')
    parser.add_argument('--use-mock', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = {input_files}
    results = []

    for i, input_file in enumerate(input_files):
        file_name = Path(input_file).stem
        file_output = output_dir / file_name
        file_output.mkdir(parents=True, exist_ok=True)

        print(f"Processing {{i+1}}/{{len(input_files)}}: {{input_file}}")

        file_results = {{"input_file": input_file}}

        if args.analysis_type in ['embeddings', 'all']:
            result = run_script('rna_embeddings.py', input_file, str(file_output / 'embeddings'), args.use_mock)
            file_results['embeddings'] = result

        if args.analysis_type in ['structure', 'all']:
            result = run_script('secondary_structure.py', input_file, str(file_output / 'structure'), args.use_mock)
            file_results['structure'] = result

        if args.analysis_type in ['classification', 'all']:
            result = run_script('rna_classification.py', input_file, str(file_output / 'classification'), args.use_mock)
            file_results['classification'] = result

        results.append(file_results)

    # Save summary
    with open(output_dir / 'batch_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Batch analysis completed. Results saved to {{output_dir}}")

if __name__ == "__main__":
    main()
"""

    # Save the batch script
    batch_script_path = MCP_ROOT / "temp_batch_script.py"
    with open(batch_script_path, 'w') as f:
        f.write(script_content)

    try:
        return job_manager.submit_job(
            script_path=str(batch_script_path),
            args={
                "analysis_type": analysis_type,
                "use_mock": use_mock
            },
            job_name=job_name or f"batch_analysis_{len(input_files)}_files"
        )
    except Exception as e:
        # Clean up the temp script on error
        if batch_script_path.exists():
            batch_script_path.unlink()
        return {"status": "error", "error": f"Failed to submit batch job: {str(e)}"}


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()