"""Job state persistence and status management."""

from enum import Enum
from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStore:
    """Handles job metadata persistence."""

    def __init__(self, jobs_dir: Path):
        """Initialize job store.

        Args:
            jobs_dir: Directory to store job metadata
        """
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, job_id: str, metadata: Dict[str, Any]) -> None:
        """Save job metadata to disk.

        Args:
            job_id: Job identifier
            metadata: Job metadata dictionary
        """
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        meta_file = job_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def load_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job metadata from disk.

        Args:
            job_id: Job identifier

        Returns:
            Job metadata dictionary or None if not found
        """
        meta_file = self.jobs_dir / job_id / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def list_jobs(self, status: Optional[str] = None) -> list:
        """List all jobs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of job metadata dictionaries
        """
        jobs = []
        for job_dir in self.jobs_dir.iterdir():
            if job_dir.is_dir():
                metadata = self.load_metadata(job_dir.name)
                if metadata:
                    if status is None or metadata.get("status") == status:
                        jobs.append(metadata)

        # Sort by submission time (newest first)
        jobs.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)
        return jobs

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists.

        Args:
            job_id: Job identifier

        Returns:
            True if job exists, False otherwise
        """
        return (self.jobs_dir / job_id).exists()

    def get_job_dir(self, job_id: str) -> Path:
        """Get job directory path.

        Args:
            job_id: Job identifier

        Returns:
            Path to job directory
        """
        return self.jobs_dir / job_id