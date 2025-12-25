"""Job management for long-running RNA-FM tasks."""

import uuid
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger

from .store import JobStore, JobStatus


class JobManager:
    """Manages asynchronous job execution for RNA-FM tasks."""

    def __init__(self, jobs_dir: Path = None):
        """Initialize job manager.

        Args:
            jobs_dir: Directory to store job data (default: ./jobs)
        """
        if jobs_dir is None:
            jobs_dir = Path(__file__).parent.parent.parent / "jobs"

        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(self.jobs_dir)
        self._running_processes: Dict[str, subprocess.Popen] = {}

        logger.info(f"JobManager initialized with jobs_dir: {self.jobs_dir}")

    def submit_job(
        self,
        script_path: str,
        args: Dict[str, Any],
        job_name: str = None
    ) -> Dict[str, Any]:
        """Submit a new job for background execution.

        Args:
            script_path: Path to the script to run
            args: Arguments to pass to the script
            job_name: Optional name for the job

        Returns:
            Dict with job_id and status
        """
        job_id = str(uuid.uuid4())[:8]
        job_dir = self.store.get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = {
            "job_id": job_id,
            "job_name": job_name or f"job_{job_id}",
            "script": script_path,
            "args": args,
            "status": JobStatus.PENDING.value,
            "submitted_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "output_files": []
        }

        self.store.save_metadata(job_id, metadata)
        logger.info(f"Job {job_id} submitted: {script_path}")

        # Start job in background thread
        self._start_job_async(job_id, script_path, args, job_dir)

        return {
            "status": "submitted",
            "job_id": job_id,
            "message": f"Job submitted. Use get_job_status('{job_id}') to check progress."
        }

    def _start_job_async(self, job_id: str, script_path: str, args: Dict, job_dir: Path):
        """Start job execution in background thread.

        Args:
            job_id: Job identifier
            script_path: Path to script to execute
            args: Script arguments
            job_dir: Job output directory
        """
        def run_job():
            try:
                self._execute_job(job_id, script_path, args, job_dir)
            except Exception as e:
                logger.error(f"Job {job_id} execution failed: {e}")
                metadata = self.store.load_metadata(job_id)
                if metadata:
                    metadata["status"] = JobStatus.FAILED.value
                    metadata["error"] = f"Execution error: {str(e)}"
                    metadata["completed_at"] = datetime.now().isoformat()
                    self.store.save_metadata(job_id, metadata)

        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

    def _execute_job(self, job_id: str, script_path: str, args: Dict, job_dir: Path):
        """Execute the actual job.

        Args:
            job_id: Job identifier
            script_path: Path to script to execute
            args: Script arguments
            job_dir: Job output directory
        """
        # Update status to running
        metadata = self.store.load_metadata(job_id)
        if not metadata:
            return

        metadata["status"] = JobStatus.RUNNING.value
        metadata["started_at"] = datetime.now().isoformat()
        self.store.save_metadata(job_id, metadata)

        try:
            # Build command
            cmd = ["python", script_path]

            # Add arguments
            for key, value in args.items():
                if value is not None:
                    flag_name = f"--{key.replace('_', '-')}"
                    if isinstance(value, bool):
                        # Boolean flags: only add the flag if True
                        if value:
                            cmd.append(flag_name)
                    else:
                        # Regular arguments: add flag and value
                        cmd.extend([flag_name, str(value)])

            # Set output directory
            output_dir = job_dir / "output"
            output_dir.mkdir(exist_ok=True)
            cmd.extend(["--output", str(output_dir)])

            # Setup logging
            log_file = job_dir / "job.log"
            logger.info(f"Starting job {job_id}: {' '.join(cmd)}")

            # Execute command
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path(script_path).parent)
                )

                # Track the process
                self._running_processes[job_id] = process

                # Wait for completion
                return_code = process.wait()

            # Update status based on return code
            if return_code == 0:
                metadata["status"] = JobStatus.COMPLETED.value
                metadata["output_files"] = self._collect_output_files(output_dir)
                logger.info(f"Job {job_id} completed successfully")
            else:
                metadata["status"] = JobStatus.FAILED.value
                metadata["error"] = f"Process exited with code {return_code}"
                logger.error(f"Job {job_id} failed with return code {return_code}")

        except Exception as e:
            metadata["status"] = JobStatus.FAILED.value
            metadata["error"] = str(e)
            logger.error(f"Job {job_id} failed: {e}")

        finally:
            metadata["completed_at"] = datetime.now().isoformat()
            self.store.save_metadata(job_id, metadata)
            self._running_processes.pop(job_id, None)

    def _collect_output_files(self, output_dir: Path) -> list:
        """Collect list of output files created by the job.

        Args:
            output_dir: Job output directory

        Returns:
            List of output file paths relative to output_dir
        """
        if not output_dir.exists():
            return []

        output_files = []
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(output_dir)
                output_files.append(str(relative_path))

        return sorted(output_files)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a submitted job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with job status information
        """
        metadata = self.store.load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        result = {
            "job_id": job_id,
            "job_name": metadata.get("job_name"),
            "status": metadata["status"],
            "submitted_at": metadata.get("submitted_at"),
            "started_at": metadata.get("started_at"),
            "completed_at": metadata.get("completed_at"),
        }

        if metadata["status"] == JobStatus.FAILED.value:
            result["error"] = metadata.get("error")

        if metadata["status"] == JobStatus.COMPLETED.value:
            result["output_files"] = metadata.get("output_files", [])

        return result

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get results of a completed job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with job results or error
        """
        metadata = self.store.load_metadata(job_id)
        if not metadata:
            return {"status": "error", "error": f"Job {job_id} not found"}

        if metadata["status"] != JobStatus.COMPLETED.value:
            return {
                "status": "error",
                "error": f"Job not completed. Current status: {metadata['status']}"
            }

        job_dir = self.store.get_job_dir(job_id)
        output_dir = job_dir / "output"

        result = {
            "status": "success",
            "job_id": job_id,
            "job_name": metadata.get("job_name"),
            "completed_at": metadata.get("completed_at"),
            "output_directory": str(output_dir),
            "output_files": metadata.get("output_files", [])
        }

        return result

    def get_job_log(self, job_id: str, tail: int = 50) -> Dict[str, Any]:
        """Get log output from a job.

        Args:
            job_id: Job identifier
            tail: Number of lines from end (0 for all lines)

        Returns:
            Dictionary with log content
        """
        job_dir = self.store.get_job_dir(job_id)
        log_file = job_dir / "job.log"

        if not log_file.exists():
            return {"status": "error", "error": f"Log not found for job {job_id}"}

        try:
            with open(log_file) as f:
                lines = f.readlines()

            if tail > 0:
                log_lines = lines[-tail:]
            else:
                log_lines = lines

            return {
                "status": "success",
                "job_id": job_id,
                "log_lines": [line.rstrip() for line in log_lines],
                "total_lines": len(lines)
            }

        except Exception as e:
            return {"status": "error", "error": f"Failed to read log: {str(e)}"}

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            Success or error message
        """
        # Check if job is running
        if job_id in self._running_processes:
            try:
                process = self._running_processes[job_id]
                process.terminate()

                # Wait a bit for graceful termination
                time.sleep(1)
                if process.poll() is None:
                    process.kill()

                # Update metadata
                metadata = self.store.load_metadata(job_id)
                if metadata:
                    metadata["status"] = JobStatus.CANCELLED.value
                    metadata["completed_at"] = datetime.now().isoformat()
                    self.store.save_metadata(job_id, metadata)

                self._running_processes.pop(job_id, None)
                logger.info(f"Job {job_id} cancelled")
                return {"status": "success", "message": f"Job {job_id} cancelled"}

            except Exception as e:
                logger.error(f"Failed to cancel job {job_id}: {e}")
                return {"status": "error", "error": f"Failed to cancel job: {str(e)}"}

        else:
            metadata = self.store.load_metadata(job_id)
            if not metadata:
                return {"status": "error", "error": f"Job {job_id} not found"}

            current_status = metadata.get("status")
            if current_status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
                return {"status": "error", "error": f"Job {job_id} is already {current_status}"}
            else:
                return {"status": "error", "error": f"Job {job_id} is not currently running"}

    def list_jobs(self, status: Optional[str] = None) -> Dict[str, Any]:
        """List all jobs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            Dictionary with job list
        """
        try:
            jobs_data = self.store.list_jobs(status)

            # Format for display
            jobs = []
            for metadata in jobs_data:
                job_info = {
                    "job_id": metadata["job_id"],
                    "job_name": metadata.get("job_name"),
                    "status": metadata["status"],
                    "submitted_at": metadata.get("submitted_at"),
                    "script": metadata.get("script", "").split("/")[-1]  # Just filename
                }
                jobs.append(job_info)

            return {"status": "success", "jobs": jobs, "total": len(jobs)}

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return {"status": "error", "error": f"Failed to list jobs: {str(e)}"}


# Global job manager instance
job_manager = JobManager()