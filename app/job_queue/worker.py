"""Async job queue with single serial worker for VRAM safety.

Processes face swap requests one at a time to prevent GPU OOM from
concurrent ONNX inference.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from app.config import JOB_RESULT_TTL_SECONDS, QUEUE_TIMEOUT_SECONDS
from app.job_queue.models import Job, JobStatus

logger = logging.getLogger(__name__)

ProcessFunc = Callable[[str, dict], Coroutine[Any, Any, Any]]


class AsyncJobQueue:
    """Serial job queue backed by asyncio.Queue with a single worker coroutine.

    Each job is processed sequentially. Results are stored in memory and
    cleaned up after a configurable TTL.
    """

    def __init__(
        self,
        process_fn: ProcessFunc,
        timeout: int = QUEUE_TIMEOUT_SECONDS,
        ttl: int = JOB_RESULT_TTL_SECONDS,
    ) -> None:
        """Initialize the queue.

        Args:
            process_fn: Async callable (job_id, input_data) -> result.
            timeout: Maximum seconds a job may run before being considered failed.
            ttl: Seconds to retain completed job results before cleanup.
        """
        self._queue: asyncio.Queue = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._process_fn = process_fn
        self._timeout = timeout
        self._ttl = ttl
        self._worker_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def queue_depth(self) -> int:
        """Number of jobs waiting in the queue."""
        return self._queue.qsize()

    @property
    def active_jobs(self) -> int:
        """Number of jobs currently processing (always 0 or 1)."""
        return sum(
            1 for j in self._jobs.values()
            if j.status == JobStatus.PROCESSING
        )

    def submit(self, input_data: dict) -> str:
        """Submit a job to the queue and return the job ID immediately.

        Args:
            input_data: Request payload dict.

        Returns:
            Unique job ID string.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            input_params=input_data,
        )
        self._jobs[job_id] = job
        self._queue.put_nowait(job)
        logger.info("Job %s queued (depth: %d)", job_id, self.queue_depth)
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Return a job by ID, or None if not found."""
        return self._jobs.get(job_id)

    async def start(self) -> None:
        """Start the worker and cleanup background tasks."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Queue worker started")

    async def stop(self) -> None:
        """Gracefully stop the worker, draining remaining jobs."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:  # pragma: no cover
                pass
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:  # pragma: no cover
                pass
        logger.info("Queue worker stopped")

    async def _worker_loop(self) -> None:
        """Main worker: process one job at a time from the queue."""
        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            logger.info("Processing job %s", job.job_id)

            try:
                result = await asyncio.wait_for(
                    self._process_fn(job.job_id, job.input_params),
                    timeout=self._timeout,
                )
                job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = datetime.now(timezone.utc)
                logger.info("Job %s completed", job.job_id)
            except asyncio.TimeoutError:
                job.status = JobStatus.FAILED
                job.error = f"Job timed out after {self._timeout}s"
                job.completed_at = datetime.now(timezone.utc)
                logger.error("Job %s timed out", job.job_id)
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.error_traceback = str(exc)
                job.completed_at = datetime.now(timezone.utc)
                logger.exception("Job %s failed", job.job_id)
            finally:
                self._queue.task_done()

    async def _cleanup_loop(self) -> None:
        """Periodically remove old completed/failed jobs."""
        while self._running:
            try:
                await asyncio.sleep(self._ttl)
                now = datetime.now(timezone.utc)
                stale = [
                    jid for jid, job in self._jobs.items()
                    if job.completed_at is not None
                    and (now - job.completed_at).total_seconds() > self._ttl
                ]
                for jid in stale:
                    del self._jobs[jid]
                if stale:
                    logger.debug("Cleaned up %d stale jobs", len(stale))
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Cleanup error")
