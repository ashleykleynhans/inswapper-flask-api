"""Tests for the async job queue worker."""

import asyncio

import pytest

from app.job_queue.models import JobStatus
from app.job_queue.worker import AsyncJobQueue


async def _noop_process(job_id: str, input_data: dict) -> dict:
    await asyncio.sleep(0.01)
    return {"image": "base64testdata"}


async def _slow_process(job_id: str, input_data: dict) -> dict:
    await asyncio.sleep(0.05)
    return {"image": "base64testdata"}


class TestAsyncJobQueue:
    """Tests for AsyncJobQueue."""

    @pytest.mark.asyncio
    async def test_submit_returns_job_id(self):
        q = AsyncJobQueue(process_fn=_noop_process, ttl=1)
        await q.start()
        try:
            job_id = q.submit({"source_image": "test"})
            assert isinstance(job_id, str)
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_initial_status_is_queued(self):
        q = AsyncJobQueue(process_fn=_noop_process, ttl=1)
        await q.start()
        try:
            job_id = q.submit({"source_image": "test"})
            job = q.get_job(job_id)
            assert job is not None
            assert job.status == JobStatus.QUEUED
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_job_completes(self):
        q = AsyncJobQueue(process_fn=_noop_process, ttl=1)
        await q.start()
        try:
            job_id = q.submit({"source_image": "test"})
            await asyncio.sleep(0.1)
            job = q.get_job(job_id)
            assert job is not None
            assert job.status == JobStatus.COMPLETED
            assert job.result == {"image": "base64testdata"}
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_queue_depth_tracks_pending(self):
        q = AsyncJobQueue(process_fn=_slow_process, ttl=1)
        await q.start()
        try:
            assert q.queue_depth == 0
            q.submit({"source_image": "test1"})
            q.submit({"source_image": "test2"})
            assert q.queue_depth <= 2
            await asyncio.sleep(0.15)
            assert q.queue_depth == 0
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_serial_processing(self):
        q = AsyncJobQueue(process_fn=_slow_process, ttl=1)
        await q.start()
        try:
            job1 = q.submit({"source_image": "test1"})
            job2 = q.submit({"source_image": "test2"})
            await asyncio.sleep(0.3)

            j1 = q.get_job(job1)
            j2 = q.get_job(job2)
            assert j1.status == JobStatus.COMPLETED
            assert j2.status == JobStatus.COMPLETED
            assert j1.started_at is not None
            assert j2.started_at is not None
            assert j1.started_at <= j2.started_at
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_get_job_not_found(self):
        q = AsyncJobQueue(process_fn=_noop_process, ttl=1)
        await q.start()
        try:
            assert q.get_job("nonexistent-id") is None
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        async def failing_process(job_id, input_data):
            raise ValueError("Test error")

        q = AsyncJobQueue(process_fn=failing_process, ttl=1)
        await q.start()
        try:
            job_id = q.submit({"source_image": "test"})
            await asyncio.sleep(0.1)
            job = q.get_job(job_id)
            assert job.status == JobStatus.FAILED
            assert "Test error" in job.error
        finally:
            await q.stop()
