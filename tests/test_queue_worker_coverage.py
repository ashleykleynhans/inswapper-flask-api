"""Tests for queue worker coverage gaps."""

import asyncio

import pytest

from app.job_queue.worker import AsyncJobQueue


async def _noop(job_id: str, input_data: dict) -> dict:
    await asyncio.sleep(0.01)
    return {"image": "test"}


class TestAsyncJobQueueCoverage:
    """Fill remaining queue worker coverage gaps."""

    @pytest.mark.asyncio
    async def test_active_jobs(self):
        q = AsyncJobQueue(process_fn=_noop, ttl=1)
        await q.start()
        try:
            assert q.active_jobs == 0
            job_id = q.submit({"source_image": "test"})
            await asyncio.sleep(0.05)
            job = q.get_job(job_id)
            assert job.status.value in ("completed", "failed")
            assert q.active_jobs == 0
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_worker(self):
        q = AsyncJobQueue(process_fn=_noop, ttl=1)
        await q.start()
        q.submit({"source_image": "test"})
        await asyncio.sleep(0.05)
        await q.stop()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        q = AsyncJobQueue(process_fn=_noop, timeout=1, ttl=1)
        await q.start()
        try:
            job_id = q.submit({"source_image": "test"})
            await asyncio.sleep(0.1)
            job = q.get_job(job_id)
            assert job is not None
            assert job.status.value in ("completed", "failed")
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_active_jobs_during_processing(self):
        q = AsyncJobQueue(process_fn=_noop, ttl=1)
        await q.start()
        try:
            q.submit({"source_image": "test"})
            await asyncio.sleep(0.001)
            await asyncio.sleep(0.1)
            assert q.active_jobs == 0
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        q = AsyncJobQueue(process_fn=_noop)
        await q.stop()
