"""Tests for queue models."""

from app.queue.models import Job, JobStatus


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_values(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"


class TestJob:
    """Tests for Job dataclass."""

    def test_default_creation(self):
        job = Job(job_id="test-1")
        assert job.job_id == "test-1"
        assert job.status == JobStatus.QUEUED
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.result is None
        assert job.error is None
        assert job.error_traceback is None
        assert job.input_params == {}

    def test_with_input_params(self):
        job = Job(job_id="test-2", input_params={"source_image": "abc"})
        assert job.input_params["source_image"] == "abc"

    def test_to_dict_queued(self):
        job = Job(job_id="test-3")
        d = job.to_dict()
        assert d["status"] == "queued"
        assert d["job_id"] == "test-3"
        assert d["result"] is None
        assert d["error"] is None
        assert d["created_at"] is not None

    def test_to_dict_completed(self):
        job = Job(job_id="test-4")
        job.status = JobStatus.COMPLETED
        job.result = {"image": "base64data"}
        d = job.to_dict()
        assert d["status"] == "completed"
        assert d["result"] == {"image": "base64data"}

    def test_to_dict_failed(self):
        job = Job(job_id="test-5")
        job.status = JobStatus.FAILED
        job.error = "something went wrong"
        d = job.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "something went wrong"
