"""Integration tests for GeneWeaver FastAPI REST endpoints."""

import asyncio
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_endpoint(async_client: AsyncClient):
    """Test health check and capability probe."""
    res = await async_client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "torch_version" in data
    assert "cuda_available" in data
    assert "cpu_threads" in data


@pytest.mark.asyncio
async def test_serve_dashboard(async_client: AsyncClient):
    """Test serving the main frontend dashboard."""
    res = await async_client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "GeneWeaver" in res.text


@pytest.mark.asyncio
async def test_alignment_validation_errors(async_client: AsyncClient):
    """Test Pydantic schema validation for guide RNA and PAM formats."""
    # Test short gRNA
    res_short = await async_client.post(
        "/api/v1/align",
        json={"grna_sequence": "GAGTCC", "pam": "NGG"},
    )
    assert res_short.status_code == 422

    # Test invalid nucleotides
    res_invalid_base = await async_client.post(
        "/api/v1/align",
        json={"grna_sequence": "GAGTCCGAGCAGAAGAAG12", "pam": "NGG"},
    )
    assert res_invalid_base.status_code == 422

    # Test mismatch threshold beyond bounds
    res_mm = await async_client.post(
        "/api/v1/align",
        json={"grna_sequence": "GAGTCCGAGCAGAAGAAGAA", "mismatch_threshold": 10},
    )
    assert res_mm.status_code == 422


@pytest.mark.asyncio
async def test_full_alignment_job_lifecycle(async_client: AsyncClient):
    """Submit alignment job, poll status to completion, and inspect hit table."""
    guide = "GAGTCCGAGCAGAAGAAGAA"
    custom_target = f">locus_1\nNNNN{guide}TGGNNNN{guide[:18]}AATGG"

    payload = {
        "grna_sequence": guide,
        "pam": "NGG",
        "target_sequence": custom_target,
        "mismatch_threshold": 3,
        "device_preference": "cpu",
    }

    # 1. Submit job
    submit_res = await async_client.post("/api/v1/align", json=payload)
    assert submit_res.status_code == 202
    job_data = submit_res.json()
    job_id = job_data["id"]
    assert job_id is not None

    # 2. Poll job status until COMPLETED (up to 5 seconds)
    completed = False
    for _ in range(25):
        poll_res = await async_client.get(f"/api/v1/jobs/{job_id}")
        assert poll_res.status_code == 200
        status_data = poll_res.json()
        if status_data["status"] == "COMPLETED":
            completed = True
            break
        await asyncio.sleep(0.2)

    assert completed is True
    assert status_data["total_hits"] >= 1
    assert status_data["execution_time_ms"] > 0

    # 3. Retrieve results
    results_res = await async_client.get(f"/api/v1/jobs/{job_id}/results")
    assert results_res.status_code == 200
    hits_data = results_res.json()
    assert hits_data["total_hits"] >= 1
    assert len(hits_data["hits"]) >= 1

    first_hit = hits_data["hits"][0]
    assert first_hit["sequence"] == guide
    assert first_hit["mismatches"] == 0
    assert first_hit["cleavage_tier"] == "High"
    assert first_hit["mit_score"] == 100.0


@pytest.mark.asyncio
async def test_nonexistent_job_returns_404(async_client: AsyncClient):
    """Ensure non-existent job queries yield 404."""
    res = await async_client.get("/api/v1/jobs/nonexistent-uuid-12345")
    assert res.status_code == 404

    res_results = await async_client.get("/api/v1/jobs/nonexistent-uuid-12345/results")
    assert res_results.status_code == 404


@pytest.mark.asyncio
async def test_benchmark_endpoint(async_client: AsyncClient):
    """Test synthetic benchmark endpoint execution."""
    res = await async_client.get("/api/v1/benchmark?num_kmers=10000")
    assert res.status_code == 200
    bench_data = res.json()
    assert "benchmark_metrics" in bench_data
    assert len(bench_data["benchmark_metrics"]) >= 1
    assert bench_data["benchmark_metrics"][0]["kmers_per_sec"] > 0
