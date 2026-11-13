"""FastAPI route controllers for GeneWeaver."""

import datetime
import json
import math
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
import torch

from app.api.schemas import (
    AlignmentHitResponse,
    AlignmentRequest,
    BenchmarkMetrics,
    BenchmarkResponse,
    HitListResponse,
    JobResponse,
)
from app.core.config import settings
from app.db.database import async_session_factory, get_db
from app.db.models import AlignmentHit, BenchmarkRecord, Job
from app.engine.aligner import CrisprAligner

router = APIRouter(prefix=settings.API_V1_PREFIX, tags=["CRISPR Engine"])


async def _execute_alignment_job(
    job_id: str,
    guide_seq: str,
    target_fasta: str,
    pam: str,
    mismatch_threshold: int,
    device_preference: str,
) -> None:
    """Background task executing the tensor alignment engine and persisting results."""
    async with async_session_factory() as db:
        # Fetch existing job record
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return

        try:
            job.status = "RUNNING"
            job.progress = 10.0
            await db.commit()

            # Initialize high-performance tensor aligner
            aligner = CrisprAligner(device_preference=device_preference)
            job.device_used = aligner.device_name

            # Run alignment pipeline
            hits, summary = aligner.align_sequence(
                guide_seq=guide_seq,
                fasta_content=target_fasta,
                pam_pattern=pam,
                mismatch_threshold=mismatch_threshold,
            )

            # Persist hits in batches to optimize DB transaction performance
            for hit in hits:
                hit_orm = AlignmentHit(
                    job_id=job_id,
                    target_name=hit.target_name,
                    start_pos=hit.start_pos,
                    end_pos=hit.end_pos,
                    strand=hit.strand,
                    sequence=hit.sequence,
                    pam_sequence=hit.pam,
                    mismatches=hit.mismatches,
                    mismatch_positions_json=json.dumps(hit.mismatch_positions),
                    mit_score=hit.mit_score,
                    cfd_score=hit.cfd_score,
                    cleavage_tier=hit.cleavage_tier,
                )
                db.add(hit_orm)

            # Update job statistics
            job.status = "COMPLETED"
            job.progress = 100.0
            job.total_kmers_scanned = summary["total_kmers_scanned"]
            job.total_pam_sites = summary["total_pam_sites"]
            job.total_hits = summary["total_hits"]
            job.execution_time_ms = summary["execution_time_ms"]
            job.device_used = summary["device_used"]
            job.summary_json = json.dumps(summary)
            job.completed_at = datetime.datetime.now(datetime.timezone.utc)

            await db.commit()

        except Exception as exc:
            await db.rollback()
            # Reload job for failure update
            res = await db.execute(select(Job).where(Job.id == job_id))
            job_failed = res.scalar_one_or_none()
            if job_failed:
                job_failed.status = "FAILED"
                job_failed.error_message = str(exc)
                job_failed.completed_at = datetime.datetime.now(datetime.timezone.utc)
                await db.commit()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Service health and acceleration capabilities endpoint."""
    cuda_avail = torch.cuda.is_available()
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "torch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_avail else None,
        "cpu_threads": torch.get_num_threads(),
    }


@router.post("/align", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_alignment(
    req: AlignmentRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Submit a CRISPR gRNA sequence for GPU-accelerated genome alignment."""
    # Resolve target sequence
    fasta_content = req.target_sequence
    target_name = "Custom Genomic Sequence"

    if not fasta_content:
        # Load sample reference chr19 snippet
        sample_path = settings.SAMPLE_DATA_DIR / "reference_chr19.fasta"
        if sample_path.exists():
            fasta_content = sample_path.read_text(encoding="utf-8")
            target_name = "Human Chr19 Sample Locus (GRCh38)"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No target sequence provided and default reference not found.",
            )

    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        grna_sequence=req.grna_sequence,
        pam=req.pam,
        target_name=target_name,
        status="PENDING",
        progress=0.0,
        mismatch_threshold=req.mismatch_threshold,
        device_used="Detecting...",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Dispatch alignment background task
    background_tasks.add_task(
        _execute_alignment_job,
        job_id=job_id,
        guide_seq=req.grna_sequence,
        target_fasta=fasta_content,
        pam=req.pam,
        mismatch_threshold=req.mismatch_threshold,
        device_preference=req.device_preference,
    )

    return JobResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        grna_sequence=job.grna_sequence,
        pam=job.pam,
        target_name=job.target_name,
        mismatch_threshold=job.mismatch_threshold,
        total_kmers_scanned=job.total_kmers_scanned,
        total_pam_sites=job.total_pam_sites,
        total_hits=job.total_hits,
        execution_time_ms=job.execution_time_ms,
        device_used=job.device_used,
        created_at=job.created_at.isoformat() if job.created_at else None,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)) -> JobResponse:
    """Retrieve the status and summary metrics of an alignment job."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found.",
        )

    summary_stats = json.loads(job.summary_json) if job.summary_json else None

    return JobResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        grna_sequence=job.grna_sequence,
        pam=job.pam,
        target_name=job.target_name,
        mismatch_threshold=job.mismatch_threshold,
        total_kmers_scanned=job.total_kmers_scanned,
        total_pam_sites=job.total_pam_sites,
        total_hits=job.total_hits,
        execution_time_ms=job.execution_time_ms,
        device_used=job.device_used,
        error_message=job.error_message,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        summary_stats=summary_stats,
    )


@router.get("/jobs/{job_id}/results", response_model=HitListResponse)
async def get_job_results(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    tier: Optional[str] = Query(default=None, description="Filter by risk tier (High, Medium, Low)"),
    max_mismatches: Optional[int] = Query(default=None, ge=0, le=6),
    db: AsyncSession = Depends(get_db),
) -> HitListResponse:
    """Retrieve paginated alignment hits and cleavage predictions."""
    # Ensure job exists
    job_res = await db.execute(select(Job).where(Job.id == job_id))
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found.",
        )

    query = select(AlignmentHit).where(AlignmentHit.job_id == job_id)
    if tier:
        query = query.where(AlignmentHit.cleavage_tier == tier.capitalize())
    if max_mismatches is not None:
        query = query.where(AlignmentHit.mismatches <= max_mismatches)

    # Count total matching hits
    count_query = select(func.count()).select_from(query.subquery())
    total_count = (await db.execute(count_query)).scalar() or 0

    # Paginate and order by lowest mismatch count, highest CFD score
    offset = (page - 1) * page_size
    query = (
        query.order_by(AlignmentHit.mismatches.asc(), AlignmentHit.cfd_score.desc())
        .offset(offset)
        .limit(page_size)
    )

    hits_res = await db.execute(query)
    hits = hits_res.scalars().all()

    total_pages = max(1, math.ceil(total_count / page_size))

    hit_items = [
        AlignmentHitResponse(
            id=h.id,
            job_id=h.job_id,
            target_name=h.target_name,
            start_pos=h.start_pos,
            end_pos=h.end_pos,
            strand=h.strand,
            sequence=h.sequence,
            pam_sequence=h.pam_sequence,
            mismatches=h.mismatches,
            mismatch_positions=json.loads(h.mismatch_positions_json),
            mit_score=h.mit_score,
            cfd_score=h.cfd_score,
            cleavage_tier=h.cleavage_tier,
        )
        for h in hits
    ]

    return HitListResponse(
        job_id=job_id,
        total_hits=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        hits=hit_items,
    )


@router.get("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    num_kmers: int = Query(default=50000, ge=1000, le=500000),
    device_preference: str = Query(default="auto"),
    db: AsyncSession = Depends(get_db),
) -> BenchmarkResponse:
    """Benchmark tensor alignment throughput across batch sizes."""
    aligner = CrisprAligner(device_preference=device_preference)
    
    # Generate synthetic candidate k-mers
    guide = "GAGTCCGAGCAGAAGAAGAA"
    test_batches = [10000, 50000, min(100000, num_kmers)]
    metrics_list: List[BenchmarkMetrics] = []

    for b_size in test_batches:
        candidates = [
            (i, i + 20, "+", guide[:17] + "NNN", "TGG")
            for i in range(b_size)
        ]
        
        t0 = time.perf_counter()
        aligner.align_candidates(
            guide_seq=guide,
            candidates=candidates,
            target_name="synthetic_bench",
            mismatch_threshold=3,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        throughput = round((b_size / (latency_ms / 1000.0)), 1) if latency_ms > 0 else 0.0

        metrics_list.append(
            BenchmarkMetrics(
                device=aligner.device_name,
                batch_size=b_size,
                latency_ms=latency_ms,
                kmers_per_sec=throughput,
            )
        )

        # Record in DB
        record = BenchmarkRecord(
            device=aligner.device_name,
            batch_size=b_size,
            latency_ms=latency_ms,
            kmers_per_sec=throughput,
            cuda_available=aligner.is_cuda,
            torch_version=torch.__version__,
        )
        db.add(record)

    await db.commit()

    return BenchmarkResponse(
        timestamp=time.time(),
        torch_version=torch.__version__,
        cuda_available=aligner.is_cuda,
        benchmark_metrics=metrics_list,
    )
