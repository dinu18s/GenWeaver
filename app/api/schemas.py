"""Pydantic v2 schemas for GeneWeaver API validation."""

import datetime
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class AlignmentRequest(BaseModel):
    """CRISPR alignment submission request."""
    
    grna_sequence: str = Field(
        ...,
        description="20-nt guide RNA query sequence (5' to 3')",
        examples=["GAGTCCGAGCAGAAGAAGAA"],
    )
    pam: str = Field(
        default="NGG",
        description="Protospacer Adjacent Motif (IUPAC notation)",
        examples=["NGG", "NAG"],
    )
    target_sequence: Optional[str] = Field(
        default=None,
        description="Genomic reference DNA sequence or multi-line FASTA string",
    )
    reference_accession: Optional[str] = Field(
        default=None,
        description="Sample reference accession (e.g. 'chr19_sample')",
    )
    mismatch_threshold: int = Field(
        default=3,
        ge=0,
        le=6,
        description="Maximum allowed Hamming distance / mismatches (0-6)",
    )
    device_preference: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="Hardware acceleration target",
    )

    @field_validator("grna_sequence")
    @classmethod
    def validate_grna(cls, v: str) -> str:
        clean = re.sub(r"\s+", "", v).upper().replace("U", "T")
        if len(clean) != 20:
            raise ValueError(f"gRNA sequence must be exactly 20 nucleotides (received {len(clean)})")
        if not re.match(r"^[ACGT]+$", clean):
            raise ValueError("gRNA sequence contains invalid characters (only A, C, G, T, U allowed)")
        return clean

    @field_validator("pam")
    @classmethod
    def validate_pam(cls, v: str) -> str:
        clean = re.sub(r"\s+", "", v).upper().replace("U", "T")
        if not (2 <= len(clean) <= 8):
            raise ValueError("PAM length must be between 2 and 8 nucleotides")
        if not re.match(r"^[ACGTURYMKWSBDHVN]+$", clean):
            raise ValueError("PAM must follow valid IUPAC nucleotide nomenclature")
        return clean


class AlignmentHitResponse(BaseModel):
    """Detailed hit representation in API response."""
    
    id: int
    job_id: str
    target_name: str
    start_pos: int
    end_pos: int
    strand: str
    sequence: str
    pam_sequence: str
    mismatches: int
    mismatch_positions: List[int]
    mit_score: float
    cfd_score: float
    cleavage_tier: str


class HitListResponse(BaseModel):
    """Paginated alignment results."""
    
    job_id: str
    total_hits: int
    page: int
    page_size: int
    total_pages: int
    hits: List[AlignmentHitResponse]


class JobResponse(BaseModel):
    """Status and summary of an alignment job."""
    
    id: str
    status: str
    progress: float
    grna_sequence: str
    pam: str
    target_name: Optional[str] = None
    mismatch_threshold: int
    total_kmers_scanned: int
    total_pam_sites: int
    total_hits: int
    execution_time_ms: float
    device_used: str
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    summary_stats: Optional[Dict[str, Any]] = None


class BenchmarkRunRequest(BaseModel):
    """Request payload to execute a synthetic benchmark."""
    
    num_kmers: int = Field(default=50000, ge=1000, le=500000)
    device_preference: Literal["auto", "cuda", "cpu"] = "auto"


class BenchmarkMetrics(BaseModel):
    """Throughput and latency metrics for benchmark."""
    
    device: str
    batch_size: int
    latency_ms: float
    kmers_per_sec: float


class BenchmarkResponse(BaseModel):
    """System benchmark results comparing acceleration architectures."""
    
    timestamp: float
    torch_version: str
    cuda_available: bool
    benchmark_metrics: List[BenchmarkMetrics]
