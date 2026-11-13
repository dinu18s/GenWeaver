"""SQLAlchemy ORM models for GeneWeaver."""

import datetime
from typing import Any, Dict, List
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from app.db.database import Base


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class Job(Base):
    """Alignment execution job model."""
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, index=True)
    grna_sequence = Column(String(30), nullable=False)
    pam = Column(String(10), nullable=False, default="NGG")
    target_name = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING", index=True)
    progress = Column(Float, nullable=False, default=0.0)
    mismatch_threshold = Column(Integer, nullable=False, default=3)
    
    total_kmers_scanned = Column(Integer, nullable=False, default=0)
    total_pam_sites = Column(Integer, nullable=False, default=0)
    total_hits = Column(Integer, nullable=False, default=0)
    execution_time_ms = Column(Float, nullable=False, default=0.0)
    device_used = Column(String(100), nullable=False, default="Unknown")
    error_message = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=_utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    hits = relationship(
        "AlignmentHit",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="AlignmentHit.mismatches.asc(), AlignmentHit.cfd_score.desc()",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "grna_sequence": self.grna_sequence,
            "pam": self.pam,
            "target_name": self.target_name,
            "status": self.status,
            "progress": self.progress,
            "mismatch_threshold": self.mismatch_threshold,
            "total_kmers_scanned": self.total_kmers_scanned,
            "total_pam_sites": self.total_pam_sites,
            "total_hits": self.total_hits,
            "execution_time_ms": self.execution_time_ms,
            "device_used": self.device_used,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AlignmentHit(Base):
    """Off-target alignment hit model."""
    __tablename__ = "alignment_hits"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    target_name = Column(String(255), nullable=False)
    start_pos = Column(Integer, nullable=False)
    end_pos = Column(Integer, nullable=False)
    strand = Column(String(1), nullable=False, default="+")
    sequence = Column(String(30), nullable=False)
    pam_sequence = Column(String(10), nullable=False)
    mismatches = Column(Integer, nullable=False, index=True)
    mismatch_positions_json = Column(Text, nullable=False, default="[]")
    
    mit_score = Column(Float, nullable=False, default=0.0)
    cfd_score = Column(Float, nullable=False, default=0.0)
    cleavage_tier = Column(String(20), nullable=False, default="Low", index=True)

    # Relationships
    job = relationship("Job", back_populates="hits")

    def to_dict(self) -> Dict[str, Any]:
        import json
        return {
            "id": self.id,
            "job_id": self.job_id,
            "target_name": self.target_name,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "strand": self.strand,
            "sequence": self.sequence,
            "pam_sequence": self.pam_sequence,
            "mismatches": self.mismatches,
            "mismatch_positions": json.loads(self.mismatch_positions_json),
            "mit_score": self.mit_score,
            "cfd_score": self.cfd_score,
            "cleavage_tier": self.cleavage_tier,
        }


class BenchmarkRecord(Base):
    """Execution benchmark record model."""
    __tablename__ = "benchmarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    device = Column(String(100), nullable=False)
    batch_size = Column(Integer, nullable=False)
    latency_ms = Column(Float, nullable=False)
    kmers_per_sec = Column(Float, nullable=False)
    cuda_available = Column(Boolean, nullable=False)
    torch_version = Column(String(50), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "device": self.device,
            "batch_size": self.batch_size,
            "latency_ms": self.latency_ms,
            "kmers_per_sec": self.kmers_per_sec,
            "cuda_available": self.cuda_available,
            "torch_version": self.torch_version,
        }
