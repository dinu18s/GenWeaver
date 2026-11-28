"""Application configuration settings for GeneWeaver."""

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Central configuration class for GeneWeaver."""
    
    PROJECT_NAME: str = Field(default="GeneWeaver")
    PROJECT_DESCRIPTION: str = Field(default="GPU-Accelerated CRISPR Alignment & Off-Target Scoring Engine")
    VERSION: str = Field(default="1.0.0")
    API_V1_PREFIX: str = Field(default="/api/v1")
    
    # Database
    DATABASE_URL: str = Field(default=os.getenv("GENEWEAVER_DATABASE_URL", "sqlite+aiosqlite:///./geneweaver.db"))
    
    # Engine Settings
    DEFAULT_PAM: str = Field(default="NGG")
    DEFAULT_MISMATCH_THRESHOLD: int = Field(default=3)
    MAX_MISMATCH_THRESHOLD: int = Field(default=6)
    DEFAULT_BATCH_SIZE: int = Field(default=65536)
    DEVICE_PREFERENCE: Literal["auto", "cuda", "cpu"] = Field(
        default=os.getenv("GENEWEAVER_DEVICE_PREFERENCE", "auto") # type: ignore
    )
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    STATIC_DIR: Path = BASE_DIR / "app" / "static"
    SAMPLE_DATA_DIR: Path = BASE_DIR / "sample_data"
    
    # Thresholds for Cleavage Likelihood Tiers
    HIGH_RISK_CFD_THRESHOLD: float = 0.50
    MEDIUM_RISK_CFD_THRESHOLD: float = 0.10
    HIGH_RISK_MIT_THRESHOLD: float = 50.0
    MEDIUM_RISK_MIT_THRESHOLD: float = 20.0


settings = Settings()
