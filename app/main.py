"""GeneWeaver FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import settings
from app.db.database import engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager handling database startup and shutdown."""
    # Ensure static directory exists
    settings.STATIC_DIR.mkdir(parents=True, exist_ok=True)
    settings.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database tables
    await init_db()

    yield

    # Cleanly dispose database connection pool
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for microservice architecture and client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount Static Assets
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    """Serve the GeneWeaver interactive visualization dashboard."""
    index_file = settings.STATIC_DIR / "index.html"
    return FileResponse(str(index_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
