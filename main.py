"""GeneWeaver - Python Web Application Entrypoint.

Run directly from your terminal:
    python main.py
or:
    python run.py
"""

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

# Add project root to sys.path so 'app' imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
import torch


def open_browser_when_ready(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Wait for uvicorn to initialize, then launch default web browser."""
    time.sleep(1.2)
    url = f"http://{host}:{port}"
    print(f"\n🌐 Opening GeneWeaver in your browser at: {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def print_banner(host: str, port: int) -> None:
    """Print informative startup diagnostics."""
    cuda_status = f"✅ CUDA Available ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "⚡ CPU Mode (Vectorized multi-core GEMM)"
    
    banner = f"""
================================================================================
   🧬  GENEWEAVER: GPU-Accelerated CRISPR Alignment & Off-Target Engine
================================================================================
   • Backend:      Python {sys.version.split()[0]} (FastAPI + PyTorch {torch.__version__})
   • Acceleration: {cuda_status}
   • Database:     Async SQLite (SQLAlchemy 2.0 ORM)
   • Web UI:       http://{host}:{port}
   • Swagger API:  http://{host}:{port}/docs
================================================================================
   🚀 Starting server... Press Ctrl+C to stop.
"""
    print(banner)


def main() -> None:
    """Main application runner."""
    host = "127.0.0.1"
    port = 8000

    print_banner(host, port)

    # Launch browser automatically in a background daemon thread
    threading.Thread(target=open_browser_when_ready, args=(host, port), daemon=True).start()

    # Run FastAPI via Uvicorn
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
