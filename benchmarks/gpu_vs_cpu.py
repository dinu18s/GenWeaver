"""GeneWeaver Benchmark Suite: High-Performance GPU vs CPU Tensor Alignment Throughput."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.engine.aligner import CrisprAligner


def run_synthetic_benchmark(
    batch_sizes: List[int] = [10000, 50000, 100000],
    iterations: int = 5,
) -> Dict[str, Any]:
    """Execute high-precision throughput profiling on available hardware architectures."""
    print("=" * 85)
    print(" GeneWeaver: High-Performance CRISPR Alignment Benchmark")
    print(f" PyTorch Version : {torch.__version__}")
    print(f" CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" CUDA Device     : {torch.cuda.get_device_name(0)}")
    print(f" CPU Threads     : {torch.get_num_threads()}")
    print("=" * 85)

    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")

    guide = "GAGTCCGAGCAGAAGAAGAA"
    results: Dict[str, Any] = {
        "timestamp": time.time(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_results": [],
        "cuda_results": [] if torch.cuda.is_available() else None,
    }

    for dev_name in devices_to_test:
        print(f"\n[+] Profiling device: {dev_name.upper()}...")
        aligner = CrisprAligner(device_preference=dev_name)

        # 1. Raw Tensor GEMM Throughput
        print("  --- Pure Tensor GEMM Distance Calculation ---")
        for b_size in batch_sizes:
            # Generate random 20-mers
            bases = np.array(["A", "C", "G", "T"])
            rand_kmers = ["".join(np.random.choice(bases, size=20)) for _ in range(b_size)]
            
            # Encode
            guide_tensor = aligner.encode_sequences([guide]).T
            batch_tensor = aligner.encode_sequences(rand_kmers)

            # Warmup
            for _ in range(2):
                _ = torch.matmul(batch_tensor[:1000], guide_tensor)

            gemm_latencies = []
            for _ in range(iterations):
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.no_grad():
                    matches = torch.matmul(batch_tensor, guide_tensor).squeeze(-1)
                    mismatches = 20.0 - matches
                    _ = mismatches <= 3.0

                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                gemm_latencies.append((t1 - t0) * 1000.0)

            mean_gemm = float(np.mean(gemm_latencies))
            gemm_tps = float(round(b_size / (mean_gemm / 1000.0), 1)) if mean_gemm > 0 else 0.0
            print(
                f"   [GEMM] Batch: {b_size:7,d} | "
                f"Latency: {mean_gemm:7.3f} ms | "
                f"Throughput: {gemm_tps:12,.1f} k-mers/s"
            )

        # 2. End-to-End Pipeline Throughput (Realistic Genomic Hit Rate)
        print("  --- End-to-End Pipeline (Tensor Filter + MIT/CFD Scoring) ---")
        for b_size in batch_sizes:
            # Generate candidate list with realistic 0.1% hit rate
            candidates = [
                (i, i + 20, "+", guide[:17] + "NNN" if i % 1000 == 0 else rand_kmers[i], "TGG")
                for i in range(b_size)
            ]

            latencies = []
            for _ in range(iterations):
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                hits = aligner.align_candidates(guide, candidates, "benchmark", 3)

                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

            mean_latency = float(np.mean(latencies))
            throughput = float(round(b_size / (mean_latency / 1000.0), 1)) if mean_latency > 0 else 0.0

            metric = {
                "batch_size": b_size,
                "gemm_latency_ms": round(mean_gemm, 3),
                "gemm_kmers_per_sec": gemm_tps,
                "e2e_latency_ms": round(mean_latency, 3),
                "e2e_kmers_per_sec": throughput,
                "device": aligner.device_name,
            }

            if dev_name == "cpu":
                results["cpu_results"].append(metric)
            else:
                results["cuda_results"].append(metric)

            print(
                f"   [E2E]  Batch: {b_size:7,d} | "
                f"Latency: {mean_latency:7.3f} ms | "
                f"Throughput: {throughput:12,.1f} k-mers/s"
            )

    output_path = ROOT_DIR / "benchmarks" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Benchmark metrics successfully saved to {output_path}")
    print("=" * 85)
    return results


if __name__ == "__main__":
    run_synthetic_benchmark()
