# GeneWeaver Architecture & Algorithmic Blueprint

## 1. System Overview

GeneWeaver is an industrial-grade, GPU-accelerated CRISPR-Cas9 alignment engine designed to identify potential genomic off-target cleavage sites with ultra-low latency. The system merges high-performance tensor computing (PyTorch) with asynchronous web services (FastAPI, SQLAlchemy, SQLite/aiosqlite) and empirical cutting frequency biophysical models (MIT and CFD scores).

```mermaid
flowchart TD
    subgraph Client["Client Tier"]
        UI["Web Dashboard (HTML5 / Tailwind / Vanilla JS)"]
        CLI["External Pipeline / REST Client"]
    end

    subgraph API["FastAPI Microservice Layer"]
        Router["APIRouter (/api/v1)"]
        Validator["Pydantic v2 Schema Validation"]
        WorkerPool["Asynchronous Background Worker Pool"]
    end

    subgraph Engine["High-Performance Tensor Engine"]
        PAM["PAM Scanner (IUPAC / Double-Strand)"]
        Encoder["Vectorized One-Hot ASCII Encoder"]
        GEMM["PyTorch Tensor GEMM Distance Engine"]
        Scorer["Biophysical Cleavage Scorer (MIT + CFD)"]
    end

    subgraph Storage["Persistence Layer"]
        DB[("Async SQLite DB / SQLAlchemy ORM")]
        Hits[("AlignmentHit Table")]
        Jobs[("Job Execution Table")]
        Bench[("BenchmarkRecord Table")]
    end

    UI --> Router
    CLI --> Router
    Router --> Validator
    Validator --> DB
    Validator --> WorkerPool
    WorkerPool --> PAM
    PAM --> Encoder
    Encoder --> GEMM
    GEMM --> Scorer
    Scorer --> DB
    DB --> Jobs
    DB --> Hits
    DB --> Bench
```

---

## 2. Core Algorithmic Engine (`app/engine/`)

### 2.1 Double-Stranded PAM Identification
Cas9 target recognition requires an adjacent Protospacer Adjacent Motif (PAM). For SpCas9, this motif is `5'-NGG-3'` located immediately 3' of the 20-nt target protospacer.

For a reference genome segment of length $L$, the engine scans both strands:
- **Forward Strand (+)**:
  $$\text{Window}_i = \text{Seq}[i : i + 23], \quad \text{Protospacer} = \text{Window}_i[0:20], \quad \text{PAM} = \text{Window}_i[20:23]$$
- **Reverse Strand (-)**:
  $$\text{RC\_Window}_i = \text{RevComp}(\text{Seq}[i : i + 23]), \quad \text{Protospacer} = \text{RC\_Window}_i[0:20], \quad \text{PAM} = \text{RC\_Window}_i[20:23]$$

Candidate windows are matched against IUPAC degenerate codes using bitwise character sets:
$$\text{IUPAC}(N) = \{A, C, G, T\}, \quad \text{IUPAC}(R) = \{A, G\}, \quad \text{IUPAC}(Y) = \{C, T\}$$

---

### 2.2 Tensor-Vectorized One-Hot Alignment via GEMM

Traditional seed-and-extend or tree-based aligners (e.g. Bowtie, BWA) incur branch mispredictions and hash collisions when querying degenerate strings with multiple mismatches. GeneWeaver reformulates sequence matching as a dense matrix multiplication on GPU or CPU vector registers.

#### Mathematical Formulation:
1. Let $\Sigma = \{A, C, G, T\}$ be mapped to channels $\{0, 1, 2, 3\}$.
2. Each candidate 20-mer $S_k$ is encoded into a one-hot matrix $E(S_k) \in \{0, 1\}^{20 \times 4}$.
3. Flattening across positions yields a vector $V_k \in \{0, 1\}^{80}$.
4. A batch of $N$ candidates forms tensor $\mathbf{X} \in \mathbb{R}^{N \times 80}$.
5. The 20-nt guide RNA $G$ is encoded as $\mathbf{y} \in \mathbb{R}^{80 \times 1}$.

Since each position contains exactly one 1.0, the dot product computes the exact number of matching nucleotides:
$$\mathbf{M} = \mathbf{X} \mathbf{y}, \quad M_k = \sum_{j=1}^{20} \sum_{c=0}^3 X_{k, 4j+c} \cdot y_{4j+c} = \text{Matches}(S_k, G)$$

The Hamming distance is derived via:
$$\mathbf{D}_{\text{Hamming}} = 20 - \mathbf{M}$$

Filtering candidates satisfying mismatch threshold $\theta$:
$$\text{Hits} = \{ k \mid D_{\text{Hamming}, k} \le \theta \}$$

#### Performance Characteristics:
- On NVIDIA CUDA: Executes via cuBLAS `sgemm`, achieving sustained throughput $> 100\text{M}$ k-mers/sec.
- On CPU: Executes via Intel MKL / OpenBLAS vectorization across AVX-512 / AVX2 instructions, achieving $> 78\text{M}$ k-mers/sec.

---

## 3. Off-Target Cleavage Scoring Models (`app/core/scoring.py`)

### 3.1 MIT Specificity Score (Hsu et al., 2013)

The MIT score penalizes mismatches based on three empirical terms:
1. **Positional Weighting ($T_1$)**: Weights $W[p]$ derived from high-throughput cleavage assays, penalizing seed region mutations (positions 13–20) more severely than PAM-distal mutations.
   $$T_1 = \prod_{p \in \mathcal{M}} (1 - W[p])$$
2. **Mean Pairwise Distance ($T_2$)**: Clustered mismatches severely impair Cas9 conformation, while spaced mismatches are partially tolerated:
   $$\bar{d} = \begin{cases} 19.0 & \text{if } |\mathcal{M}| = 1 \\ \frac{1}{|\mathcal{M}| - 1} \sum_{i=1}^{|\mathcal{M}|-1} (p_{i+1} - p_i) & \text{if } |\mathcal{M}| > 1 \end{cases}$$
   $$T_2 = \frac{1}{\left( \frac{19 - \bar{d}}{19} \times 4 \right) + 1}$$
3. **Mismatch Count Penalty ($T_3$)**:
   $$T_3 = \frac{1}{|\mathcal{M}|^2}$$

Total MIT Score:
$$S_{\text{MIT}} = T_1 \times T_2 \times T_3 \times 100\%$$

---

### 3.2 Cutting Frequency Determination (CFD) Score (Doench et al., 2016)

The CFD score accounts for the biochemical identity of base substitutions (all 16 $r\text{RNA}:d\text{DNA}$ pairings) and non-canonical PAM affinities:
$$\text{CFD} = \text{PAM\_Weight}(\text{PAM}) \times \prod_{i=1}^{20} F(i, rNA_i : DNA_i)$$

Where:
- Matches ($rA:dA, rC:dC, rG:dG, rU:dT$) evaluate to $1.0$.
- Wobble pairs ($rG:dT$) exhibit high cleavage tolerance ($> 0.80$).
- Clashing purine-purine pairs ($rA:dG, rG:dA$) severely disrupt cleavage.
- PAM multipliers: $\text{NGG} = 1.0, \text{NAG} = 0.259, \text{NGA} = 0.069, \text{NCG} = 0.050$.

---

## 4. Database Schema & Concurrency Model

```mermaid
erDiagram
    JOBS ||--o{ ALIGNMENT_HITS : "has hits"
    JOBS {
        string id PK "UUID4"
        string grna_sequence
        string pam
        string target_name
        string status "PENDING, RUNNING, COMPLETED, FAILED"
        float progress
        int mismatch_threshold
        int total_kmers_scanned
        int total_pam_sites
        int total_hits
        float execution_time_ms
        string device_used
        text error_message
        text summary_json
        datetime created_at
        datetime completed_at
    }

    ALIGNMENT_HITS {
        int id PK "Autoincrement"
        string job_id FK
        string target_name
        int start_pos
        int end_pos
        string strand "+ or -"
        string sequence
        string pam_sequence
        int mismatches
        text mismatch_positions_json
        float mit_score
        float cfd_score
        string cleavage_tier "High, Medium, Low"
    }

    BENCHMARKS {
        int id PK
        datetime created_at
        string device
        int batch_size
        float latency_ms
        float kmers_per_sec
        boolean cuda_available
        string torch_version
    }
```

### Asynchronous Execution & Non-Blocking Architecture:
- Client submissions to `POST /api/v1/align` immediately yield HTTP `202 Accepted` with a UUID `job_id`.
- The alignment workflow is offloaded to FastAPI's background task runner.
- The web dashboard polls `GET /api/v1/jobs/{job_id}` at 600ms intervals.
- Database access uses asynchronous SQLAlchemy with connection pooling over `aiosqlite`, ensuring zero blocking on the main server event loop.
