"""GPU-Accelerated CRISPR-Cas9 Tensor Alignment Engine using PyTorch."""

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.core.scoring import (
    calculate_cfd_score,
    calculate_mit_score,
    classify_cleavage_tier,
)
from app.engine.pam import (
    extract_candidate_windows,
    parse_fasta,
    sanitize_sequence,
)

# ASCII base mapping table for vectorized conversion
# 'A': 0, 'C': 1, 'G': 2, 'T': 3
_BASE_LUT = np.zeros(256, dtype=np.int64)
_BASE_LUT[ord("A")] = 0
_BASE_LUT[ord("C")] = 1
_BASE_LUT[ord("G")] = 2
_BASE_LUT[ord("T")] = 3


@dataclass
class AlignmentHitResult:
    """Individual alignment hit representation."""
    target_name: str
    start_pos: int
    end_pos: int
    strand: str
    sequence: str
    pam: str
    mismatches: int
    mismatch_positions: List[int]
    mit_score: float
    cfd_score: float
    cleavage_tier: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrisprAligner:
    """High-performance tensor alignment engine for CRISPR gRNA off-target discovery.
    
    Supports automatic GPU acceleration with CUDA and seamless vector-optimized
    fallback to multi-core CPU.
    """

    def __init__(
        self,
        device_preference: str = "auto",
        batch_size: int = settings.DEFAULT_BATCH_SIZE,
    ) -> None:
        self.device_preference = device_preference
        self.batch_size = batch_size
        self.device = self._resolve_device(device_preference)
        self.is_cuda = self.device.type == "cuda"

    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        """Resolve torch device based on availability and preference."""
        if preference in ("auto", "cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda")
        return torch.device("cpu")

    @property
    def device_name(self) -> str:
        """Return human-readable device identifier."""
        if self.is_cuda:
            return f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        cpu_threads = torch.get_num_threads()
        return f"CPU ({cpu_threads} threads, PyTorch Vectorized)"

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Vectorized one-hot encoding of 20-nt DNA sequences into PyTorch float tensors.
        
        Args:
            sequences: List of 20-character uppercase DNA strings.
            
        Returns:
            torch.Tensor of shape (N, 80) where 80 = 20 positions * 4 one-hot channels.
        """
        n = len(sequences)
        if n == 0:
            return torch.empty((0, 80), dtype=torch.float32, device=self.device)

        # Concatenate and read as ASCII byte buffer
        byte_buffer = "".join(sequences).encode("ascii")
        byte_arr = np.frombuffer(byte_buffer, dtype=np.uint8).reshape(n, 20)
        
        # Map ASCII bytes directly to base indices 0..3 using precomputed lookup table
        base_indices = _BASE_LUT[byte_arr]
        idx_tensor = torch.from_numpy(base_indices).to(self.device)
        
        # One-hot encode (N, 20, 4) and flatten to (N, 80)
        one_hot = F.one_hot(idx_tensor, num_classes=4).float().view(n, 80)
        return one_hot

    def align_candidates(
        self,
        guide_seq: str,
        candidates: List[Tuple[int, int, str, str, str]],
        target_name: str,
        mismatch_threshold: int = 3,
    ) -> List[AlignmentHitResult]:
        """Align a list of PAM-passing candidate windows against the gRNA guide using GPU/CPU tensors.
        
        Args:
            guide_seq: 20-nt guide sequence.
            candidates: List of (start_idx, end_idx, strand, protospacer_20nt, pam).
            target_name: Reference sequence / chromosome name.
            mismatch_threshold: Maximum tolerated mismatches (Hamming distance).
            
        Returns:
            List of AlignmentHitResult objects meeting the mismatch threshold.
        """
        if not candidates:
            return []

        guide_clean = sanitize_sequence(guide_seq)[:20]
        if len(guide_clean) < 20:
            raise ValueError(f"Guide RNA sequence must be 20 nt (got {len(guide_clean)})")

        # Encode guide sequence to shape (80, 1) for GEMM
        guide_tensor = self.encode_sequences([guide_clean]).T  # shape (80, 1)

        results: List[AlignmentHitResult] = []
        total_candidates = len(candidates)

        # Process in batches to maintain high cache locality and prevent GPU OOM
        for offset in range(0, total_candidates, self.batch_size):
            batch_candidates = candidates[offset : offset + self.batch_size]
            batch_kmers = [c[3] for c in batch_candidates]

            # Vectorized candidate one-hot encoding
            batch_tensor = self.encode_sequences(batch_kmers)  # (B, 80)

            # GEMM: (B, 80) x (80, 1) -> (B, 1) = matching base count
            with torch.no_grad():
                matches = torch.matmul(batch_tensor, guide_tensor).squeeze(-1)
                mismatches = 20.0 - matches
                
                # Filter candidates satisfying Hamming distance threshold
                hit_mask = mismatches <= float(mismatch_threshold)
                hit_indices = torch.nonzero(hit_mask, as_tuple=True)[0]

            if len(hit_indices) == 0:
                continue

            hit_indices_list = hit_indices.cpu().tolist()
            for idx in hit_indices_list:
                start_pos, end_pos, strand, protospacer, pam = batch_candidates[idx]
                
                # Determine mismatch positions (1-indexed from 1 to 20)
                mismatch_pos = [
                    pos + 1
                    for pos in range(20)
                    if guide_clean[pos] != protospacer[pos]
                ]
                mm_count = len(mismatch_pos)

                # Compute biophysical and cutting frequency scores
                mit_score = calculate_mit_score(guide_clean, protospacer, mismatch_pos)
                cfd_score = calculate_cfd_score(guide_clean, protospacer, pam)
                tier = classify_cleavage_tier(mit_score, cfd_score, mm_count)

                results.append(
                    AlignmentHitResult(
                        target_name=target_name,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        strand=strand,
                        sequence=protospacer,
                        pam=pam,
                        mismatches=mm_count,
                        mismatch_positions=mismatch_pos,
                        mit_score=mit_score,
                        cfd_score=cfd_score,
                        cleavage_tier=tier,
                    )
                )

        return results

    def align_sequence(
        self,
        guide_seq: str,
        fasta_content: str,
        pam_pattern: str = "NGG",
        mismatch_threshold: int = 3,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Tuple[List[AlignmentHitResult], Dict[str, Any]]:
        """Complete alignment pipeline: parse FASTA, scan PAM windows, and perform tensor matching.
        
        Returns:
            Tuple of (hits_list, summary_stats_dict).
        """
        start_time = time.perf_counter()
        records = parse_fasta(fasta_content)
        
        if not records:
            raise ValueError("No valid FASTA records or sequences provided.")

        all_hits: List[AlignmentHitResult] = []
        total_kmers_scanned = 0
        total_pam_sites = 0

        total_records = len(records)
        for rec_idx, (header, sequence) in enumerate(records):
            seq_len = len(sequence)
            total_kmers_scanned += max(0, seq_len - 22) * 2  # both strands

            # Extract PAM-passing candidate protospacer windows
            candidates = list(
                extract_candidate_windows(
                    sequence=sequence,
                    pam_pattern=pam_pattern,
                    grna_length=20,
                    search_both_strands=True,
                )
            )
            total_pam_sites += len(candidates)

            # Align candidates against guide
            hits = self.align_candidates(
                guide_seq=guide_seq,
                candidates=candidates,
                target_name=header,
                mismatch_threshold=mismatch_threshold,
            )
            all_hits.extend(hits)

            if progress_callback:
                progress = round(((rec_idx + 1) / total_records) * 100.0, 1)
                progress_callback(progress)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Sort hits: prioritize fewest mismatches, then highest CFD score
        all_hits.sort(key=lambda h: (h.mismatches, -h.cfd_score, -h.mit_score))

        # Summarize risk profile
        high_risk_count = sum(1 for h in all_hits if h.cleavage_tier == "High")
        med_risk_count = sum(1 for h in all_hits if h.cleavage_tier == "Medium")
        low_risk_count = sum(1 for h in all_hits if h.cleavage_tier == "Low")

        mismatch_dist = {i: 0 for i in range(mismatch_threshold + 1)}
        for h in all_hits:
            if h.mismatches in mismatch_dist:
                mismatch_dist[h.mismatches] += 1

        summary = {
            "total_kmers_scanned": total_kmers_scanned,
            "total_pam_sites": total_pam_sites,
            "total_hits": len(all_hits),
            "execution_time_ms": elapsed_ms,
            "device_used": self.device_name,
            "high_risk_hits": high_risk_count,
            "medium_risk_hits": med_risk_count,
            "low_risk_hits": low_risk_count,
            "mismatch_distribution": mismatch_dist,
        }

        return all_hits, summary
