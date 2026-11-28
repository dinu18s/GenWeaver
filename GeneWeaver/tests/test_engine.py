"""Unit and integration test suite for GeneWeaver Core Engine & Scoring."""

import pytest
import torch

from app.core.scoring import (
    calculate_cfd_score,
    calculate_mit_score,
    classify_cleavage_tier,
)
from app.engine.aligner import CrisprAligner
from app.engine.pam import (
    extract_candidate_windows,
    is_pam_match,
    pam_to_regex,
    parse_fasta,
    reverse_complement,
    sanitize_sequence,
)


class TestPAMModule:
    """Test suite for PAM parsing, validation, and genomic window extraction."""

    def test_reverse_complement_basic(self):
        assert reverse_complement("ATCG") == "CGAT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("GATTACA") == "TGTAATC"

    def test_reverse_complement_rna_handling(self):
        assert reverse_complement("AUCG") == "CGAT"
        assert reverse_complement("uucc") == "GGAA"

    def test_pam_canonical_matching(self):
        for valid_pam in ["AGG", "CGG", "GGG", "TGG"]:
            assert is_pam_match(valid_pam, "NGG") is True
            assert is_pam_match(valid_pam.lower(), "NGG") is True

        for invalid_pam in ["AGA", "AGT", "AGC", "TGA", "AAA"]:
            assert is_pam_match(invalid_pam, "NGG") is False

    def test_pam_degenerate_iupac(self):
        # NAG motif
        assert is_pam_match("AAG", "NAG") is True
        assert is_pam_match("CAG", "NAG") is True
        assert is_pam_match("CGG", "NAG") is False

        # SaCas9 NNGRRT motif
        assert is_pam_match("CCGAAT", "NNGRRT") is True
        assert is_pam_match("TTGGAT", "NNGRRT") is True
        assert is_pam_match("CCGATT", "NNGRRT") is False

    def test_sanitize_sequence(self):
        raw = " \n a t c g \r\n u u n x \t "
        clean = sanitize_sequence(raw)
        assert clean == "ATCGTTNN"

    def test_parse_fasta_multi_record(self):
        fasta = """>chr1_sample
ACGTACGT
ACGTACGT
>chr2_sample
TGCATGCA"""
        records = parse_fasta(fasta)
        assert len(records) == 2
        assert records[0][0] == "chr1_sample"
        assert records[0][1] == "ACGTACGTACGTACGT"
        assert records[1][0] == "chr2_sample"
        assert records[1][1] == "TGCATGCA"

    def test_window_extraction_strands(self):
        guide = "GAGTCCGAGCAGAAGAAGAA"  # 20 nt
        pam = "TGG"  # 3 nt
        # Forward target
        seq_fwd = "NNN" + guide + pam + "NNN"
        windows_fwd = list(extract_candidate_windows(seq_fwd, pam_pattern="NGG", search_both_strands=True))
        assert len(windows_fwd) >= 1
        fwd_hit = [w for w in windows_fwd if w[2] == "+"][0]
        assert fwd_hit[0] == 3
        assert fwd_hit[1] == 23
        assert fwd_hit[3] == guide
        assert fwd_hit[4] == pam

        # Reverse target
        rc_target = reverse_complement(guide + pam)
        seq_rev = "NNN" + rc_target + "NNN"
        windows_rev = list(extract_candidate_windows(seq_rev, pam_pattern="NGG", search_both_strands=True))
        rev_hit = [w for w in windows_rev if w[2] == "-"][0]
        assert rev_hit[3] == guide
        assert rev_hit[4] == pam


class TestScoringAlgorithms:
    """Test suite for MIT and CFD biophysical cutting risk models."""

    def test_mit_exact_match(self):
        guide = "GAGTCCGAGCAGAAGAAGAA"
        score = calculate_mit_score(guide, guide)
        assert score == 100.0

    def test_mit_positional_sensitivity(self):
        guide = "GAGTCCGAGCAGAAGAAGAA"
        # Mutation at PAM-distal position 1 (5' end)
        mut_distal = "A" + guide[1:]
        score_distal = calculate_mit_score(guide, mut_distal)

        # Mutation at seed position 19 (proximal to PAM)
        mut_seed = guide[:18] + "C" + guide[19:]
        score_seed = calculate_mit_score(guide, mut_seed)

        # Seed mutation has much higher penalty, so remaining cutting score is much lower
        assert score_distal > score_seed
        assert score_distal > 80.0
        assert score_seed < 40.0

    def test_mit_multiple_mismatches_distance_effect(self):
        guide = "GAGTCCGAGCAGAAGAAGAA"
        # Test distance factor independently by selecting positions with identical zero weight:
        # Positions 1, 4, 5, 8 all have weight 0.0 in Hsu et al.
        # Clustered pair: positions 4 and 5 (distance = 1) -> mutate pos 4 ('T'->'A') and pos 5 ('C'->'A')
        clustered = guide[:3] + "AA" + guide[5:]
        score_clustered = calculate_mit_score(guide, clustered)

        # Distant pair: positions 1 and 8 (distance = 7) -> mutate pos 1 ('G'->'A') and pos 8 ('A'->'T')
        distant = "A" + guide[1:7] + "T" + guide[8:]
        score_distant = calculate_mit_score(guide, distant)

        # Hsu et al. imposes greater penalty on clustered mismatches (higher score for distant)
        assert score_distant > score_clustered
        assert score_clustered == 5.2198
        assert score_distant == 7.0896

    def test_cfd_exact_match(self):
        guide = "GAGTCCGAGCAGAAGAAGAA"
        score = calculate_cfd_score(guide, guide, pam_seq="TGG")
        assert score == 1.0

    def test_cfd_pam_penalties(self):
        guide = "GAGTCCGAGCAGAAGAAGAA"
        cfd_ngg = calculate_cfd_score(guide, guide, pam_seq="CGG")
        cfd_nag = calculate_cfd_score(guide, guide, pam_seq="CAG")
        cfd_nga = calculate_cfd_score(guide, guide, pam_seq="CGA")

        assert cfd_ngg == 1.0
        assert round(cfd_nag, 3) == 0.259
        assert round(cfd_nga, 3) == 0.069

    def test_cleavage_tier_classification(self):
        assert classify_cleavage_tier(100.0, 1.0, 0) == "High"
        assert classify_cleavage_tier(75.0, 0.6, 1) == "High"
        assert classify_cleavage_tier(35.0, 0.25, 2) == "Medium"
        assert classify_cleavage_tier(10.0, 0.02, 4) == "Low"


class TestTensorAligner:
    """Test suite for PyTorch vectorized tensor matching and hardware fallbacks."""

    @pytest.fixture
    def aligner(self):
        return CrisprAligner(device_preference="cpu")

    def test_tensor_encoding_shape_and_values(self, aligner):
        seqs = ["ACGTACGTACGTACGTACGT", "AAAAAAAAAAAAAAAAAAAA"]
        tensor = aligner.encode_sequences(seqs)
        assert tensor.shape == (2, 80)
        assert tensor.dtype == torch.float32
        # Sum of each row should equal 20 (one 1.0 per position)
        assert torch.allclose(tensor.sum(dim=-1), torch.tensor([20.0, 20.0]))

    def test_align_perfect_hit(self, aligner):
        guide = "GAGTCCGAGCAGAAGAAGAA"
        fasta = f">chr_test\nNNNN{guide}TGGNNNN"
        hits, summary = aligner.align_sequence(guide, fasta, pam_pattern="NGG", mismatch_threshold=3)

        assert len(hits) == 1
        hit = hits[0]
        assert hit.sequence == guide
        assert hit.pam == "TGG"
        assert hit.mismatches == 0
        assert hit.cleavage_tier == "High"
        assert hit.mit_score == 100.0
        assert hit.cfd_score == 1.0

    def test_mismatch_threshold_cutoff(self, aligner):
        guide = "GAGTCCGAGCAGAAGAAGAA"
        # 1-mm target
        t1 = guide[:19] + "T"
        # 4-mm target
        t4 = "TTTT" + guide[4:]

        fasta = f">test_mm\n{t1}TGG\nNNNN\n{t4}TGG"

        # Threshold = 2 should find t1 but NOT t4
        hits_2mm, _ = aligner.align_sequence(guide, fasta, mismatch_threshold=2)
        assert len(hits_2mm) == 1
        assert hits_2mm[0].mismatches == 1

        # Threshold = 4 should find both t1 and t4
        hits_4mm, _ = aligner.align_sequence(guide, fasta, mismatch_threshold=4)
        assert len(hits_4mm) == 2

    def test_zero_hits_on_divergent_sequence(self, aligner):
        guide = "AAAAAAAAAAAAAAAAAAAA"
        divergent = ">divergent\n" + "C" * 200 + "AGG" + "C" * 50
        hits, summary = aligner.align_sequence(guide, divergent, mismatch_threshold=2)
        assert len(hits) == 0
        assert summary["total_hits"] == 0
