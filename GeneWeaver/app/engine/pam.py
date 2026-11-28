"""PAM identification, validation, and genomic window extraction for CRISPR targets."""

import re
from typing import Dict, Generator, List, Set, Tuple

IUPAC_MAP: Dict[str, Set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "U": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}

COMPLEMENT_MAP: Dict[str, str] = {
    "A": "T",
    "T": "A",
    "U": "A",
    "C": "G",
    "G": "C",
    "N": "N",
    "R": "Y",
    "Y": "R",
    "S": "S",
    "W": "W",
    "K": "M",
    "M": "K",
    "B": "V",
    "D": "H",
    "H": "D",
    "V": "B",
}


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a nucleotide sequence."""
    seq_clean = seq.upper().replace("U", "T")
    return "".join(COMPLEMENT_MAP.get(b, "N") for b in reversed(seq_clean))


def pam_to_regex(pam_pattern: str) -> str:
    """Convert an IUPAC PAM string (e.g. 'NGG', 'NNGRRT') into a regular expression."""
    regex_parts = []
    for char in pam_pattern.upper():
        allowed = IUPAC_MAP.get(char, {"A", "C", "G", "T"})
        if len(allowed) == 1:
            regex_parts.append(list(allowed)[0])
        elif len(allowed) == 4:
            regex_parts.append("[ACGT]")
        else:
            regex_parts.append(f"[{''.join(sorted(allowed))}]")
    return "".join(regex_parts)


def is_pam_match(pam_seq: str, pam_pattern: str = "NGG") -> bool:
    """Check if a candidate PAM sequence satisfies an IUPAC PAM pattern."""
    if len(pam_seq) != len(pam_pattern):
        return False
    pam_seq = pam_seq.upper().replace("U", "T")
    pam_pattern = pam_pattern.upper()
    
    for b, p in zip(pam_seq, pam_pattern):
        allowed = IUPAC_MAP.get(p, {"A", "C", "G", "T"})
        if b not in allowed:
            return False
    return True


def sanitize_sequence(seq: str) -> str:
    """Clean and normalize a nucleotide sequence to uppercase ACGT."""
    # Strip whitespace, newlines, carriage returns, and convert U to T
    cleaned = re.sub(r"\s+", "", seq).upper().replace("U", "T")
    # Replace non-standard bases with N
    return re.sub(r"[^ACGTN]", "N", cleaned)


def parse_fasta(fasta_content: str) -> List[Tuple[str, str]]:
    """Parse raw FASTA string into a list of (header, sequence) pairs.
    
    If no FASTA headers are present (plain raw sequence), a default header is assigned.
    """
    lines = fasta_content.strip().splitlines()
    records: List[Tuple[str, str]] = []
    current_header = ""
    current_seq_parts: List[str] = []
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if line_clean.startswith(">"):
            if current_header or current_seq_parts:
                records.append((current_header or "seq_1", "".join(current_seq_parts)))
                current_seq_parts = []
            current_header = line_clean[1:].strip()
        else:
            current_seq_parts.append(line_clean)
            
    if current_seq_parts or current_header:
        records.append((current_header or "target_sequence", "".join(current_seq_parts)))
        
    # Sanitize all parsed sequences
    return [(hdr, sanitize_sequence(s)) for hdr, s in records if s]


def extract_candidate_windows(
    sequence: str,
    pam_pattern: str = "NGG",
    grna_length: int = 20,
    search_both_strands: bool = True
) -> Generator[Tuple[int, int, str, str, str], None, None]:
    """Extract all candidate target protospacers meeting PAM criteria.
    
    Args:
        sequence: Normalized genomic DNA sequence.
        pam_pattern: IUPAC PAM motif (e.g., 'NGG').
        grna_length: Length of guide RNA (default: 20 nt).
        search_both_strands: Whether to scan forward (+) and reverse (-) strands.
        
    Yields:
        Tuple of (start_idx, end_idx, strand, protospacer_20nt, pam_seq)
        where start_idx and end_idx are 0-based coordinates on the reference sequence.
    """
    seq_len = len(sequence)
    pam_len = len(pam_pattern)
    window_len = grna_length + pam_len
    
    if seq_len < window_len:
        return
        
    # Forward strand (+) scanning
    for i in range(seq_len - window_len + 1):
        window = sequence[i : i + window_len]
        protospacer = window[:grna_length]
        pam = window[grna_length:]
        
        if is_pam_match(pam, pam_pattern):
            yield (i, i + grna_length, "+", protospacer, pam)
            
    # Reverse strand (-) scanning
    if search_both_strands:
        for i in range(seq_len - window_len + 1):
            window = sequence[i : i + window_len]
            # On the minus strand, the reverse complement represents the 5'->3' target
            rc_window = reverse_complement(window)
            rc_protospacer = rc_window[:grna_length]
            rc_pam = rc_window[grna_length:]
            
            if is_pam_match(rc_pam, pam_pattern):
                yield (i, i + grna_length, "-", rc_protospacer, rc_pam)
