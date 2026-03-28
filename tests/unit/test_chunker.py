"""Unit tests for scripts.seed.chunk_contracts.

Nine tests in three groups:
  Group 1 (normal window & overlap) — tests 1-3
  Group 2 (short-file edge cases)   — tests 4-6
  Group 3 (overlap boundary trimming) — tests 7-9
"""

from __future__ import annotations

from scripts.seed.chunk_contracts import (
    OVERLAP,
    WINDOW,
    _sentence_starts,
    chunk_document,
)

# ---------------------------------------------------------------------------
# Group 1: normal window and overlap behaviour on a standard multi-sentence input
# ---------------------------------------------------------------------------


def test_long_text_produces_multiple_chunks() -> None:
    # 30 repetitions of a 27-char sentence → 810 chars, well above WINDOW=400.
    sentence = "The quick brown fox jumps. "
    text = sentence * 30
    chunks = chunk_document(text, "doc_multi")
    assert len(chunks) >= 2, "Text longer than WINDOW must produce more than one chunk"


def test_chunk_ids_are_sequential_from_zero() -> None:
    sentence = "Alpha sentence one ends. Beta sentence two ends. "
    text = sentence * 20  # 980 chars
    chunks = chunk_document(text, "doc_ids")
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks))), "chunk_id must be zero-based and contiguous"


def test_consecutive_chunks_overlap_in_character_range() -> None:
    sentence = "Short sent end here. "  # 21 chars
    text = sentence * 50  # 1050 chars
    chunks = chunk_document(text, "doc_overlap")
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        assert chunks[i + 1]["char_start"] < chunks[i]["char_end"], (
            f"chunk {i + 1} must start before chunk {i} ends (overlap required)"
        )


# ---------------------------------------------------------------------------
# Group 2: short-file edge cases
# ---------------------------------------------------------------------------


def test_file_shorter_than_window_emits_single_chunk() -> None:
    text = "Short file. Only one window here."
    assert len(text) < WINDOW
    chunks = chunk_document(text, "short_doc")
    assert len(chunks) == 1
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == len(text)
    assert chunks[0]["chunk_id"] == 0


def test_file_exactly_window_size_emits_single_chunk() -> None:
    text = "x" * WINDOW
    chunks = chunk_document(text, "exact_doc")
    assert len(chunks) == 1
    assert chunks[0]["char_end"] == WINDOW


def test_no_period_space_falls_back_to_hard_split() -> None:
    # No ". " anywhere → _sentence_starts returns [0] → hard split at WINDOW.
    text = "A" * (WINDOW + OVERLAP + 10)  # 490 chars, no sentence boundaries
    chunks = chunk_document(text, "no_boundary_doc")
    assert len(chunks) >= 2, "Text with no boundaries but > WINDOW must produce > 1 chunk"
    # First chunk must start at 0 and be exactly WINDOW chars (hard cut).
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == WINDOW


# ---------------------------------------------------------------------------
# Group 3: overlap boundary trimming — no partial sentence at chunk start
# ---------------------------------------------------------------------------


def test_second_chunk_starts_at_sentence_boundary() -> None:
    # 27-char repeating sentence: boundaries at multiples of 27.
    sentence = "The quick brown fox jumps. "
    text = sentence * 30
    chunks = chunk_document(text, "boundary_doc")
    assert len(chunks) >= 2
    boundaries = set(_sentence_starts(text))
    assert chunks[1]["char_start"] in boundaries, (
        f"chunk[1] starts at {chunks[1]['char_start']}, not a sentence boundary"
    )


def test_all_chunk_starts_are_sentence_boundaries() -> None:
    # Short sentences (31 chars each) so many boundaries fall in the overlap window.
    sentences = [f"Sentence number {i:03d} ends here" for i in range(30)]
    text = ". ".join(sentences) + "."
    chunks = chunk_document(text, "all_boundary_doc")
    boundaries = set(_sentence_starts(text))
    for chunk in chunks:
        assert chunk["char_start"] in boundaries, (
            f"chunk {chunk['chunk_id']} starts at {chunk['char_start']}, not a boundary"
        )


def test_overlap_trims_back_not_forward() -> None:
    # Sentence every 27 chars.  The overlap candidate (end - OVERLAP) will
    # land between two sentence starts; _prev_boundary must choose the EARLIER
    # one (trim back), meaning char_start < (end - OVERLAP).
    sentence = "The quick brown fox jumps. "  # 27 chars
    text = sentence * 30
    chunks = chunk_document(text, "trim_back_doc")
    assert len(chunks) >= 2
    # For each interior chunk, verify start <= end_prev - OVERLAP
    # (i.e., we moved back, never forward from the candidate).
    for i in range(1, len(chunks)):
        prev_end = chunks[i - 1]["char_end"]
        candidate = prev_end - OVERLAP
        assert chunks[i]["char_start"] <= candidate, (
            f"chunk {i} start {chunks[i]['char_start']} > candidate {candidate}: "
            "trimming went forward instead of back"
        )
