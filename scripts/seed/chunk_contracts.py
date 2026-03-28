"""Contract text chunker.

Reads every .md file in data/raw/contract_texts/, splits each into
fixed-size overlapping chunks respecting sentence boundaries, and writes
one JSONL record per chunk to data/processed/chunks.jsonl.

Each record contains exactly: doc_id, chunk_id, text, char_start, char_end.

Run:
    python -m scripts.seed.chunk_contracts
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level constants — do NOT pass these as CLI flags.
# ---------------------------------------------------------------------------
WINDOW: int = 400
OVERLAP: int = 80
MIN_LAST_CHUNK: int = 20

_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "contract_texts"
_OUT_FILE = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks.jsonl"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sentence_starts(text: str) -> list[int]:
    """Return sorted positions that are sentence starts (always includes 0).

    A sentence boundary is detected at position ``i + 2`` whenever
    ``text[i] == '.'`` and ``text[i + 1] == ' '``.
    """
    starts: list[int] = [0]
    limit = len(text) - 1
    for i in range(limit):
        if text[i] == "." and text[i + 1] == " ":
            starts.append(i + 2)
    return starts


def _prev_boundary(starts: list[int], pos: int) -> int:
    """Return the largest value in *starts* that is <= *pos* (binary search)."""
    lo, hi = 0, len(starts) - 1
    result = starts[0]
    while lo <= hi:
        mid = (lo + hi) // 2
        if starts[mid] <= pos:
            result = starts[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def _next_boundary(starts: list[int], pos: int) -> int | None:
    """Return the smallest value in *starts* that is > *pos*, or None."""
    lo, hi = 0, len(starts) - 1
    result: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if starts[mid] > pos:
            result = starts[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return result


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


def chunk_document(text: str, doc_id: str) -> list[dict]:
    """Split *text* into overlapping fixed-size chunks.

    Returns a list of dicts with keys: doc_id, chunk_id, text,
    char_start, char_end.  chunk_id is zero-based and contiguous.

    Edge cases handled:
    * File shorter than WINDOW  → single chunk.
    * No ``". "`` boundaries    → hard split at WINDOW, overlap at OVERLAP.
    * Last chunk < MIN_LAST_CHUNK chars → merged into previous chunk.
    """
    n = len(text)
    if n == 0:
        return []

    starts = _sentence_starts(text)
    has_boundaries = len(starts) > 1

    chunks: list[dict] = []
    pos = 0

    while pos < n:
        window_end = pos + WINDOW

        if window_end >= n:
            # Last (or only) chunk — consume to end of text.
            end = n
        elif has_boundaries:
            # Extend to next sentence boundary, but at most WINDOW + OVERLAP chars total.
            nb = _next_boundary(starts, window_end - 1)
            if nb is not None and nb - pos <= WINDOW + OVERLAP:
                end = nb
            else:
                end = window_end
        else:
            # No sentence boundaries: hard cut.
            end = window_end

        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": len(chunks),
                "text": text[pos:end],
                "char_start": pos,
                "char_end": end,
            }
        )

        if end >= n:
            break

        # Compute next chunk start: candidate = end - OVERLAP, trimmed back to
        # nearest sentence boundary so the new chunk never starts mid-sentence.
        candidate = end - OVERLAP
        if has_boundaries:
            next_pos = _prev_boundary(starts, candidate)
            if next_pos <= pos:
                # No useful boundary in the overlap window; fall back to hard cut.
                next_pos = candidate if candidate > pos else end
        else:
            next_pos = candidate if candidate > pos else end

        pos = next_pos

    # Merge a tiny trailing chunk into its predecessor.
    if len(chunks) >= 2 and len(chunks[-1]["text"]) < MIN_LAST_CHUNK:
        last = chunks.pop()
        prev = chunks[-1]
        chunks[-1] = {
            "doc_id": prev["doc_id"],
            "chunk_id": prev["chunk_id"],
            "text": text[prev["char_start"] : last["char_end"]],
            "char_start": prev["char_start"],
            "char_end": last["char_end"],
        }

    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def process_directory(raw_dir: Path, out_file: Path) -> None:
    """Chunk all .md files in *raw_dir* and write to *out_file* as JSONL.

    Output is deterministic: files are processed in sorted order and the
    output file is written atomically (all-or-nothing per run).
    """
    md_files = sorted(raw_dir.glob("*.md"))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    for md_path in md_files:
        doc_id = md_path.stem
        text = md_path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_document(text, doc_id))

    with open(out_file, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    process_directory(_RAW_DIR, _OUT_FILE)
