"""Integration test: process all real contract .md files and validate JSONL output.

Does not require Neo4j or any external service.
"""

from __future__ import annotations

from pathlib import Path

from scripts.seed.chunk_contracts import chunk_document

_CONTRACT_DIR = (
    Path(__file__).parent.parent.parent / "data" / "raw" / "contract_texts"
)

_REQUIRED_KEYS = {"doc_id", "chunk_id", "text", "char_start", "char_end"}


def test_all_contract_files_produce_valid_chunks() -> None:
    md_files = sorted(_CONTRACT_DIR.glob("*.md"))
    assert md_files, f"No .md files found in {_CONTRACT_DIR}"

    seen_docs: set[str] = set()

    for md_path in md_files:
        doc_id = md_path.stem
        text = md_path.read_text(encoding="utf-8")
        chunks = chunk_document(text, doc_id)

        # Every file must contribute at least one chunk.
        assert chunks, f"{md_path.name} produced no chunks"

        seen_docs.add(doc_id)
        expected_ids = list(range(len(chunks)))

        for chunk in chunks:
            # Exactly the five required keys, nothing more, nothing less.
            assert set(chunk.keys()) == _REQUIRED_KEYS, (
                f"{doc_id}: unexpected keys {set(chunk.keys()) ^ _REQUIRED_KEYS}"
            )

            # doc_id matches the filename stem.
            assert chunk["doc_id"] == doc_id

            # chunk_id is a non-negative integer.
            assert isinstance(chunk["chunk_id"], int) and chunk["chunk_id"] >= 0

            # text is non-empty and within the 480-char hard cap.
            assert chunk["text"], f"{doc_id} chunk {chunk['chunk_id']} has empty text"
            assert len(chunk["text"]) <= 480, (
                f"{doc_id} chunk {chunk['chunk_id']} text length "
                f"{len(chunk['text'])} exceeds 480"
            )

            # char offsets are consistent with text.
            assert chunk["char_end"] - chunk["char_start"] == len(chunk["text"]), (
                f"{doc_id} chunk {chunk['chunk_id']}: "
                "char_end - char_start != len(text)"
            )
            assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"], (
                f"{doc_id} chunk {chunk['chunk_id']}: "
                "text slice does not match stored text"
            )

        # chunk_ids are zero-based, sequential, and contiguous.
        actual_ids = [c["chunk_id"] for c in chunks]
        assert actual_ids == expected_ids, (
            f"{doc_id}: chunk_ids {actual_ids} are not sequential from 0"
        )

    # Every .md file in the directory is represented.
    assert seen_docs == {p.stem for p in md_files}
