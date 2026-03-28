"""Unit tests for SentenceTransformerProvider.

11 tests covering the acceptance criteria for T007:
  2 risky — embed empty list (value and encode-not-called)
  1 — embed single text returns length-1 list
  1 — embed batch returns correct count
  3 risky — all vectors same dimension with mixed-length input strings
  2 risky — model loaded in __init__ not on first embed() call
  1 — isinstance check against EmbeddingProvider
  1 — invalid model name raises on init, not on embed
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphrag_assistant.providers.base import EmbeddingProvider
from graphrag_assistant.providers.embedding import SentenceTransformerProvider

_FAKE_DIM = 384
_ST_PATH = "graphrag_assistant.providers.embedding.SentenceTransformer"


def _encode_fn(dim: int = _FAKE_DIM):
    """Return a side-effect function that produces fake vectors of *dim* length."""

    def encode(texts, batch_size=32, show_progress_bar=False):
        return [[0.1] * dim for _ in texts]

    return encode


def _mock_model(dim: int = _FAKE_DIM) -> MagicMock:
    m = MagicMock()
    m.encode.side_effect = _encode_fn(dim)
    return m


# ---------------------------------------------------------------------------
# Risky: empty list
# ---------------------------------------------------------------------------


@patch(_ST_PATH)
def test_embed_empty_returns_empty_list(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_model()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    assert provider.embed([]) == []


@patch(_ST_PATH)
def test_embed_empty_does_not_call_encode(mock_st: MagicMock) -> None:
    model = _mock_model()
    mock_st.return_value = model
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    provider.embed([])
    assert model.encode.call_count == 0


# ---------------------------------------------------------------------------
# Single text and batch count
# ---------------------------------------------------------------------------


@patch(_ST_PATH)
def test_embed_single_text_returns_one_vector(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_model()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    result = provider.embed(["procurement clause analysis"])
    assert len(result) == 1


@patch(_ST_PATH)
def test_embed_batch_returns_correct_count(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_model()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    assert len(provider.embed(texts)) == len(texts)


# ---------------------------------------------------------------------------
# Risky: dimension consistency across mixed-length inputs
# ---------------------------------------------------------------------------


@patch(_ST_PATH)
def test_all_vectors_same_dimension_short_inputs(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_model()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    dims = [len(v) for v in provider.embed(["a", "b", "c"])]
    assert len(set(dims)) == 1


@patch(_ST_PATH)
def test_all_vectors_same_dimension_mixed_length_inputs(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_model()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    texts = [
        "x",
        "a much longer sentence with many words representing a full contract clause",
        "medium length text about procurement",
        "y",
    ]
    dims = [len(v) for v in provider.embed(texts)]
    assert len(set(dims)) == 1


@patch(_ST_PATH)
def test_all_vectors_same_dimension_large_batch(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_model()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    texts = [
        f"contract document {i} with variable length content {'x' * i}"
        for i in range(50)
    ]
    dims = [len(v) for v in provider.embed(texts)]
    assert len(set(dims)) == 1


# ---------------------------------------------------------------------------
# Risky: model loaded eagerly in __init__, not on first embed()
# ---------------------------------------------------------------------------


@patch(_ST_PATH)
def test_model_loaded_in_init_not_on_embed(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_model()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    assert mock_st.call_count == 1
    provider.embed(["test"])
    assert mock_st.call_count == 1


@patch(_ST_PATH)
def test_model_not_reloaded_on_repeated_embed_calls(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_model()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    for _ in range(5):
        provider.embed(["text"])
    assert mock_st.call_count == 1


# ---------------------------------------------------------------------------
# isinstance and invalid model
# ---------------------------------------------------------------------------


@patch(_ST_PATH)
def test_isinstance_check_passes(mock_st: MagicMock) -> None:
    mock_st.return_value = MagicMock()
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    assert isinstance(provider, EmbeddingProvider)


@patch(_ST_PATH)
def test_invalid_model_name_raises_on_init(mock_st: MagicMock) -> None:
    mock_st.side_effect = OSError("model not found: does-not-exist-xyzzy")
    with pytest.raises(OSError):
        SentenceTransformerProvider("does-not-exist-xyzzy")
