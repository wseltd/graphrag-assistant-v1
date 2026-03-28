"""Sentence-transformers implementation of EmbeddingProvider.

The model is loaded eagerly in ``__init__`` so that FastAPI's lifespan hook
controls when the cost is paid and concurrent ingestion requests cannot
trigger parallel model downloads.
"""
from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from graphrag_assistant.config import settings
from graphrag_assistant.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding provider backed by a sentence-transformers model.

    Default model: all-MiniLM-L6-v2 (384 dims, CPU-friendly, no GPU required).
    Batch size 32 balances memory and throughput on CPU during ingestion.
    """

    def __init__(self, model_name: str | None = None) -> None:
        name = model_name if model_name is not None else settings.embedding_model
        logger.info("Loading embedding model: %s", name)
        self._model = SentenceTransformer(name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self._model!r})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        Args:
            texts: Texts to embed.  An empty list returns [] immediately.

        Returns:
            List of float vectors in the same order as *texts*.
            All vectors have the same length (the model's output dimension).
        """
        if not texts:
            return []
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
