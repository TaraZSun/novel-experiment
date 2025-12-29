from __future__ import annotations
from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer
from settings import EmbeddingConfig
class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig):
        self.model = SentenceTransformer(config.model)
        self.normalize = config.normalize
        self.batch_size = config.batch_size

    def embed(self, texts: Sequence[str])-> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            normalize_embedding = self.normalize,
            show_progress_bar = True,
        )
        return np.ndarray(vectors, dtype=float)

