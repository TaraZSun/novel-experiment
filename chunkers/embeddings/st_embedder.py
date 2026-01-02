from __future__ import annotations
from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer
from settings import EmbeddingConfig

class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig):
        self.model = SentenceTransformer(config.st_model)
        self.normalize = config.normalize

    def embed(self, texts: Sequence[str])-> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings = self.normalize,
            show_progress_bar = True,
        )
        return np.asarray(vectors, dtype=float)

