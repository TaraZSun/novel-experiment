from __future__ import annotations
from typing import Sequence
import numpy as np
from settings import EmbeddingConfig


class OpenAIEmbedder:
    def __init__(self, client, config: EmbeddingConfig):
        self.client = client
        self.model = config.openai_model 
        self.normalize = config.normalize
        self.batch_size = config.batch_size

    def embed(self, texts: Sequence[str])->np.ndarray:
        response = self.client.embeddings.create(
            input = list(texts),
            model=self.model
        )
        data = [item.embedding for item in response.data]
        vectors = np.asarray(data, dtype=float)
        if self.normalize and vectors.szie>0:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
         
        return vectors / np.where(norms==0,1.0, norms)
