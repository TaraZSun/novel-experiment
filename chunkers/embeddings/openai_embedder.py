from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
from settings import EmbeddingConfig
import os

class OpenAIEmbedder:
    def __init__(self, config: EmbeddingConfig, *, client=None, api_key: Optional[str]=None):
        from openai import OpenAI
        if client is not None:
            self.client = client
        else:
            if api_key is None:
                api_key = os.environ.get(config.openai_api_key_env)
            if not api_key:
                raise ValueError(f"missing api_key.{config.openai_api_key_env}")
            self.client = OpenAI(api_key=api_key)
        self.model = config.openai_model 
        self.normalize = config.normalize
        self.batch_size = max(1, int(config.batch_size))

    def embed(self, texts: Sequence[str])->np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0,0), dtype=float)

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i:i+self.batch_size]
            
            response = self.client.embeddings.create(
                input = batch,
                model=self.model
            )
            all_vectors.extend([item.embedding for item in response.data])
        vectors = np.asarray(all_vectors, dtype=float)

        if self.normalize and vectors.szie>0:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
         
        return vectors / np.where(norms==0,1.0, norms)
