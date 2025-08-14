# semantic_chunker.py
from .base_chunker import BaseChunker
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticChunker(BaseChunker):
    """Chunk text based on semantic similarity between sentences."""

    def __init__(self, chunk_size: int = 500, threshold: float = 0.7):
        super().__init__(chunk_size)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    def chunk(self, text: str) -> List[Dict]:
        sentences = [s.strip() for s in text.split(". ") if s.strip()]
        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        chunks = []
        current_chunk = sentences[0]
        chunk_id = 1

        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i - 1], embeddings[i])
            if sim >= self.threshold and self.count_tokens(current_chunk + ". " + sentences[i]) <= self.chunk_size:
                current_chunk += ". " + sentences[i]
            else:
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk,
                    "token_count": self.count_tokens(current_chunk),
                    "strategy": "semantic",
                })
                chunk_id += 1
                current_chunk = sentences[i]

        if current_chunk:
            chunks.append({
                "id": chunk_id,
                "text": current_chunk,
                "token_count": self.count_tokens(current_chunk),
                "strategy": "semantic",
            })

        return chunks

