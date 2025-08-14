from .base_chunker import BaseChunker
from typing import List, Dict


class FixedSizeChunker(BaseChunker):
    """Chunk text into fixed-size token windows (with optional overlap)."""

    def __init__(self, chunk_size: int = 500, overlap: int = 0):
        super().__init__(chunk_size)
        self.overlap = overlap

    def chunk(self, text: str) -> List[Dict]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        chunk_id = 1

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.tokenizer.decode(tokens[start:end])
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "token_count": self.count_tokens(chunk_text),
                "strategy": "fixed_size",
            })
            chunk_id += 1
            start += self.chunk_size - self.overlap

        return chunks

