from .base_chunker import BaseChunker
from typing import List, Dict
from constants import OVERLAP, CHUNK_SIZE, BATCH_SIZE

class FixedSizeChunker(BaseChunker):
    """Chunk text into fixed-size token windows (with optional overlap)."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, batch_size: int = BATCH_SIZE)-> None:
        super().__init__(chunk_size)
        assert 0 <= overlap < chunk_size, "Overlap must be non-negative and less than chunk size."
        self.overlap = overlap
        self.batch_size = batch_size

    def chunk(self, text: str) -> List[Dict]:
        text = self._preprocess_text(text)
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0
        chunk_id = 1

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = " ".join(self.tokenizer.decode(tokens[start:end]).split())
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "token_count": self.count_tokens(chunk_text),
            })
            chunk_id += 1
            start += self.chunk_size - self.overlap

        return chunks

    def chunk_batches(self, text: str, batch_size: int = BATCH_SIZE) -> List[List[Dict]]:
        """Return a list of chunk batches."""
        all_chunks = self.chunk(text)
        return [all_chunks[i:i + batch_size] for i in range(0, len(all_chunks), batch_size)]

    def chunk_batches_generator(self, text: str, batch_size: int = BATCH_SIZE)-> List[Dict]:
        """Yield batches one by one (memory-efficient for large text)."""
        all_chunks = self.chunk(text)
        for i in range(0, len(all_chunks), batch_size):
            yield all_chunks[i:i + batch_size]
