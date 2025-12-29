from .base_chunker import BaseChunker
from constants import CHUNK_SIZE
from utils import schema, write_chunks

class FixedSizeChunker(BaseChunker):
    """Chunk text into fixed-size token windows."""

    def __init__(self, chunk_size: int = CHUNK_SIZE)-> None:
        super().__init__(chunk_size)
       

    def chunk(self, text: str) -> list[schema.Chunk]:
        text = self._preprocess_text(text)
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0
        chunk_id = 1

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = " ".join(self.tokenizer.decode(tokens[start:end]).split())
            chunks.append(
                schema.Chunk(
                id=chunk_id,
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                )
                )
            chunk_id += 1
            start += self.chunk_size

        return chunks

    def write_chunks(self, chunks: list[schema.Chunk], output_dir: str) -> None:
        """Write chunks to JSON files using the utility function."""
        write_chunks.write_chunks_to_json(chunks, output_dir)
