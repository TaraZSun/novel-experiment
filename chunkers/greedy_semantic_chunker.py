import numpy as np
from utils import schema, write_chunks
from chunkers import base_chunker
from settings import ChunkerConfig
import consts 
from pathlib import Path
chunk_size = consts.CHUNK_SIZE

class GreedySemanticChunker(base_chunker.BaseChunker):
    def __init__(
            self,
            *,
            config: ChunkerConfig,
            embedder: base_chunker.BaseChunker,
            similarity_threshold: float,
            segment_size: int= consts.SEGMENT_SIZE,
            joiner: str = consts.JOINER,
            precollapse_min_tokens: int = consts.PRECOLLAPSE_MIN_TOKRNS,
            fill_floor: float=consts.FILL_FLOOR,
    )->None:
        super().__init__(config=config)
        self.embedder = embedder
        self.segment_size = segment_size
        self.joiner = joiner
        self.similarity_threshold = similarity_threshold
        self.fill_floor = fill_floor
        self.precollapse_min_tokens = precollapse_min_tokens

    def chunk(self, text: str) -> list[schema.Chunk]:
        """
        Greedy semantic chunking based on local similarity thresholds.
        """
        paragraphs = self._preprocess_text(text)
        if not paragraphs:
            return []
        
        groups = self._group_small_paragraphs(paragraphs)
        embeddings = self.embedder.embed(groups)
        chunks = self._greedy_merge(groups, embeddings)

        for i, chunk in enumerate(chunks, start=1):
            chunk.index = i
        return chunks
    

    def _group_small_paragraphs(self, paragraphs: list[list[str]]) -> list[str]:
        """
        combine small paragraphs into larger groups based on fill_floor
        """
        grouped = []
        buffer = ""
        buffer_tokens = 0


        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if not buffer:
                buffer = para
                buffer_tokens = para_tokens
                continue
            if buffer_tokens < self.precollapse_min_tokens:
                candidate = f"{buffer}{self.joiner}{para}"
                candidate_tokens = self.count_tokens(candidate)

                if candidate_tokens <= 2*self.precollapse_min_tokens:
                    buffer = candidate
                    buffer_tokens = candidate_tokens
                    continue
            grouped.append(buffer)
            buffer = para
            buffer_tokens = para_tokens
        if buffer:
            grouped.append(buffer)
        return grouped
    def _greedy_merge(self, groups: list[str], embeddings: np.ndarray) -> list[schema.Chunk]:
        """
        Greedyly merge groups based on similarity threshold.
        """
        if not groups:
            return []
        chunks = []
        current_chunk = groups[0]
        for i in range(1, len(groups)):
            similarity = float(np.dot(embeddings[i-1], embeddings[i]))
            candidate = f"{current_chunk}{self.joiner}{groups[i]}"
            candidate_tokens = self.count_tokens(candidate)
            current_tokens = self.count_tokens(current_chunk)

            should_merge = (
                candidate_tokens <= chunk_size and
                (
                    similarity >= self.similarity_threshold or
                    current_tokens < self.fill_floor
                )
            )  
            if should_merge:
                current_chunk = candidate
            else:
                chunks.append(schema.Chunk(
                    id=len(chunks)+1,
                    text=current_chunk,
                    token_count=self.count_tokens(current_chunk),
                ))
                current_chunk = groups[i]

            # Handle last chunk
        if current_chunk:
            chunks.append(schema.Chunk(
                id=len(chunks)+1,
                text=current_chunk,
                token_count=self.count_tokens(current_chunk),
            ))
        return chunks
    def write_chunks(
            self,
            chunks: list[schema.Chunk],
            output_dir: Path,
    ) -> None:
        write_chunks.write_chunks_to_json(chunks, output_dir)