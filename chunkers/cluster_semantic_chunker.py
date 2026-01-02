from .base_chunker import BaseChunker
import numpy as np
import re
from utils import schema, write_chunks
from chunkers.embeddings.base_embedder import BaseEmbedder
from settings import ChunkerConfig
import consts 
from pathlib import Path

segment_size = consts.SEGMENT_SIZE
joiner = consts.JOINER
chunk_size = consts.CHUNK_SIZE

class ClusterSemanticChunker(BaseChunker):
    """
    Cluster-based semantic chunker using dynamic programming to 
    optimize chunk boundaries based on semantic similarity from global optimization.

    Based on Chroma's ClusterSemanticChunker appoach:
    1. Split text into initial small chunks.(about 50 tokens)
    2. Embed the chunks using a provided embedder.
    3. Use dynamic programming to find optimal chunk boundaries that maximize semantic coherence.
    """

    def __init__(self, *, 
                config: ChunkerConfig, 
                embedder: BaseEmbedder,
                segment_size: int = segment_size,
                joiner: str = joiner,
                ) -> None:
        super().__init__(config=config)
        self.embedder = embedder
        self.segment_size = segment_size
        self.joiner = joiner

    def chunk(self, text: str) -> list[schema.Chunk]:
        """
        Main chunking pipeline using dynamic programming for optimal chunking.
        stage 1: split into fine-grained segments
        """
        segments = self._split_into_segments(text)
        if not segments:
            return []
        embeddings = self._compute_embeddings(segments)
        chunk_boundaries = self._find_optimal_chunks(segments, embeddings)
        chunks = self._create_chunks_from_boundaries(segments, chunk_boundaries)
        self._reindex_chunks(chunks)
        return chunks

    # Stage 1: Split text into fine-grained segments
    def _split_into_segments(self, text: str) -> list[str]:
        """
        Split text into segments of approximately segment_size tokens.
        Using recursive splitting on sentence boundaries for better coherence.
        """
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        segments = []
        for para in paragraphs:
            sentences = self._split_sentences(para)
            current_segment = ""
            current_tokens = 0
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)

                # If adding this sentence exceeds segment size, start a new segment
                if current_tokens>0 and current_tokens + sentence_tokens > self.segment_size:
                    segments.append(current_segment.strip())
                    current_segment = sentence
                    current_tokens = sentence_tokens
                else:
                    current_segment = f"{current_segment} {sentence}".strip()
                    current_tokens += sentence_tokens
            # Add any remaining segment
            if current_segment:
                segments.append(current_segment.strip())
        return segments
    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.
        """
        # Split based on sentence boundaries: .,!,?
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    # Stage 2: Compute embeddings for segments
    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Compute embeddings for each segment using the provided embedder.
        """
        return self.embedder.embed(texts)
    
    # Stage 3: Dynamic programming to find optimal chunk boundaries
    def _find_optimal_chunks(
            self,
            segments: list[str],
            embeddings: np.ndarray
    )->list[int]:
        """
        Use dynamic programming to find optimal chunk boundaries.

        Goal: maximize semantic coherence within chunks.
        Constraints: each chunk <= chunk_size tokens.

        returns: list of chunk end indices(exclusive)
        Example: [3,7,10] means chunks are segments[0:3], segments[3:7], segments[7:10]
        """
        n = len(segments)
        segment_tokens = [self.count_tokens(seg) for seg in segments]
        cumulative_sum_tokens = [0]
        for tokens in segment_tokens:
            cumulative_sum_tokens.append(cumulative_sum_tokens[-1] + tokens)
        
        dp = [(-float('inf'), -1) for _ in range(n+1)]
        dp[0] = (0.0, 0)

        for end in range(1, n+1):
            for start in range(end):
                chunk_token_count = cumulative_sum_tokens[end] - cumulative_sum_tokens[start]
                if chunk_token_count > chunk_size:
                    continue

                chunk_similarity = self._calculate_chunk_similarity(embeddings[start:end])
                total_similarity = dp[start][0] + chunk_similarity
                if total_similarity > dp[end][0]:
                    dp[end] = (total_similarity, start)

        # Backtrack to find optimal chunk boundaries
        boundaries = []
        current = n
        while current > 0:
            start = dp[current][1]
            boundaries.append(current)
            current = start

        boundaries.reverse()
        return boundaries
    def _calculate_chunk_similarity(self, chunk_embeddings: np.ndarray) -> float:
        """
        Calculate average pairwise cosine similarity within a chunk.
        For embedding already normalized, cosine similarity = dot product.
        """
        if len(chunk_embeddings) <= 1:
            return 0.0
        sim_matrix = np.dot(chunk_embeddings, chunk_embeddings.T)
        n=len(chunk_embeddings)
        total_similarity = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_similarity += sim_matrix[i][j]
                count += 1
        return total_similarity / count if count > 0 else 0.0
    

    # Stage 4: Create chunks from boundaries
    def _create_chunks_from_boundaries(
            self,
            segments: list[str],
            boundaries: list[int]
    ) -> list[schema.Chunk]:
        """
        Create Chunk objects from segments and chunk boundaries.
        """
        chunks = []
        start_idx = 0
        for end_idx in boundaries:
            chunk_text = self.joiner.join(segments[start_idx:end_idx])
            chunk_tokens = self.count_tokens(chunk_text)
            chunks.append(schema.Chunk(
                id=len(chunks)+1,
                text=chunk_text,
                token_count=chunk_tokens,
            ))
           
            start_idx = end_idx
        return chunks
    
    # Reindex chunks
    def _reindex_chunks(self, chunks: list[schema.Chunk]) -> None:
        """
        Reindex chunks to ensure sequential IDs.
        """
        for idx, chunk in enumerate(chunks):
            chunk.id = idx + 1

    def write_chunks(self, chunks: list[schema.Chunk], output_dir:Path) -> None:
        """
        Write chunks to a file using utility function.
        """
        write_chunks.write_chunks_to_json(chunks, output_dir)
       