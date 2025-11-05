# chunkers/semantic_chunker.py
from constants import CHUNK_SIZE
from .base_chunker import BaseChunker
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from utils import schema, write_chunks


class SemanticChunker(BaseChunker):
    """
    Paragraph-level semantic chunker:
      1) Split by blank lines
      2) Pre-collapse consecutive very short paragraphs to reduce fragmentation
      3) Merge adjacent paragraphs based on semantic similarity + "fill-first" rule
      4) Hard-split any chunks that exceed the token limit
      5) Greedy post-packing to bring small chunks closer to chunk_size (~500)
    Output: list of utils.schema.Chunk(id, text, token_count)
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        threshold: float = 0.8,           # similarity threshold (0.75–0.85 works well)
        pack_fill: float = 0.95,          # post-pack target fill ratio (0.9–0.98)
        precollapse_min_tokens: int = 80, # minimum tokens for pre-collapse step
        joiner: str = "\n\n",             # separator between merged paragraphs
    ) -> None:
        super().__init__(chunk_size)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.pack_fill = pack_fill
        self.precollapse_min_tokens = precollapse_min_tokens
        self.joiner = joiner

    # -------------------- main entry point --------------------

    def chunk(self, text: str) -> list[schema.Chunk]:
        # 1) Split text into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return []

        # 2) Pre-collapse consecutive short paragraphs
        groups = self._precollapse_short_paragraphs(
            paragraphs,
            min_tokens=self.precollapse_min_tokens,
            joiner=self.joiner,
        )

        # 3) Compute embeddings for each group
        embeddings = self.model.encode(
            groups,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # 4) Merge based on similarity + fill-first rule
        chunks: list[schema.Chunk] = []
        current = groups[0]
        chunk_id = 1
        fill_floor = int(0.9 * self.chunk_size)  # try to fill at least 90% before flushing

        for i in range(1, len(groups)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            candidate = (current + self.joiner + groups[i]).strip()

            cand_tokens = self.count_tokens(candidate)
            cur_tokens = self.count_tokens(current)

            if cand_tokens <= self.chunk_size and (
                sim >= self.threshold or cur_tokens < fill_floor
            ):
                # merge if similar enough or current chunk not filled yet
                current = candidate
            else:
                # flush current chunk (with hard-split safeguard)
                new_chunks = self._flush_with_hardsplit(current, start_id=chunk_id)
                chunks.extend(new_chunks)
                chunk_id = chunks[-1].id + 1
                current = groups[i]

        # flush the last one
        new_chunks = self._flush_with_hardsplit(current, start_id=chunk_id)
        chunks.extend(new_chunks)

        # 5) greedy post-pack small chunks to reach ~chunk_size
        packed = self._pack_to_budget(chunks, fill_ratio=self.pack_fill, joiner=self.joiner)

        # reindex chunks to ensure consecutive IDs
        for i, ch in enumerate(packed, 1):
            ch.id = i

        return packed

    def write_chunks(self, chunks: list[schema.Chunk], output_dir: str) -> None:
        """Write chunks to individual JSON files using the utility function."""
        write_chunks.write_chunks_to_json(chunks, output_dir)

    # -------------------- internal helper functions --------------------

    def _precollapse_short_paragraphs(
        self,
        paragraphs: list[str],
        min_tokens: int = 80,
        joiner: str = "\n\n",
    ) -> list[str]:
        """
        Combine consecutive short paragraphs into larger text groups
        to reduce fragmentation.
        Rule: accumulate paragraphs until min_tokens is reached, then flush.
        """
        groups: list[str] = []
        buf = ""
        buf_tok = 0
        for p in paragraphs:
            t = self.count_tokens(p)
            if not buf:
                buf, buf_tok = p, t
                continue
            # merge until reaching min_tokens, with a soft upper limit
            if buf_tok < min_tokens:
                candidate = (buf + joiner + p).strip()
                c_tok = self.count_tokens(candidate)
                if c_tok <= 2 * min_tokens:
                    buf, buf_tok = candidate, c_tok
                    continue
            groups.append(buf)
            buf, buf_tok = p, t
        if buf:
            groups.append(buf)
        return groups

    def _flush_with_hardsplit(self, text: str, start_id: int) -> list[schema.Chunk]:
        """
        Create one or more chunks from text.
        If text exceeds chunk_size, it is split into multiple chunks by tokens.
        """
        out: list[schema.Chunk] = []
        tok = self.count_tokens(text)
        cid = start_id
        if tok <= self.chunk_size:
            out.append(schema.Chunk(id=cid, text=text, token_count=tok))
            return out

        # hard split by tokens
        toks = self.tokenizer.encode(text)
        for i in range(0, len(toks), self.chunk_size):
            part = self.tokenizer.decode(toks[i:i + self.chunk_size]).strip()
            out.append(schema.Chunk(
                id=cid,
                text=part,
                token_count=self.count_tokens(part),
            ))
            cid += 1
        return out

    def _pack_to_budget(
        self,
        chunks: list[schema.Chunk],
        fill_ratio: float = 0.95,
        joiner: str = "\n\n",
    ) -> list[schema.Chunk]:
        """
        Greedily merge adjacent small chunks until the result
        is close to the token budget (fill_ratio * chunk_size).
        """
        packed: list[schema.Chunk] = []
        buf_text, buf_tok = "", 0
        target = int(fill_ratio * self.chunk_size)

        def flush():
            nonlocal buf_text, buf_tok
            if not buf_text:
                return
            packed.append(schema.Chunk(
                id=len(packed) + 1,
                text=buf_text,
                token_count=buf_tok,
            ))
            buf_text, buf_tok = "", 0

        for ch in chunks:
            if not buf_text:
                buf_text, buf_tok = ch.text, ch.token_count
                if buf_tok >= target:
                    flush()
                continue

            candidate = (buf_text + joiner + ch.text).strip()
            cand_tok = self.count_tokens(candidate)

            if cand_tok <= self.chunk_size:
                buf_text, buf_tok = candidate, cand_tok
                if buf_tok >= target:
                    flush()
            else:
                flush()
                buf_text, buf_tok = ch.text, ch.token_count
                if buf_tok >= target:
                    flush()

        flush()
        return packed
