# chunkers/paragraph_chunker.py
import re
from typing import List, Dict
from .base_chunker import BaseChunker

class ParagraphChunker(BaseChunker):
    """
    Chunk by paragraphs up to ~size tokens.
    - ignores empty/whitespace paragraphs
    - merges multiple short paragraphs until the token budget is reached
    - if a single paragraph is longer than size, it falls back to sentence splits
    """

    def __init__(self, chunk_size: int = 500, min_tokens: int = 1, joiner: str = "\n\n"):
        super().__init__(chunk_size)
        self.min_tokens = max(0, min_tokens)
        self.joiner = joiner

    def _split_paragraphs(self, text: str) -> List[str]:
        # normalize and drop leading BOM/newlines
        text = text.lstrip("\ufeff\r\n")
        # split on 1+ blank lines
        paras = re.split(r"\n\s*\n", text)
        # strip and keep non-empty
        return [p.strip() for p in paras if p.strip()]

    def _split_sentences(self, paragraph: str) -> List[str]:
        # simple sentence splitter; good enough as a fallback
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]

    def chunk(self, text: str) -> List[Dict]:
        paragraphs = self._split_paragraphs(text)

        chunks: List[Dict] = []
        buf: List[str] = []
        buf_tokens = 0
        cid = 1

        def flush():
            nonlocal buf, buf_tokens, cid
            if not buf:
                return
            chunk_text = self.joiner.join(buf).strip()
            tok = self.count_tokens(chunk_text)
            if tok >= self.min_tokens:
                chunks.append({
                    "id": cid,
                    "text": chunk_text,
                    "token_count": tok,
                    "strategy": "paragraph",
                })
                cid += 1
            buf = []
            buf_tokens = 0

        for p in paragraphs:
            t = self.count_tokens(p)

            # if one paragraph is too large, fall back to sentence-level packing
            if t > self.chunk_size:
                # flush what we have so far
                flush()
                for s in self._split_sentences(p):
                    st = self.count_tokens(s)
                    if st > self.chunk_size:
                        # extremely long sentence: hard split by tokens
                        toks = self.tokenizer.encode(s)
                        for i in range(0, len(toks), self.chunk_size):
                            part = self.tokenizer.decode(toks[i:i+self.chunk_size])
                            chunks.append({
                                "id": cid,
                                "text": part,
                                "token_count": self.count_tokens(part),
                                "strategy": "paragraph_fallback_sentence",
                            })
                            cid += 1
                        continue

                    if buf_tokens + st <= self.chunk_size and buf:
                        buf.append(s)
                        buf_tokens += st
                    else:
                        flush()
                        buf = [s]
                        buf_tokens = st
                continue  # next paragraph

            # normal paragraph packing
            if buf_tokens + t <= self.chunk_size and buf:
                buf.append(p)
                buf_tokens += t
            else:
                flush()
                buf = [p]
                buf_tokens = t

        flush()
        return chunks
