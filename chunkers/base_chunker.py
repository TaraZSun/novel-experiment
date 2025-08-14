import abc
import tiktoken
from typing import List, Dict


class BaseChunker(abc.ABC):
    """Abstract base class for chunkers."""

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @abc.abstractmethod
    def chunk(self, text: str) -> List[Dict]:
        pass
