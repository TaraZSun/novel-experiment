import abc
import tiktoken
from typing import List, Dict
from bs4 import BeautifulSoup
import re
import logging
from constants import (
    MARKDOWN_LINK_REGEX,
    MARKDOWN_BOLD_REGEX,
    MARKDOWN_ITALIC_REGEX,
    MARKDOWN_CODE_REGEX,
    MARKDOWN_HEADER_REGEX
)
logging.basicConfig(level=logging.INFO)


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

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the text by cleaning HTML/Markdown and normalizing whitespace.
        """
        # 1. Remove HTML tags with BeautifulSoup
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except Exception:
            # If BeautifulSoup fails, we can choose to log the error or handle it as needed.
            logging.warning("BeautifulSoup failed to parse HTML. Proceeding without HTML cleaning.")

        # 2. Remove Markdown syntax using regex
        MARKDOWN_REGEXES = [
            (MARKDOWN_LINK_REGEX,r'\1'),  # Keep the link text, remove the URL
            (MARKDOWN_BOLD_REGEX,r'\2'),  # Keep the bold text, remove the asterisks/underscores
            (MARKDOWN_ITALIC_REGEX,r'\2'),  # Keep the italic text, remove the asterisks/underscores
            (MARKDOWN_CODE_REGEX,r'\1'),  # Keep the code text, remove the backticks
            (MARKDOWN_HEADER_REGEX,r'\1'),  # Keep the header text, remove the hashes and leading spaces
        ]

        for pattern, repl in MARKDOWN_REGEXES:
            text = re.sub(pattern, repl, text)

        # 3. strip and normalize whitespace
        text = " ".join(text.split())

        return text
        

