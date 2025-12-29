# constants.py
from dataclasses import dataclass
@dataclass
class ChunkerConfig:
    chunk_size:int = 500
    overlap: int = 50
    encoding_name: str = "cl100k_base"
    clean_html: bool = True
    clean_markdown: bool = True
    preserve_paragraphs: bool = True