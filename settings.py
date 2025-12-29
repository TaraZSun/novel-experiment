from __future__ import annotations
import consts
from dataclasses import dataclass

@dataclass()
class ChunkConfig:
    chunk_size:int = consts.CHUNK_SIZE
    overlap: int = consts.OVERLAP
    encoding_name: str = consts.ENCODING_NAME
    
@dataclass(frozen=True)
class PreprocessConfig:
    clean_html: bool = True
    clean_markdown: bool = True
    preserve_paragraphs: bool = True

@dataclass(frozen=True)
class ChunkerConfig:
    chunk: ChunkConfig = ChunkConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    

@dataclass()
class EmbeddingConfig:
    local_embedding_model: bool=True
    provider:str = "st"
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool=True
    openai_model:str = "text-embedding-3-small"
    openai_api_key_env:str = "OPENAI_API_KEY"