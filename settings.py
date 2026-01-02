from __future__ import annotations
import consts
from dataclasses import dataclass, field

@dataclass()
class ChunkConfig:
    chunk_size:int = consts.CHUNK_SIZE
    overlap: int = consts.OVERLAP
    encoding_name: str = consts.ENCODING_NAME

@dataclass()
class Chunker:
    id: int
    text: str
    token_count: int
    
@dataclass(frozen=True)
class PreprocessConfig:
    clean_html: bool = True
    clean_markdown: bool = True
    preserve_paragraphs: bool = True

@dataclass(frozen=True)
class ChunkerConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    

@dataclass()
class EmbeddingConfig:
    local_embedding: bool=True
    provider:str = "st"
    st_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool=True
    openai_model:str = "text-embedding-3-small"
    openai_api_key_env:str = "OPENAI_API_KEY"