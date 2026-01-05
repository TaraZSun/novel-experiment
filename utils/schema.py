# This module defines the schema for chunker configurations using Pydantic.
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    """Schema for chunker configuration."""
    index: int | None = Field(None, description="Index of the chunk in the sequence.")
    id: int = Field(..., description="Unique identifier for the chunker configuration.")
    text: str = Field(..., description="The text content of the chunk.")
    token_count: int = Field(..., description="Number of tokens in the chunk.")
    
class ChunkSource(BaseModel):
    """Schema for chunk source configuration."""
    doc_id: int = Field(..., description="Unique identifier for the chunk source.")
    path: str = Field(..., description="File path of the chunk source.")
    sections: list[str] = Field(..., description="List of sections in the chunk source.")
    page: int | None = Field(None, description="Page number if applicable.")

class IndexChunker(BaseModel):
    """Schema for index chunker configuration."""
    chunk_id: int = Field(..., description="Unique identifier for the index chunker.")
    text: str = Field(..., description="The text content of the index chunker.")
    token_count: int = Field(..., description="Number of tokens in the index chunker.")
    source: ChunkSource = Field(..., description="Source information for the chunker.")
    start_char: int = Field(..., description="Starting character index of the chunk.")
    end_char: int = Field(..., description="Ending character index of the chunk.")
    embedding: list[float] = Field(..., description="Embedding vector for the chunk.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata for the chunker.")