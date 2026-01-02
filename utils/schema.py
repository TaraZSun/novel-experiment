# This module defines the schema for chunker configurations using Pydantic.
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    """Schema for chunker configuration."""
    index: int | None = Field(None, description="Index of the chunk in the sequence.")
    id: int = Field(..., description="Unique identifier for the chunker configuration.")
    text: str = Field(..., description="The text content of the chunk.")
    token_count: int = Field(..., description="Number of tokens in the chunk.")
    