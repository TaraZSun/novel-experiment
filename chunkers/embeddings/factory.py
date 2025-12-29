from __future__ import annotations
from settings import EmbeddingConfig
import base_embedder
import st_embedder
import openai_embedder

def build_embedder(config: EmbeddingConfig)->base_embedder.BaseEmbedder:
    provider = config.provider

    if provider=="st":
        return st_embedder.SentenceTransformer(config.model, normalize=config.normalize)
    
    if provider=="openai":
        return openai_embedder.OpenAIEmbedder(model=config.openai_model, api_key = config.openai_api_key_env, normalize=config.normalize)
    raise ValueError(f"Unknown embedder or No embedder: {provider}")