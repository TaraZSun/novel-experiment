from __future__ import annotations
from settings import EmbeddingConfig
from .base_embedder import BaseEmbedder
from .st_embedder import SentenceTransformerEmbedder
from chunkers.greedy_semantic_chunker import GreedySemanticChunker
from chunkers.cluster_semantic_chunker import ClusterSemanticChunker
from .openai_embedder import OpenAIEmbedder

def build_embedder(config: EmbeddingConfig)->BaseEmbedder:
    provider = config.provider

    if provider=="st":
        return SentenceTransformerEmbedder(config)
    
    if provider=="openai":
        return OpenAIEmbedder(model=config.openai_model, 
                              api_key = config.openai_api_key_env, 
                              normalize=config.normalize)
    raise ValueError(f"Unknown embedder or No embedder: {provider}")


def build_chunker(config, embedder):
    if config.chunker_type == "greedy_semantic":
        return GreedySemanticChunker(config=config, embedder=embedder, similarity_threshold=config.similarity_threshold)
    elif config.chunker_type == "cluster_semantic":
        return ClusterSemanticChunker(config=config, embedder=embedder)