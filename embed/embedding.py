# utils/build_embeddings.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chunks(chunks_dir: str):
    """Load all chunk JSONs from a folder (ignore manifest files)."""
    dir_path = Path(chunks_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    texts = []
    metas = []

    for file in sorted(dir_path.glob("*.json")):
        if file.name.startswith("_"):  # skip manifest or hidden files
            continue
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data.get("text", "").strip()
            if not text:
                continue
            texts.append(text)
            metas.append({
                "id": data.get("id"),
                "path": str(file),
                "token_count": data.get("token_count"),
            })

    logger.info(f"Loaded {len(texts)} chunks from {dir_path}")
    return texts, metas


def build_faiss_index(chunks_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    """Compute embeddings for all chunks and save FAISS index + metadata."""
    texts, metas = load_chunks(chunks_dir)
    model = SentenceTransformer(model_name)
    logger.info(f"Embedding {len(texts)} chunks with model: {model_name}")

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    # Create FAISS index (cosine similarity = inner product with normalized vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save
    out_dir = Path(chunks_dir)
    faiss_path = out_dir / "index.faiss"
    meta_path = out_dir / "meta.json"

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved FAISS index to {faiss_path}")
    logger.info(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS embedding index from chunk JSON files.")
    parser.add_argument("--chunks_dir", type=str, required=True, help="Path to chunk folder (e.g., chunks/semantic)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    args = parser.parse_args()

    build_faiss_index(args.chunks_dir, args.model)
