# utils/retriever.py
import json
from pathlib import Path

import numpy 
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


class Retriever:
    def __init__(
        self,
        chunks_dir: str,
        embed_model_name: str = "all-MiniLM-L6-v2",
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )-> None:
        self.chunks_dir = Path(chunks_dir)
        self.index = faiss.read_index(str(self.chunks_dir / "index.faiss"))
        self.meta = json.loads((self.chunks_dir / "meta.json").read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer(embed_model_name)
        self.reranker = CrossEncoder(rerank_model_name)

    def _load_text(self, json_path: str) -> str:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return (data.get("text") or "").strip()

    def search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 50,
        preview_chars: int = 220,
    ) -> list[dict]:
        """
       
        """
        
        q_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q_vec, fetch_k)
        cand_ids = ids[0]
        ann_scores = scores[0]

        candidates = []
        pairs = []  # for cross-encoder
        for idx, ann_score in zip(cand_ids, ann_scores):
            m = self.meta[int(idx)]
            text = self._load_text(m["path"])
            if not text:
                continue
            candidates.append(
                {
                    "id": m.get("id"),
                    "path": m["path"],
                    "token_count": m.get("token_count"),
                    "text": text,
                    "ann_score": float(ann_score),
                }
            )
            pairs.append((query, text))

        if not candidates:
            return []

       
        rerank_scores: list[float] = self.reranker.predict(pairs).tolist()
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: (x["rerank_score"], x["ann_score"]), reverse=True)
        results = []
        for c in candidates[:k]:
            preview = c["text"][:preview_chars].replace("\n", " ")
            results.append(
                {
                    "id": c["id"],
                    "path": c["path"],
                    "token_count": c["token_count"],
                    "ann_score": round(c["ann_score"], 3),
                    "rerank_score": round(c["rerank_score"], 3),
                    "preview": preview + ("..." if len(c["text"]) > preview_chars else ""),
                    "text": c["text"],  
                }
            )
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FAISS retrieval + CrossEncoder rerank")
    parser.add_argument("--chunks_dir", required=True, help="e.g., chunks/semantic")
    parser.add_argument("--query", required=True, help="your search query")
    parser.add_argument("--k", type=int, default=5, help="final results to return")
    parser.add_argument("--fetch_k", type=int, default=50, help="FAISS candidates before rerank")
    args = parser.parse_args()

    r = Retriever(args.chunks_dir)
    hits = r.search(args.query, k=args.k, fetch_k=args.fetch_k)

    print(f"\nTop {len(hits)} results for: {args.query!r}\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. id={h['id']}  rerank={h['rerank_score']:.3f}  ann={h['ann_score']:.3f}")
        print(f"   {h['path']}")
        print(f"   {h['preview']}\n")
