import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def _encode(self, texts):
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=len(texts) > 50
        ).astype("float32")

    def build_index(self, chunks):
        self.chunks = list(chunks)
        embeddings = self._encode([c["text"] for c in self.chunks])
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine (vectors are normalized)
        self.index.add(embeddings)

    def add_chunks(self, new_chunks):
        """Update the RAG without rebuilding: embed only the new chunks."""
        if not new_chunks:
            return
        if self.index is None:
            return self.build_index(new_chunks)
        self.index.add(self._encode([c["text"] for c in new_chunks]))
        self.chunks.extend(new_chunks)

    def sources(self):
        return sorted({c["source"] for c in self.chunks})

    def search(self, query, top_k=5, sources=None, max_per_source=None):
        """sources: restrict to these papers. max_per_source: diversify across papers."""
        n = len(self.chunks)
        pool = n if (sources or max_per_source) else min(top_k, n)
        scores, indices = self.index.search(self._encode([query]), pool)
        wanted = {s.lower() for s in sources} if sources else None
        per_source, results = {}, []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            src = chunk["source"]
            if wanted and src.lower() not in wanted:
                continue
            if max_per_source and per_source.get(src, 0) >= max_per_source:
                continue
            per_source[src] = per_source.get(src, 0) + 1
            r = chunk.copy()
            r["score"] = float(score)
            results.append(r)
            if len(results) >= top_k:
                break
        return results

    def save(self, path="data/index"):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path="data/index"):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", "rb") as f:
            self.chunks = pickle.load(f)