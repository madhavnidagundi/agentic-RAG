from typing import List, Dict, Optional, Any, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys
from rank_bm25 import BM25Okapi

class RetrieverAgent:
    # ... existing init ...
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", embed_batch_size: int = 64):
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self.model = SentenceTransformer(self.model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.bm25: Optional[BM25Okapi] = None
        self.id_to_doc: Dict[int, Dict[str, Any]] = {}
        self.dim = self.model.get_sentence_embedding_dimension()
        self._next_id = 0
        self._emb_store = None
        self._documents: List[Dict[str, Any]] = []

    def _normalize_embeddings(self, embs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return embs / norms

    # ... existing helper methods ...

    def _normalize_score(self, raw_score: float) -> float:
        s = (raw_score + 1.0) / 2.0
        return float(max(0.0, min(1.0, s)))

    def _apply_metadata_filters(self, doc: Dict[str, Any], metadata_filter: Optional[Dict[str, Any]],
                                metadata_substr: Optional[tuple], metadata_predicate: Optional[Callable[[Dict[str,Any]],bool]]) -> bool:
        md = doc.get("metadata", {}) or {}
        if metadata_filter:
            for k, v in metadata_filter.items():
                val = md.get(k)
                if isinstance(v, list):
                    if val not in v:
                        return False
                else:
                    if val != v:
                        return False
        if metadata_substr:
            key, substr = metadata_substr
            val = str(md.get(key, "")).lower()
            if substr.lower() not in val:
                return False
        if metadata_predicate:
            try:
                if not metadata_predicate(md):
                    return False
            except Exception as e:
                print(f"[Retriever] metadata_predicate error: {e}", file=sys.stderr)
                return False
        return True

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenizer for BM25
        return text.lower().split()

    def build_index(self, docs: List[Dict[str, Any]]):
        if not docs:
            raise ValueError("Cannot build an index from an empty document collection.")
        print(f"[Retriever] Building Hybrid Index for {len(docs)} docs...")
        
        # 1. FAISS (Dense)
        texts = [d["text"] for d in docs]
        embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch_texts = texts[i:i+self.embed_batch_size]
            emb = self.model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
            embeddings.append(emb)
        if embeddings:
            embeddings = np.vstack(embeddings).astype("float32")
        else:
            embeddings = np.zeros((0, self.dim), dtype="float32")
        embeddings = self._normalize_embeddings(embeddings)

        self.index = faiss.IndexFlatIP(self.dim)
        if embeddings.shape[0] > 0:
            self.index.add(embeddings)
            self._emb_store = embeddings.copy()

        # 2. BM25 (Sparse)
        tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.id_to_doc = {}
        for i, d in enumerate(docs):
            self.id_to_doc[i] = d
        self._documents = list(docs)
        self._next_id = len(docs)
        print(f"[Retriever] Hybrid Index built. Total docs: {self.index.ntotal}")

    def retrieve(self, query: str, k: int = 10, alpha: float = 0.3,
                 metadata_filter: Optional[Dict[str, Any]] = None,
                 metadata_substr: Optional[tuple] = None,
                 metadata_predicate: Optional[Callable[[Dict[str,Any]],bool]] = None) -> Dict[str, Any]:
        """
        alpha: weight for Dense score (0.0 to 1.0). 1.0 = Pure Vector, 0.0 = Pure BM25.
        Recommended: 0.5 for balanced hybrid.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Index is empty; run build_index(docs) first.")
        if k <= 0:
            raise ValueError("k must be greater than zero.")
        k = min(k, self.index.ntotal)
        candidate_k = min(self.index.ntotal, max(k * 3, k))

        # 1. Dense Search
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        q_emb = self._normalize_embeddings(q_emb)
        D, I = self.index.search(q_emb, candidate_k)
        dense_hits = {idx: score for idx, score in zip(I[0], D[0]) if idx >= 0}

        # 2. Sparse Search (BM25)
        tokenized_query = self._tokenize(query)
        # BM25 returns scores for all docs, we need top K
        # Optimization: getting all scores might be slow for huge datasets, but okay for <50k
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Normalize BM25 scores (min-max usually, but simpler approach: divide by max)
        if len(bm25_scores) > 0 and max(bm25_scores) > 0:
             bm25_scores = bm25_scores / max(bm25_scores)
        
        top_bm25_idxs = np.argsort(bm25_scores)[::-1][:candidate_k]
        sparse_hits = {idx: bm25_scores[idx] for idx in top_bm25_idxs}

        # 3. Fusion (Weighted Sum)
        all_indices = set(dense_hits.keys()) | set(sparse_hits.keys())
        merged_scores = []
        
        for idx in all_indices:
            s_dense = dense_hits.get(idx, 0.0)
            # Normalize dense score if needed (it's cosine similarity -1 to 1, usually 0 to 1 for text)
            s_dense = (s_dense + 1) / 2 # Ensure 0-1 range roughly
            
            s_sparse = sparse_hits.get(idx, 0.0)
            
            final_score = alpha * s_dense + (1 - alpha) * s_sparse
            merged_scores.append((idx, final_score, s_dense, s_sparse))
            
        merged_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = merged_scores[:k]

        chunks = []
        for idx, score, s_d, s_s in top_k:
            doc = self.id_to_doc.get(int(idx))
            if not doc: continue
            if not self._apply_metadata_filters(doc, metadata_filter, metadata_substr, metadata_predicate):
                continue
            
            cpy_meta = dict(doc.get("metadata", {}))
            cpy_meta["_dense_score"] = float(s_d)
            cpy_meta["_bm25_score"] = float(s_s)
            
            chunks.append({
                "doc_id": doc.get("doc_id"),
                "orig_id": doc.get("orig_id"),
                "text": doc.get("text"),
                "metadata": cpy_meta,
                "score": float(score)
            })

        # ... (rest of stats logic same as before) ...
        per_orig = {}
        for c in chunks:
            oid = c["orig_id"]
            if oid not in per_orig:
                per_orig[oid] = {"orig_id": oid, "scores": [], "chunks": []}
            per_orig[oid]["scores"].append(c["score"])
            per_orig[oid]["chunks"].append(c)
            
        # ... logic for max/avg score ...
        for oid, info in per_orig.items():
            info["max_score"] = float(max(info["scores"]))
            info["avg_score"] = float(sum(info["scores"]) / len(info["scores"]))
            top_chunk = max(info["chunks"], key=lambda x: x["score"])
            info["top_chunk"] = {
                "doc_id": top_chunk["doc_id"],
                "text_snippet": top_chunk["text"][:400],
                "score": top_chunk["score"],
                "metadata": top_chunk["metadata"]
            }

        stats = {
            "k_requested": k,
            "k_returned": len(chunks),
            "max_score": float(max([c["score"] for c in chunks]) if chunks else 0.0),
        }
        return {"chunks": chunks, "per_orig": per_orig, "stats": stats}

    def add_docs(self, new_docs: List[Dict[str, Any]]):
        if not new_docs:
            return

        texts = [d["text"] for d in new_docs]
        embs_batches = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch_texts = texts[i:i+self.embed_batch_size]
            emb = self.model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
            embs_batches.append(emb)
        embs = np.vstack(embs_batches).astype("float32")
        embs = self._normalize_embeddings(embs)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embs)
            start_id = 0
            self._emb_store = embs.copy()
        else:
            start_id = self.index.ntotal
            self.index.add(embs)
            if self._emb_store is None:
                self._emb_store = embs.copy()
            else:
                self._emb_store = np.vstack([self._emb_store, embs])

        for j, d in enumerate(new_docs):
            self.id_to_doc[start_id + j] = d
        self._documents.extend(new_docs)
        self.bm25 = BM25Okapi([self._tokenize(doc["text"]) for doc in self._documents])
        self._next_id = self.index.ntotal
        print(f"[Retriever] Appended {len(new_docs)} docs. New index size: {self.index.ntotal}")
