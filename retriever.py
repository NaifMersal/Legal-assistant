import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from typing import Any, Dict, List, Optional, Tuple



class Retriever:
    """Handles loading retrieval models and performing search."""

    def __init__(self, faiss_index_path: str, documents_path: str,
                 embeddings_model: Any, metric_type: str = 'ip'):
        """
        Initializes the Retriever.

        Args:
            faiss_index_path: Path to the FAISS index file.
            documents_path: Path to the JSON file containing documents and metadata.
            embeddings_model: An embeddings model with an .encode() method.
            metric_type: The metric type used for FAISS ('ip' or 'l2').
        """
        print("Initializing Retriever...")
        self.index = faiss.read_index(faiss_index_path)
        self.embeddings_model = embeddings_model
        self.metric_type = metric_type.lower()

        if self.metric_type not in ['ip', 'l2']:
            raise ValueError("metric_type must be either 'ip' or 'l2'")

        print("Loading documents and metadata...")

        self.doc_metadata = {}
        with open(documents_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
            self._extract_documents_with_metadata(docs_data)

            
        print(f"✓ Loaded {len(self.doc_metadata)} documents with metadata")

        print("Initializing BM25...")
        # We must sort doc_ids to ensure a consistent mapping
        # between 0-based indices (used by BM25/FAISS) and external article_ids.
        self.doc_ids = sorted(self.doc_metadata.keys())
        self.corpus = [self.doc_metadata[doc_id]['text'] for doc_id in self.doc_ids]
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("✓ BM25 initialized successfully!")

    def _extract_documents_with_metadata(self, docs_data: Dict):
        """Extract documents and metadata together, storing both in optimized structures."""
        for category, subcats in docs_data.items():
            if not isinstance(subcats, dict): continue
            for subcat, systems in subcats.items():
                if not isinstance(systems, dict): continue
                for system_name, system_data in systems.items():
                    if not isinstance(system_data, dict): continue

                    system_brief = system_data.get('brief', '')
                    system_metadata = system_data.get('metadata', {})
                    parts = system_data.get('parts', {})

                    for part_name, articles in parts.items():
                        if isinstance(articles, list):
                            for article in articles:
                                if isinstance(article, dict):
                                    article_id = article.get('id')
                                    if article_id is not None:
                                        text = f"{article.get('Article_Title', '')}\n{article.get('Article_Text', '')}".strip()
                                        if text:
                                            self.doc_metadata[article_id] = {
                                                'text': text,
                                                'id': article_id,
                                                'title': article.get('Article_Title', ''),
                                                'category': category,
                                                'subcategory': subcat,
                                                'system': system_name,
                                                'system_brief': system_brief,
                                                'system_metadata': system_metadata,
                                                'part': part_name
                                            }

    def _index_to_doc_id(self, indices: List[int]) -> List[int]:
        """Maps 0-based corpus indices to their original document IDs."""
        return [self.doc_ids[i] for i in indices]

    def get_article_metadata(self, doc_id: int) -> Dict:
        """Get cached article metadata in O(1) time."""
        return self.doc_metadata.get(doc_id, None)

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        embedding = np.array(self.embeddings_model.encode([query])["dense_vecs"], dtype='float32')
        if self.metric_type == 'ip':
            # Normalize for Inner Product (Cosine Similarity)
            faiss.normalize_L2(embedding)
        return embedding

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0,1] range."""
        if len(scores) == 0:
            return scores
        min_s, max_s = np.min(scores), np.max(scores)
        return np.ones_like(scores) if max_s == min_s else (scores - min_s) / (max_s - min_s)

    def dense(self, query: str, k: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Dense retrieval using FAISS. Returns (scores, doc_ids)."""
        query_emb = self._embed_query(query)
        distances, indices = self.index.search(query_emb, k)

        # Convert distances to similarity scores [0, 1]
        if self.metric_type == 'l2':
            scores = 1.0 / (1.0 + distances[0])
        else: # 'ip'
            scores = (distances[0] + 1) / 2 # Assuming normalized vectors, IP is in [-1, 1]

        doc_ids = self._index_to_doc_id(indices[0])
        return self._normalize(scores), doc_ids

    def sparse(self, query: str, k: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Sparse retrieval using BM25. Returns (scores, doc_ids)."""
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices (relative to corpus)
        top_k_indices = np.argsort(bm25_scores)[::-1][:k]
        
        scores = bm25_scores[top_k_indices]
        doc_ids = self._index_to_doc_id(top_k_indices)
        
        return self._normalize(scores), doc_ids

    def hybrid(self, query: str, k: int = 10, dense_weight: float = 0.7, sparse_weight: float = 0.3) -> Tuple[np.ndarray, List[int]]:
        """
        Hybrid retrieval that reuses the existing dense() and sparse() helpers.
        Strategy:
        - Call self.dense(query, k) to get top-k dense candidates (doc ids and dense scores).
        - Call self.sparse(query, k=len(corpus)) to get sparse scores across the corpus.
        - Combine dense and sparse scores for the dense candidate set using provided weights.

        Returns normalized hybrid scores and the corresponding original doc IDs.
        """
        # 1) Get dense results (scores already normalized by dense())
        dense_scores, dense_doc_ids = self.dense(query, k=k)

        # 2) Get sparse scores for the whole corpus (normalized)
        # Note: sparse() expects a k but we only need the score vector; call with k=len(self.corpus)
        full_sparse_scores, _ = self.sparse(query, k=len(self.corpus))

        # Build a mapping from doc_id -> sparse_score for O(1) lookup
        sparse_score_map = {doc_id: score for doc_id, score in zip(self.doc_ids, full_sparse_scores)}

        # 3) Combine scores for dense candidates only
        results = []
        for d_score, doc_id in zip(dense_scores, dense_doc_ids):
            s_score = sparse_score_map.get(doc_id, 0.0)
            hybrid_score = (dense_weight * float(d_score)) + (sparse_weight * float(s_score))
            results.append((doc_id, hybrid_score))

        # 4) Sort by hybrid score descending
        results.sort(key=lambda x: x[1], reverse=True)

        if not results:
            return np.array([]), []

        final_doc_ids, final_scores = zip(*results)
        # Normalize hybrid scores before returning
        normalized_final_scores = self._normalize(np.array(final_scores))

        return normalized_final_scores, list(final_doc_ids)
    


class RetrievalEvaluator:
    """Loads QA dataset and computes standard retrieval metrics."""

    def __init__(self, qa_dataset_path: str, k_values: List[int] = [1, 3, 5, 10, 20]):
        """
        Initializes the Evaluator.

        Args:
            qa_dataset_path: Path to the JSON file containing QA pairs.
            k_values: A list of k values to use for @k metrics.
        """
        self.k_values = k_values
        
        print("Loading QA dataset...")
        with open(qa_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # We assume qa_pairs is a list of dicts, e.g.:
            # [{"query": "...", "relevant_ids": [123, 456]}, ...]
            self.qa_pairs = data.get('qa_pairs', [])
        print(f"✓ Loaded {len(self.qa_pairs)} QA pairs")

    def get_qa_pairs(self) -> List[Dict]:
        """Returns the loaded list of QA pairs."""
        return self.qa_pairs

    def recall_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate Recall@k."""
        if not relevant:
            return 0.0 # or 1.0 if you consider "all 0 relevant docs" retrieved
        retrieved_at_k = set(retrieved[:k])
        hits = len(retrieved_at_k & set(relevant))
        return hits / len(relevant)

    def precision_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate Precision@k."""
        if k == 0:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        hits = len(retrieved_at_k & set(relevant))
        return hits / k

    def reciprocal_rank(self, retrieved: List[int], relevant: List[int]) -> float:
        """Calculate Reciprocal Rank (MRR)."""
        relevant_set = set(relevant)
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0

    def average_precision(self, retrieved: List[int], relevant: List[int]) -> float:
        """Calculate Average Precision (MAP)."""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        hits = 0
        precision_sum = 0.0
        
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                hits += 1
                precision_at_i = hits / rank
                precision_sum += precision_at_i
                
        return precision_sum / len(relevant)

    def ndcg_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain (nDCG)@k."""
        relevant_set = set(relevant)
        # Calculate DCG@k
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2) # rank = i + 1
        
        # Calculate Ideal DCG@k (IDCG@k)
        num_relevant = len(relevant)
        ideal_k = min(num_relevant, k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
        
        return dcg / idcg if idcg > 0 else 0.0

    def hit_rate_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate Hit Rate@k."""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        retrieved_at_k = set(retrieved[:k])
        
        return 1.0 if len(retrieved_at_k & relevant_set) > 0 else 0.0