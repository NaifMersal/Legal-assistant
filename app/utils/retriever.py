import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from typing import Any, Dict, List, Optional, Tuple





class Retriever:
    """Handles loading retrieval models and performing search using aligned ordinal indices."""

    def __init__(self, faiss_index_path: str, documents_path: str,
                 embeddings_model: Any, metric_type: str = 'ip'):
        """
        Initializes the Retriever with strict alignment between FAISS index, corpus, and metadata.

        All documents are assigned ordinal IDs: 0, 1, ..., N-1, matching FAISS index positions.

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
        self.doc_metadata = []  # List indexed by ordinal doc_id (0 to N-1)
        self.sub_categories_article_ids = {}  # Maps subcat -> list of sub_categories indices

        with open(documents_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
            self._extract_documents_with_metadata(docs_data)

        print(f"✓ Loaded {len(self.doc_metadata)} documents with metadata")

        # Build corpus in the same order as metadata
        self.corpus = [meta['text'] for meta in self.doc_metadata]
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("✓ BM25 initialized successfully!")

    def _extract_documents_with_metadata(self, docs_data: Dict):
        """Extract documents in deterministic order and assign ordinal IDs (0, 1, 2, ...)."""
        current_id = 0
        for category, subcats in docs_data.items():
            for subcat, systems in subcats.items():
                self.sub_categories_article_ids[subcat] = []
                for system_name, system_data in systems.items():
                    system_brief = system_data.get('brief', '')
                    system_metadata = system_data.get('metadata', {})
                    parts = system_data.get('parts', {})
                    for part_name, articles in parts.items():
                            for article in articles:
                                    text = f"{article.get('Article_Title', '')}\n{article.get('Article_Text', '')}".strip()
                                    if text:
                                        # Assign ordinal ID
                                        doc_id = current_id
                                        article_id = article.get('id')
                                        if article_id!= doc_id:
                                            raise ValueError(f"Document ID mismatch: expected {doc_id}, found {article_id}")
                                        
                                        self.doc_metadata.append({
                                            'text': text,
                                            'id': doc_id,  # ordinal ID
                                            'title': article.get('Article_Title', ''),
                                            'category': category,
                                            'subcategory': subcat,
                                            'system': system_name,
                                            'system_brief': system_brief,
                                            'system_metadata': system_metadata,
                                            'part': part_name
                                        })
                                        self.sub_categories_article_ids[subcat].append(doc_id)
                                        current_id += 1

    def get_article_metadata(self, doc_id: int) -> Dict:
        """Get cached article metadata by ordinal ID in O(1) time."""
        return self.doc_metadata[doc_id]


    def get_subcategories_articles_ids(self, subcategories: List[str]) -> List[int]:
        """Retrieve all ordinal article IDs belonging to the specified subcategories."""
        article_ids = []
        for subcat in subcategories:
            ids = self.sub_categories_article_ids.get(subcat, [])
            article_ids.extend(ids)
        print(article_ids)
        return article_ids

    def _distances_to_scores(self, distances: np.ndarray) -> np.ndarray:
        if self.metric_type == 'l2':
            return 1.0 / (1.0 + distances)
        else:  # 'ip'
            return (distances + 1) / 2

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        embedding = np.array(self.embeddings_model.encode([query])["dense_vecs"], dtype='float32')
        if self.metric_type == 'ip':
            faiss.normalize_L2(embedding)
        return embedding

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0,1] range."""
        if len(scores) == 0:
            return scores
        min_s, max_s = np.min(scores), np.max(scores)
        return np.ones_like(scores) if max_s == min_s else (scores - min_s) / (max_s - min_s)

    def dense(self, query: str, k: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Dense retrieval using FAISS. Returns (normalized scores, ordinal doc_ids)."""
        query_emb = self._embed_query(query)
        distances, indices = self.index.search(query_emb, k)

        scores = self._distances_to_scores(distances[0])
        doc_ids = indices[0].tolist()  # Already ordinal indices
        return self._normalize(scores), doc_ids
    
    def sparse(self, query: str, k: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Sparse retrieval using BM25. Returns (normalized scores, ordinal doc_ids)."""
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Use argpartition for better performance than full argsort
        if k < len(bm25_scores):
            # Get indices of top-k scores without full sort
            top_k_indices = np.argpartition(bm25_scores, -k)[-k:]
            # Sort only the top-k results
            top_k_indices = top_k_indices[np.argsort(bm25_scores[top_k_indices])[::-1]]
        else:
            top_k_indices = np.argsort(bm25_scores)[::-1]
        
        scores = bm25_scores[top_k_indices]
        doc_ids = top_k_indices.tolist()
        return self._normalize(scores), doc_ids

    def hybrid(self, query: str, k: int = 10, dense_weight: float = 0.7, sparse_weight: float = 0.3) -> Tuple[np.ndarray, List[int]]:
        """
        Hybrid retrieval using dense and sparse signals.
        - Dense: top-k candidates from FAISS (ordinal IDs).
        - Sparse: BM25 scores only for dense candidates.
        - Combine only on dense candidates.
        """
        # Get dense results
        dense_scores, dense_doc_ids = self.dense(query, k=k)
        
        if not dense_doc_ids:
            return np.array([]), []
        
        # Get BM25 scores ONLY for the dense candidate documents
        tokenized_query = word_tokenize(query.lower())
        sparse_scores = np.array([self.bm25.get_score(tokenized_query, doc_id) for doc_id in dense_doc_ids])
        
        # Normalize sparse scores for the candidate set
        normalized_sparse = self._normalize(sparse_scores)
        
        # Combine scores using vectorized operations
        hybrid_scores = dense_weight * dense_scores + sparse_weight * normalized_sparse
        
        # Re-sort by hybrid scores
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        
        final_scores = hybrid_scores[sorted_indices]
        final_doc_ids = [dense_doc_ids[i] for i in sorted_indices]
        
        return final_scores, final_doc_ids

    def re_ranked_search(
        self,
        query: str,
        relevant_terms: List[str] = None,
        subcategory_filters: List[str] = None,
        k: int = 10,
        keyword_boost: float = 0.3
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Perform search with semantic keyword boosting (non-strict matching).

        Instead of hard filtering, this method:
        - Uses subcategory filters as hard constraints (ordinal indices)
        - Boosts documents semantically related to relevant terms via BM25
        - Dynamically expands candidate set for keyword-enhanced results

        Args:
            query: Search query string
            relevant_terms: Terms to boost semantically related documents (optional)
            subcategory_filters: Subcategories for hard filtering (optional)
            k: Number of results to return
            keyword_boost: Weight for BM25 signal (0.0–1.0)

        Returns:
            Tuple of (normalized scores, ordinal doc_ids)
        """
        # 1. Apply hard subcategory filter if provided
        filtered_indices = None
        if subcategory_filters:
            valid_doc_ids = self.get_subcategories_articles_ids(subcategory_filters)
            print(f"Filtering to {len(valid_doc_ids)} documents in subcategories: {subcategory_filters}")
            if not valid_doc_ids:
                return np.array([]), []
            filtered_indices = np.array(valid_doc_ids, dtype=np.int64)

        # 2. Determine candidate set size
        base_candidates = k
        if relevant_terms and len(relevant_terms) > 0:
            base_candidates = min(1000, k * 15)  # Expand for keyword signals

        # Ensure we don't request more than available
        base_candidates = min(base_candidates, self.index.ntotal)

        # 3. Perform dense search
        query_embedding = self._embed_query(query)
        distances, candidate_indices = self.index.search(
            query_embedding, min(base_candidates * 2, self.index.ntotal)
        )

        # Flatten
        candidate_indices = candidate_indices[0]
        distances = distances[0]

        # 4. Apply hard subcategory filtering manually (since FAISS selector is ignored in FlatIP)
        if filtered_indices is not None:
            mask = np.isin(candidate_indices, filtered_indices)
            candidate_indices = candidate_indices[mask]
            distances = distances[mask]

        # Handle empty results
        if candidate_indices.size == 0:
            return np.array([]), []

        # 5. Normalize dense scores
        dense_scores = self._distances_to_scores(distances)
        dense_scores = self._normalize(dense_scores)

        # 6. Early return if no relevant terms
        if not relevant_terms or len(relevant_terms) == 0:
            top_k_indices = candidate_indices[:k]
            top_k_scores = dense_scores[:k]
            return self._normalize(top_k_scores), top_k_indices.tolist()

        # 7. Compute BM25-based keyword relevance for candidates
        candidate_bm25_scores = np.zeros(len(candidate_indices))
        for term in relevant_terms:
            term_tokenized = word_tokenize(term.lower())
            if not term_tokenized:
                continue
            term_scores = self.bm25.get_scores(term_tokenized)  # Shape: (N,)
            candidate_bm25_scores += term_scores[candidate_indices]

        # Normalize BM25 scores
        max_bm25 = np.max(candidate_bm25_scores)
        if max_bm25 > 0:
            candidate_bm25_scores /= max_bm25
        else:
            candidate_bm25_scores = np.zeros_like(candidate_bm25_scores)

        # 8. Fuse dense and sparse signals
        combined_scores = (
            (1.0 - keyword_boost) * dense_scores +
            keyword_boost * candidate_bm25_scores
        )

        # 9. Select top-k results
        top_k_local_indices = np.argsort(combined_scores)[::-1][:k]
        top_scores = combined_scores[top_k_local_indices]
        top_doc_ids = candidate_indices[top_k_local_indices].tolist()

        return self._normalize(top_scores), top_doc_ids


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