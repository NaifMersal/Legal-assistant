import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from typing import Any, Dict, List, Optional, Tuple




class Retriever:
    """Handles loading retrieval models and performing search using aligned ordinal indices."""

    def __init__(self, faiss_index: faiss.Index, documents_path: str,
                 embeddings_model: Any, metric_type: str = 'ip'):
        """
        Initializes the Retriever with strict alignment between FAISS index, corpus, and metadata.

        All documents are assigned ordinal IDs: 0, 1, ..., N-1, matching FAISS index positions.

        Args:
            faiss_index: the FAISS index.
            documents_path: Path to the JSON file containing documents and metadata.
            embeddings_model: An embeddings model with an .encode() method.
            metric_type: The metric type used for FAISS ('ip' or 'l2').
        """
        print("Initializing Retriever...")
        self.index = faiss_index
        self.embeddings_model = embeddings_model
        self.metric_type = metric_type.lower()

        if self.metric_type not in ['ip', 'l2']:
            raise ValueError("metric_type must be either 'ip' or 'l2'")

        print("Loading documents and metadata...")
        self.docs = []  # List indexed by ordinal doc_id (0 to N-1)
        self.subcategories_article_ids = {}  # Maps subcat -> list of sub_categories indices
        self.law_id_to_name = {}  # Maps article_id to law_name
        self.laws_data = {}
        self.part_id_to_articles_ids = {}
        self.corpus = []  # List of document texts indexed by ordinal doc_id


        with open(documents_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
            self._extract_documents_with_metadata(docs_data)

        print(f"✓ Loaded {len(self.docs)} documents with metadata")

        # Build corpus in the same order as metadata
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("✓ BM25 initialized successfully!")

    def _extract_documents_with_metadata(self, docs_data: Dict):
        """Extract documents in deterministic order and assign ordinal IDs (0, 1, 2, ...)."""
        law_id = 0
        current_id = 0
        for category, subcats in docs_data.items():
            for subcat, laws in subcats.items():
                self.subcategories_article_ids[subcat] = []
                for law_name, law_data in laws.items():
                    law_brief = law_data.get('brief', '')
                    law_metadata = law_data.get('metadata', {})
                    parts = law_data.get('parts', {})
                    self.laws_data[law_name] = {}
                    self.laws_data[law_name]['brief'] = law_brief
                    self.laws_data[law_name]['metadata'] = law_metadata
                    self.laws_data[law_name]['article_ids'] = []
                    self.law_id_to_name[law_id] = law_name
                    law_id += 1
                    for part_name, articles in parts.items():
                            part_id = law_name + "|" + part_name 
                            self.part_id_to_articles_ids[part_id] = []
                            for article in articles:
                                    # Assign ordinal ID
                                    doc_id = current_id
                                    article_id = article.get('id')
                                    if article_id!= doc_id:
                                        raise ValueError(f"Document ID mismatch: expected {doc_id}, found {article_id}")
                                            # Combine elements with clean formatting

                                    title = article.get('Article_Title', '').strip()
                                    brief = law_brief.strip()
                                    text = article.get('Article_Text', '').strip()
                                    # Filter out empty parts and join with double newlines for clarity
                            
                                    self.docs.append({
                                        'text': text,
                                        'id': doc_id,  # ordinal ID
                                        'title': article.get('Article_Title', ''),
                                        'category': category,
                                        'subcategory': subcat,
                                        'law_name': law_name,
                                        'law_brief': law_brief,
                                        'law_metadata': law_metadata,
                                        'part': part_name
                                    })
                                    meta_str = ", ".join([f"{k}: {v}" for k, v in law_metadata.items()])
                                    text_parts = [
                                        f"Law Title: {title}" if title else "",
                                        f"Law Brief: {brief}" if brief else "",
                                        f"Law Text: {text}" if text else "",
                                        f"Law Metadata: {meta_str}" if meta_str else "",
                                    ]
                                    entry = "\n\n".join(filter(None, text_parts)).strip()
                                    self.corpus.append(entry)
                                    self.laws_data[law_name]['article_ids'].append(doc_id)
                                    self.subcategories_article_ids[subcat].append(doc_id)
                                    self.part_id_to_articles_ids[part_id].append(doc_id)
                                    current_id += 1



    def get_subcategories_articles_ids(self, subcategories: List[str]) -> List[int]:
        """Retrieve all article IDs belonging to the specified subcategories."""
        article_ids = []
        for subcat in subcategories:
            ids = self.subcategories_article_ids.get(subcat, [])
            article_ids.extend(ids)
        return article_ids
    
    
    def get_laws_article_ids(self, law_names: List[str]) -> List[int]:
        """Retrieve all article IDs belonging to the specified laws."""
        article_ids = []
        for law_name in law_names:
            ids = self.laws_data.get(law_name, {}).get('article_ids', [])
            article_ids.extend(ids)
        return article_ids
    
    def get_parts_article_ids(self, part_ids: List[str]) -> List[int]:
        """Retrieve all article IDs belonging to the specified parts."""
        article_ids = []
        for part_id in part_ids:
            ids = self.part_id_to_articles_ids.get(part_id, [])
            article_ids.extend(ids)
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
    
    def _get_candidate_bm25_scores(self, query: str, candidate_indices: List[int]) -> np.ndarray:
        """Get BM25 scores for specific candidates using batch processing"""
        tokenized = word_tokenize(query.lower())
        return np.array(self.bm25.get_batch_scores(tokenized, candidate_indices))

    def _fuse_scores(
        self,
        scores_list: List[np.ndarray],
        weights: List[float],
        doc_ids: List[int],
        k: int
    ) -> Tuple[np.ndarray, List[int]]:
        """Fuse and re-rank scores using multiple score arrays and weights"""
        # Validate inputs
        if len(scores_list) != len(weights):
            raise ValueError("Number of score arrays must match number of weights")
        
        if len(scores_list) == 0:
            raise ValueError("At least one score array is required")
        
        # Combine scores using weighted sum
        combined = np.zeros_like(scores_list[0])
        for scores, weight in zip(scores_list, weights):
            combined += weight * scores
        
        # Get top k indices
        top_indices = np.argsort(combined)[::-1][:k]
        
        return (
            self._normalize(combined[top_indices]),
            [doc_ids[i] for i in top_indices]
        )
    def _dense(
        self, 
        index: faiss.Index, 
        query: str, 
        k: int = 10, 
        selector: faiss.IDSelector = None
    ) -> Tuple[np.ndarray, List[int]]:
        """Dense retrieval with optional filtering. Returns (normalized scores, ordinal doc_ids)."""
        query_emb = self._embed_query(query)
        params = faiss.SearchParameters(sel=selector) if selector else None
        distances, indices = index.search(query_emb, k, params=params)

        if indices.size == 0 or len(indices[0]) == 0:
            return np.array([]), []

        scores = self._distances_to_scores(distances[0])
        doc_ids = indices[0].tolist()
        return self._normalize(scores), doc_ids
    
    def _hybrid(self, index, query: str, k: int = 10, keyword_boost: float = 0.3) -> Tuple[np.ndarray, List[int]]:
        """Reuses dense search and fusion logic with minimal overhead"""
        dense_scores, dense_doc_ids = self._dense(index, query, k=5*k)
        if not dense_doc_ids:
            return np.array([]), []
        
        bm25_scores = self._get_candidate_bm25_scores(query, dense_doc_ids)
        bm25_scores = self._normalize(bm25_scores)
        
        return self._fuse_scores([dense_scores, bm25_scores], [1- keyword_boost, keyword_boost], dense_doc_ids, k)    

    def dense(self, query: str, k: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Dense retrieval using FAISS. Returns (normalized scores, ordinal doc_ids)."""
        return self._dense(self.index, query, k)

    def hybrid(self, query: str, k: int = 10, keyword_boost: float = 0.3) -> Tuple[np.ndarray, List[int]]:
        """Hybrid retrieval using dense and sparse signals."""
        return self._hybrid(self.index, query, k, keyword_boost)

    def filtered_search(
        self,
        query: str,
        relevant_terms: List[str] = None,
        subcategory_filters: List[str] = None,
        laws_filters: List[str] = None,
        parts_filters: List[str] = None,
        k: int = 10,
        keyword_boost: float = 0.3
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Re-ranked search combining dense retrieval with BM25 re-ranking and optional filtering.
        """
        # 1. Apply hard filters (unchanged)
        filtered_indices = None
        if subcategory_filters:
            valid_doc_ids = self.get_subcategories_articles_ids(subcategory_filters)
            filtered_indices = np.array(valid_doc_ids, dtype=np.int64)

        if laws_filters:
            law_filtered_ids = self.get_laws_article_ids(laws_filters)
            if filtered_indices is not None:
                filtered_indices = np.intersect1d(filtered_indices, np.array(law_filtered_ids, dtype=np.int64))
            else:
                filtered_indices = np.array(law_filtered_ids, dtype=np.int64)

        if parts_filters:
            part_filtered_ids = self.get_parts_article_ids(parts_filters)
            if filtered_indices is not None:
                filtered_indices = np.intersect1d(filtered_indices, np.array(part_filtered_ids, dtype=np.int64))
            else:
                filtered_indices = np.array(part_filtered_ids, dtype=np.int64)

        # 2. Replaced with reusable dense search with filtering
        selector = faiss.IDSelectorArray(filtered_indices) if filtered_indices is not None else None
        dense_scores, candidate_doc_ids = self._dense(self.index, query, k, selector=selector)

        if not candidate_doc_ids:
            return np.array([]), []
        
        # 3. Early return if no boosting terms
        if not relevant_terms or len(relevant_terms) == 0:
            return dense_scores[:k], candidate_doc_ids[:k]  # Already normalized

        # 4. Replaced with single BM25 batch call (major efficiency improvement)
        combined_terms = " ".join(relevant_terms)
        bm25_scores = self._get_candidate_bm25_scores(combined_terms, candidate_doc_ids)
        bm25_scores = self._normalize(bm25_scores)

        # 5. Reuse hybrid fusion logic
        return self._fuse_scores([dense_scores, bm25_scores], [1- keyword_boost, keyword_boost], candidate_doc_ids, k)
    
    def re_ranked_search(
        self,
        query: str,
        relevant_terms: List[str] = None,
        subcategory_to_scores: Dict[str, int] = None,
        laws_to_scores: Dict[str, int] = None,
        parts_to_scores: Dict[str, int] = None,
        k: int = 10,
        keyword_boost: float = 0.3,
        subcategory_global_weight: float = 0.0,
        laws_global_weight: float = 0.0,
        parts_global_weight: float = 0.0
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Re-ranked search that boosts documents based on filter matches instead of hard filtering.
        """
        candidate_set_size = 5 * k  # Get more candidates for re-ranking
        
        #  Get initial candidate set (no filtering)
        dense_scores, candidate_doc_ids = self._dense(self.index, query, candidate_set_size, selector=None)
        
        if not candidate_doc_ids:
            return np.array([]), []
        
        #  Prepare scores and weights for fusion
        scores_list = []
        weights_list = []  # Dense score weight
        
        # Compute BM25 scores if relevant terms exist
        if relevant_terms and len(relevant_terms) > 0:
            bm25_scores = np.zeros(len(candidate_doc_ids))
            combined_terms = " ".join(relevant_terms)
            bm25_scores = self._get_candidate_bm25_scores(combined_terms, candidate_doc_ids)
            bm25_scores = self._normalize(bm25_scores)
            scores_list.append(bm25_scores)
            weights_list.append(keyword_boost)
        
        #  Compute filter scores
        if subcategory_to_scores and subcategory_global_weight > 0:
            subcategory_scores = np.zeros(len(candidate_doc_ids))
            for i, doc_id in enumerate(candidate_doc_ids):
                subcat = self.docs[doc_id]['subcategory']
                subcategory_scores[i] = subcategory_to_scores.get(subcat, 0)

            scores_list.append(subcategory_scores)
            weights_list.append(subcategory_global_weight) 

        if laws_to_scores and laws_global_weight > 0:
            laws_scores = np.zeros(len(candidate_doc_ids))
            for i, doc_id in enumerate(candidate_doc_ids):
                law_name = self.docs[doc_id]['law_name']
                laws_scores[i] = laws_to_scores.get(law_name, 0)

            scores_list.append(laws_scores)
            weights_list.append(laws_global_weight)

        

        if parts_to_scores and parts_global_weight > 0:
            parts_scores = np.zeros(len(candidate_doc_ids))
            for i, doc_id in enumerate(candidate_doc_ids):
                part_name = self.docs[doc_id]['part']
                parts_scores[i] = parts_to_scores.get(part_name, 0)

            scores_list.append(parts_scores)
            weights_list.append(parts_global_weight)

        dense_weight = 1- sum(weights_list)
        scores_list.append(dense_scores)
        weights_list.append(dense_weight)
        
        #  Fuse and return top k results
        return self._fuse_scores(scores_list, weights_list, candidate_doc_ids, k)

    def sparse(self, query: str, k: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Sparse retrieval using BM25. Returns (normalized scores, ordinal doc_ids)."""
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        if k < len(bm25_scores):
            top_k_indices = np.argpartition(bm25_scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(bm25_scores[top_k_indices])[::-1]]
        else:
            top_k_indices = np.argsort(bm25_scores)[::-1]
        
        scores = bm25_scores[top_k_indices]
        doc_ids = top_k_indices.tolist()
        return self._normalize(scores), doc_ids


    def __call__(self, *args, **kwds):
        return self.hybrid(*args, **kwds)