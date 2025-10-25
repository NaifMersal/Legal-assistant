"""RAG Service - Manages RAG system initialization and queries"""
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import faiss
import nltk
from FlagEmbedding import BGEM3FlagModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage

# Import from utils
from app.utils.retrievers.retriever import Retriever
from app.utils.rag import LegalAssistantRAG, RAGConfig

from app.config import Settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service class for managing RAG system"""
    
    def __init__(self, settings: Settings):
        """Initialize RAG service with settings"""
        self.settings = settings
        self.retriever: Optional[Retriever] = None
        self.rag_system: Optional[LegalAssistantRAG] = None
        self.embeddings_model = None
        self._initialized = False
        
    def initialize(self):
        """Initialize models and RAG system"""
        if self._initialized:
            logger.info("RAG service already initialized")
            return
            
        try:
            logger.info("Initializing RAG service...")
            
            # Set FAISS threads
            faiss.omp_set_num_threads(self.settings.faiss_threads)
            
            # Download NLTK data
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK data: {e}")
            
            # Load BGE-M3 embeddings model with optimized settings for Mac
            logger.info(f"Loading embeddings model: {self.settings.embeddings_model_name}")
            import os
            # Set environment variables for better Mac compatibility
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Load with minimal resource usage
            self.embeddings_model = BGEM3FlagModel(
                self.settings.embeddings_model_name,
                use_fp16=False,  # Disable fp16 for Mac stability
                device='cpu',
                batch_size=1,  # Process one query at a time
                max_length=512,  # Reduce max length to save memory
                normalize_embeddings=True
            )
            logger.info("✓ Embeddings model loaded successfully")
            
            # Get absolute paths
            base_dir = self.settings.base_dir
            index_path = base_dir / self.settings.faiss_index_path
            documents_path = base_dir / self.settings.documents_path
            
            # Initialize retriever
            logger.info(f"Initializing retriever with {self.settings.metric_type.upper()} metric")
            self.retriever = Retriever(
                faiss_index_path=str(index_path),
                documents_path=str(documents_path),
                embeddings_model=self.embeddings_model,
                metric_type=self.settings.metric_type
            )
            logger.info("✓ Retriever initialized")
            
            # Initialize Gemini LLM
            if not self.settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment variables")
                
            logger.info(f"Initializing Gemini model: {self.settings.gemini_model}")
            llm = ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                temperature=self.settings.gemini_temperature,
                google_api_key=self.settings.google_api_key
            )
            logger.info("✓ Gemini LLM initialized")
            
            # Initialize RAG system
            rag_config = RAGConfig(
                k=self.settings.rag_k,
                dense_weight=self.settings.rag_dense_weight,
                sparse_weight=self.settings.rag_sparse_weight,
                temperature=self.settings.gemini_temperature
            )
            
            self.rag_system = LegalAssistantRAG(
                retriever=self.retriever,
                llm=llm,
                config=rag_config
            )
            logger.info("✓ RAG system initialized")
            
            self._initialized = True
            logger.info("🎉 RAG service initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
            raise
    
    def search(
        self,
        query: str,
        k: int = 12,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search - Retrieval ONLY (no LLM)
        
        Returns top-k most relevant legal documents without any LLM processing.
        Uses hybrid search combining:
        - Dense retrieval (BGE-M3 semantic embeddings)
        - Sparse retrieval (BM25 keyword matching)
        
        Args:
            query: Search query text
            k: Number of documents to return
            dense_weight: Weight for semantic similarity (0.0-1.0)
            sparse_weight: Weight for keyword matching (0.0-1.0)
            
        Returns:
            List of documents with metadata and scores
        """
        if not self._initialized:
            raise RuntimeError("RAG service not initialized")
        
        logger.info(f"Searching for: '{query}' (k={k})")
        
        scores, indices = self.retriever.hybrid(
            query,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        
        results = []
        for score, idx in zip(scores, indices):
            metadata = self.retriever.get_article_metadata(idx)
            if metadata:
                results.append({
                    'article_id': metadata['id'],
                    'law_name': metadata['system'],
                    'article_title': metadata['title'],
                    'article_text': metadata['text'],
                    'category': metadata['category'],
                    'score': float(score)
                })
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def answer(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        Get answer using full RAG system (Retriever + LLM)
        
        Complete RAG pipeline:
        1. Retrieves relevant legal documents using hybrid search
        2. Uses Gemini LLM to generate contextual answer based on retrieved documents
        3. Maintains conversation history for follow-up questions
        
        Args:
            question: User's question in natural language
            chat_history: Optional conversation history as LangChain messages
            
        Returns:
            AI-generated answer with legal context
        """
        if not self._initialized:
            raise RuntimeError("RAG service not initialized")
        
        logger.info(f"Answering question: '{question}'")
        answer = self.rag_system.answer(question, chat_history=chat_history)
        return answer
    
    def clear_history(self):
        """Clear conversation history"""
        if self._initialized and self.rag_system:
            self.rag_system.clear_history()
            logger.info("Conversation history cleared")
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized


# Global service instance
_rag_service: Optional[RAGService] = None


def get_rag_service(settings: Settings) -> RAGService:
    """Get or create RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(settings)
    return _rag_service
