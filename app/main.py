"""FastAPI application for Legal Assistant"""
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings, Settings
from app.models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
    ErrorResponse
)
from app.rag_service import get_rag_service, RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Session storage (in production, use Redis or similar)
sessions: Dict[str, list] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Legal Assistant API...")
    settings = get_settings()
    rag_service = get_rag_service(settings)
    
    try:
        rag_service.initialize()
        logger.info("✓ RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Legal Assistant API...")


# Create FastAPI app
app = FastAPI(
    title="Legal Assistant API",
    description="AI-powered Saudi Arabian Legal Assistant with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get RAG service
def get_service(settings: Settings = Depends(get_settings)) -> RAGService:
    """Get RAG service dependency"""
    service = get_rag_service(settings)
    if not service.is_initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Legal Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(service: RAGService = Depends(get_service)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=service.is_initialized
    )


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: RAGService = Depends(get_service)
):
    """
    Chat endpoint - Answers questions using full RAG system (Retriever + LLM)
    
    This endpoint uses the complete RAG pipeline:
    1. Retrieves relevant legal documents using hybrid search
    2. Generates AI-powered answers using Gemini LLM with context
    
    - **message**: User's question
    - **session_id**: Optional session ID for conversation history
    - **mode**: 'rag' for retrieval-augmented answers (default), 'llm' for general chat
    """
    try:
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get answer from RAG system
        answer = service.answer(request.message)
        
        # Store in session (simple in-memory storage)
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append({
            "role": "user",
            "content": request.message
        })
        sessions[session_id].append({
            "role": "assistant",
            "content": answer
        })
        
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            mode=request.mode,
            sources=None  # Could extract sources from RAG system if needed
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


@app.post(f"{settings.api_prefix}/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    service: RAGService = Depends(get_service)
):
    """
    Search endpoint - RAG retrieval only (returns top-k documents without LLM)
    
    This endpoint performs ONLY retrieval from the legal document database:
    - Uses hybrid search (dense BGE-M3 embeddings + sparse BM25)
    - Returns raw documents with scores
    - NO LLM processing or answer generation
    
    - **query**: Search query
    - **k**: Number of results to return (1-20, default: 12)
    - **dense_weight**: Weight for semantic similarity search (0.0-1.0, default: 0.7)
    - **sparse_weight**: Weight for keyword matching (0.0-1.0, default: 0.3)
    """
    try:
        results = service.search(
            query=request.query,
            k=request.k,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight
        )
        
        search_results = [
            SearchResult(**result) for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process search request: {str(e)}"
        )


@app.delete(f"{settings.api_prefix}/session/{{session_id}}")
async def clear_session(
    session_id: str,
    service: RAGService = Depends(get_service)
):
    """Clear conversation history for a session"""
    try:
        if session_id in sessions:
            del sessions[session_id]
        
        # Clear RAG system history
        service.clear_history()
        
        return {"message": "Session cleared successfully", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear session: {str(e)}"
        )


@app.get(f"{settings.api_prefix}/session/{{session_id}}")
async def get_session(session_id: str):
    """Get conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": sessions[session_id]
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers
    )
