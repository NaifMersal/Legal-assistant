"""Pydantic models for request/response validation"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model"""
    role: Literal["user", "assistant"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    mode: Literal["rag", "llm"] = Field("rag", description="Chat mode: 'rag' for search-based, 'llm' for general chat")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="Assistant's answer")
    session_id: str = Field(..., description="Session ID for this conversation")
    mode: str = Field(..., description="Mode used for this response")
    sources: Optional[List[dict]] = Field(None, description="Source articles used (for RAG mode)")


class SearchResult(BaseModel):
    """Search result model"""
    article_id: int
    law_name: str
    article_title: str
    article_text: str
    category: str
    score: float


class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    k: int = Field(5, ge=1, le=20, description="Number of results to return")
    dense_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for dense retrieval")
    sparse_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for sparse retrieval")


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    query: str
    results: List[SearchResult]
    total: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
