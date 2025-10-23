/**
 * API Client for Legal Assistant Backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ChatMessage {
  message: string;
  session_id?: string;
  mode?: 'rag' | 'llm';
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  mode: string;
  sources: SearchResult[] | null;
}

export interface SearchRequest {
  query: string;
  k?: number;
  dense_weight?: number;
  sparse_weight?: number;
}

export interface SearchResult {
  article_id: number;
  law_name: string;
  article_title: string;
  article_text: string;
  category: string;
  score: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: boolean;
}

class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Check API health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new APIError(response.status, 'Health check failed');
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new APIError(0, 'Failed to connect to API server');
  }
}

/**
 * Send a chat message and get AI response
 */
export async function sendChatMessage(
  message: string,
  sessionId?: string,
  mode: 'rag' | 'llm' = 'rag'
): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        mode,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        response.status,
        errorData.detail || 'Failed to send message'
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new APIError(0, 'Failed to connect to API server');
  }
}

/**
 * Search legal documents
 */
export async function searchDocuments(
  query: string,
  k: number = 5,
  denseWeight: number = 0.7,
  sparseWeight: number = 0.3
): Promise<SearchResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        k,
        dense_weight: denseWeight,
        sparse_weight: sparseWeight,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        response.status,
        errorData.detail || 'Failed to search documents'
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new APIError(0, 'Failed to connect to API server');
  }
}

/**
 * Generate a unique session ID
 */
export function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
