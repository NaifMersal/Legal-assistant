# Saudi Legal Assistant - Setup Complete ✅

## 🎉 System Overview

A full-stack AI-powered legal assistant for Saudi Arabian law, featuring hybrid retrieval (dense + sparse) with RAG and Gemini 2.5 Flash.

## 📦 Components

### Backend (FastAPI)
- **Location**: `/app`
- **Port**: `8000`
- **Status**: ✅ Running
- **Features**:
  - BGE-M3 embeddings (CPU-optimized for Mac)
  - FAISS vector search (16,371 legal documents)
  - BM25 sparse retrieval
  - Hybrid retrieval (70% dense + 30% sparse)
  - Google Gemini 2.5 Flash LLM
  - Session management
  - Health monitoring

### Frontend (React + TypeScript + Vite)
- **Location**: `/law-UI`
- **Port**: `8080`
- **Status**: ✅ Running
- **Features**:
  - Modern UI with shadcn/ui components
  - Real-time chat interface
  - Bilingual support (Arabic/English)
  - Connection status indicator
  - RAG/LLM mode switching

## 🚀 Quick Start

### Start Backend Server
```bash
cd /Users/hamad/Downloads/KAU/Legal-assistant
source venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Start Frontend
```bash
cd /Users/hamad/Downloads/KAU/Legal-assistant/law-UI
npm run dev
```

## 🌐 Access Points

- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Search Documents
```bash
POST /api/v1/search
Content-Type: application/json

{
  "query": "ما هي أحكام العقود؟",
  "k": 5,
  "dense_weight": 0.7,
  "sparse_weight": 0.3
}
```

### 3. Chat with AI
```bash
POST /api/v1/chat
Content-Type: application/json

{
  "message": "من يصدر الأنظمة في المملكة؟",
  "session_id": "optional_session_id",
  "mode": "rag"
}
```

## 🔧 Configuration

### Backend (.env)
```env
GOOGLE_API_KEY=AIzaSyCPrcLHuhKWETHeFafgYGzxV90DFlCBtZo
GEMINI_MODEL=gemini-2.5-flash
RAG_K=12
USE_FP16=False
FAISS_THREADS=1
```

### Frontend (law-UI/.env)
```env
VITE_API_URL=http://localhost:8000
```

## 🛠️ Technical Stack

### Backend
- FastAPI 0.115.0
- Python 3.9
- LangChain 0.3.7
- Google Gemini 2.5 Flash
- BGE-M3 embeddings
- FAISS vector database
- BM25 (rank-bm25)

### Frontend
- React 18
- TypeScript
- Vite 5
- shadcn/ui components
- Tailwind CSS

## 📊 Performance Optimizations

### Mac Stability Fixes Applied:
1. ✅ Disabled tokenizer parallelism (`TOKENIZERS_PARALLELISM=false`)
2. ✅ Limited threading (`OMP_NUM_THREADS=1`)
3. ✅ Disabled fp16 precision
4. ✅ CPU-only mode for embeddings
5. ✅ Reduced FAISS threads to 1
6. ✅ Batch size optimization

## 📝 Key Features

- ✅ **Hybrid Retrieval**: Combines dense (BGE-M3) and sparse (BM25) search
- ✅ **RAG System**: Tool-calling LLM with intelligent search decisions
- ✅ **Session Management**: Maintains conversation context
- ✅ **Bilingual Support**: Arabic and English interface
- ✅ **Real-time Chat**: WebSocket-like experience
- ✅ **Error Handling**: Graceful degradation and user feedback
- ✅ **Health Monitoring**: Connection status indicators

## 🎯 Usage Examples

### Example 1: Legal Question
**User**: "ما هو مصدر الإفتاء في المملكة العربية السعودية؟"
**AI**: "وفقاً للمادة السبعين من النظام الأساسي للحكم، تصدر الأنظمة والمعاهدات..."

### Example 2: Search Query
**Query**: "ما هي عقوبة السرقة؟"
**Results**: Relevant articles from نظام مكافحة الاحتيال المالي and related laws

## 🔒 Security Notes

- API key is configured (Google Gemini)
- CORS is configured for localhost development
- Session IDs are generated client-side
- No authentication implemented (development mode)

## 📈 Next Steps (Optional Enhancements)

1. Add user authentication
2. Implement rate limiting
3. Add caching layer (Redis)
4. Deploy to production (Docker + Cloud)
5. Add more languages
6. Implement feedback system
7. Add analytics dashboard

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000
pkill -f "uvicorn app.main:app"
```

### Frontend connection issues
- Verify backend is running on port 8000
- Check `.env` file has correct `VITE_API_URL`
- Clear browser cache

### BGE-M3 loading issues
- Ensure environment variables are set
- Check available RAM (model needs ~2GB)
- Verify transformers version (4.44.2)

## 📞 Support

For issues or questions, refer to:
- Backend logs: Check terminal running uvicorn
- Frontend logs: Browser console (F12)
- API documentation: http://localhost:8000/docs

---

**Status**: ✅ Fully Operational
**Last Updated**: October 23, 2025
**Developed by**: Copilot + Hamad
