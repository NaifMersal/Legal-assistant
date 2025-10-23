# 🏛️ مسألة قانونية - Legal Assistant

An intelligent AI-powered legal assistant specialized in Saudi Arabian law, featuring a hybrid RAG (Retrieval-Augmented Generation) system with a modern Arabic-first React frontend.

## 🌟 Features

- **Hybrid Retrieval System**: Combines dense (FAISS + BGE-M3) and sparse (BM25) search for accurate legal document retrieval
- **Advanced RAG Pipeline**: Tool-calling LLM (Gemini 2.5 Flash) with intelligent search decision-making
- **Smart Prompting**: Avoids unnecessary searches for greetings and general questions
- **Conversation History**: Maintains context across multiple turns for coherent conversations
- **Modern RTL UI**: Beautiful Arabic-first interface with right-to-left support, built with React + TypeScript + shadcn/ui
- **Custom Branding**: Professional مسألة قانونية branding with custom icons and color scheme
- **RESTful API**: FastAPI backend with comprehensive endpoints and WebSocket support
- **Real-time Updates**: Live connection status and typing indicators

## 🏗️ Architecture

```
┌─────────────────┐
│  React UI (RTL) │ Arabic-first Interface
│  مسألة قانونية   │ TypeScript + shadcn/ui
└────────┬────────┘
         │ HTTP/REST + WebSocket
┌────────▼────────┐
│  FastAPI Server │ Python 3.9+
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
┌───▼──┐  ┌───▼────┐
│ RAG  │  │Retriever│
│System│  │(Hybrid) │
└───┬──┘  └───┬────┘
    │         │
┌───▼──────┐  ┌───▼────┐
│ Gemini   │  │ FAISS  │
│2.5 Flash │  │ + BM25 │
│(Tool Use)│  │16.3K docs
└──────────┘  └────────┘
```

## 📁 Project Structure

```
Legal-assistant/
├── app/                      # FastAPI backend
│   ├── main.py              # Main API application with WebSocket
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── rag_service.py       # RAG service wrapper
│   └── utils/               # Utility modules
│       ├── rag.py           # RAG implementation
│       └── retriever.py     # Hybrid retrieval system
├── law-UI/                  # React frontend (RTL)
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ChatInterface.tsx  # Main chat UI
│   │   │   ├── Hero.tsx     # Landing hero section
│   │   │   ├── Features.tsx # Features showcase
│   │   │   └── ui/          # shadcn/ui components
│   │   ├── pages/           # Page components
│   │   │   └── Index.tsx    # Main landing page
│   │   └── lib/             # Utilities
│   ├── public/              # Static assets
│   │   ├── logo-small.png   # مسألة قانونية logo
│   │   ├── bot-icon-small.png
│   │   ├── user-icon-small.png
│   │   └── favicon.png
│   └── package.json
├── data/                    # Data files
│   ├── saudi_laws_scraped.json  # 16,371 legal documents
│   ├── m3_legal_faiss.index     # FAISS vector index
│   └── evaluation_data/         # QA datasets for testing
├── start_server.sh          # Quick start script (Mac/Linux)
├── start_server.bat         # Quick start script (Windows)
├── test_api.py             # API testing script
├── check_setup.py          # Setup verification script
├── requirements.txt         # Python dependencies
└── .env.example            # Environment variables template
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Node.js 18+** (for frontend)
- **Google API Key** (for Gemini 2.5 Flash)
- **8GB+ RAM** recommended for embedding model

### Quick Start (Recommended)

**For macOS/Linux:**
```bash
chmod +x start_server.sh
./start_server.sh
```

**For Windows:**
```bash
start_server.bat
```

These scripts will automatically:
- Create/activate virtual environment
- Install Python dependencies
- Check for `.env` file
- Start the FastAPI backend on port 8000

### Manual Setup

#### Backend Setup

1. **Clone the repository**
   ```bash
   cd Legal-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

5. **Run the backend server**
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`
   - Docs: `http://localhost:8000/docs`
   - Health: `http://localhost:8000/health`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd law-UI
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run development server**
   ```bash
   npm run dev
   ```

   The UI will be available at `http://localhost:5173`
   
   **Custom port (optional):**
   ```bash
   npm run dev -- --port 8081
   ```

### Verify Setup

Run the setup checker:
```bash
python check_setup.py
```

Test the API:
```bash
python test_api.py
```

## 🔌 API Endpoints

### Chat
```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "ما هو مصدر الإفتاء في المملكة؟",
  "session_id": "optional-session-id",
  "mode": "rag"
}
```

### Search
```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "أحكام العقود",
  "k": 5,
  "dense_weight": 0.7,
  "sparse_weight": 0.3
}
```

### Health Check
```http
GET /health
```

## ⚙️ Configuration

Key settings in `.env`:

```env
# API Key (Required)
GOOGLE_API_KEY=your-api-key-here

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_TEMPERATURE=0.1

# RAG Settings
RAG_K=12                    # Number of documents to retrieve
RAG_DENSE_WEIGHT=0.7       # Weight for semantic search (FAISS)
RAG_SPARSE_WEIGHT=0.3      # Weight for keyword search (BM25)

# Server
PORT=8000
DEBUG=False
CORS_ORIGINS=http://localhost:5173,http://localhost:8081

# Database
FAISS_INDEX_PATH=data/m3_legal_faiss.index
LAWS_DATA_PATH=data/saudi_laws_scraped.json
```

## 🧪 Development

### Testing the Notebook
Open `rag_hybrid_retrieval_setup.ipynb` to:
- Test retrieval system
- Experiment with different models
- Evaluate performance metrics

### Running Tests
```bash
# Backend tests (when implemented)
pytest

# Frontend tests
cd law-UI
npm test
```

## 📊 Data

The system uses:
- **saudi_laws_scraped.json**: 16,371 hierarchical Saudi law documents
- **m3_legal_faiss.index**: Pre-built FAISS vector index using BGE-M3 embeddings
- **Evaluation datasets**: Curated QA pairs for testing and validation

### Data Statistics
- Total Documents: 16,371
- Embedding Model: BAAI/bge-m3
- Vector Dimensions: 1024
- Index Type: FAISS (Flat L2)

## 🎨 UI Features

- **Full RTL Support**: Native Arabic right-to-left layout
- **Custom Branding**: مسألة قانونية professional design
- **Orange Theme**: Vibrant #FFA629 color palette
- **Custom Icons**: Professional bot, user, and logo icons
- **Responsive Design**: Works on desktop and mobile
- **Real-time Status**: Live backend connection indicator
- **Smooth Animations**: Polished fade-in effects and transitions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **Saudi Bureau of Experts** for comprehensive law data
- **BAAI** for BGE-M3 multilingual embeddings
- **Google** for Gemini 2.5 Flash LLM capabilities
- **Meta** for FAISS vector search library
- **shadcn/ui** for beautiful React components
- **Vercel** for modern web tooling

## 📞 Support

For issues and questions, please open a GitHub issue.

## 🔗 Links

- Documentation: See `SETUP.md` for detailed setup instructions
- API Docs: `http://localhost:8000/docs` (when server is running)
- Frontend: `http://localhost:5173` (default dev server)

---

**مسألة قانونية** - Built with ❤️ for Saudi Arabian legal research
