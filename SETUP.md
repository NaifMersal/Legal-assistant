# 🚀 Quick Setup Guide

## Step 1: Backend Setup (5 minutes)

### 1.1 Set up your Google API Key

1. **Get your API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Create `.env` file**:
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` and add your key**:
   ```env
   GOOGLE_API_KEY=your-actual-api-key-here
   ```

### 1.2 Start the Backend Server

**Option A: Using the startup script (Recommended)**
```bash
chmod +x start_server.sh
./start_server.sh
```

**Option B: Manual setup**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

✅ **Backend should now be running at:**
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

---

## Step 2: Frontend Setup (3 minutes)

### 2.1 Install Dependencies

```bash
cd law-UI
npm install
```

### 2.2 Start Development Server

```bash
npm run dev
```

✅ **Frontend should now be running at:**
- UI: `http://localhost:5173`

---

## Step 3: Test the System

1. **Open your browser** to `http://localhost:5173`

2. **Try asking a question** (in Arabic or English):
   - "ما هو مصدر الإفتاء في المملكة؟"
   - "What are the contract rules in Saudi law?"

3. **Check API docs** at `http://localhost:8000/docs`
   - Try the `/api/v1/chat` endpoint
   - Try the `/api/v1/search` endpoint

---

## 🔧 Troubleshooting

### Backend won't start?

1. **Check Python version**: `python3 --version` (need 3.9+)
2. **Check API key**: Make sure `GOOGLE_API_KEY` is set in `.env`
3. **Check dependencies**: `pip install -r requirements.txt`
4. **Check data files**:
   - `data/m3_legal_faiss.index` exists?
   - `data/saudi_laws_scraped.json` exists?

### Frontend won't start?

1. **Check Node version**: `node --version` (need 18+)
2. **Clear cache**: `rm -rf node_modules && npm install`
3. **Check backend**: Is backend running on port 8000?

### API returns errors?

1. **Check logs** in the terminal where backend is running
2. **Verify API key** is correct in `.env`
3. **Check health endpoint**: `curl http://localhost:8000/health`

---

## 📝 Next Steps

### Connect Frontend to Backend (Step 2 of our plan)

Once both are running, we'll update the frontend to:
1. Call the real API instead of simulated responses
2. Display source citations
3. Handle streaming responses
4. Add chat history management

Ready to proceed? Let me know! 🚀
