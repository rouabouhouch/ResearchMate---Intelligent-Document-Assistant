# ResearchMate - Intelligent Document Assistant
![Uploading image.png…]()

## Overview
ResearchMate is a Retrieval-Augmented Generation (RAG) system that allows you to upload documents and ask questions about their content. It combines document processing with AI-powered question answering using Ollama's language models.

## Features
- **Document Upload**: Upload PDF and TXT documents
- **Smart Retrieval**: Finds relevant text chunks using embeddings
- **AI-Powered Answers**: Uses Ollama's LLama 3.1 model for responses
- **Document Management**: Track and manage uploaded documents
- **Fast Backend**: FastAPI backend with state persistence
- **Modern UI**: Streamlit-based interactive frontend

## Architecture
```
ResearchMate/
├── backend/           # FastAPI backend with RAG engine
├── frontend/         # Streamlit web interface
└── state files       # Persistent storage for documents
```

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama installed and running
- Required Ollama models pulled

### Step 1: Install Ollama and Models
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Step 2: Install Python Dependencies
```bash
# Backend dependencies
cd backend
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
pip install -r requirements.txt
```

### Step 3: Start the Application

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
streamlit run app.py
```

### Step 4: Access the Application
- Open your browser and go to: `http://localhost:8501`
- The backend API runs on: `http://localhost:8000`

## Project Structure

### Backend (`backend/`)
```
backend/
├── main.py              # FastAPI server with endpoints
├── rag_engine.py        # Core RAG implementation
├── requirements.txt     # Python dependencies
├── rag_state.pkl       # Persistent embeddings (auto-generated)
└── docs_state.pkl      # Document metadata (auto-generated)
```

### Frontend (`frontend/`)
```
frontend/
├── app.py              # Streamlit web interface
├── requirements.txt    # Frontend dependencies
└── .streamlit/         # Streamlit configuration
```

## API Endpoints

### Backend Endpoints
- `GET /` - API status and statistics
- `POST /upload` - Upload and process documents
- `POST /query` - Ask questions about documents
- `GET /documents` - List uploaded documents
- `GET /rag/stats` - Get RAG engine statistics
- `GET /rag/reset` - Reset RAG engine (testing)

### Example API Calls
```bash
# Check API status
curl http://localhost:8000/

# Upload a document
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload

# Query documents
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}' \
  http://localhost:8000/query
```

## Technology Stack

### Backend
- **FastAPI** - High-performance web framework
- **Ollama** - Local LLM inference
- **NumPy** - Numerical computations
- **Scikit-learn** - Cosine similarity calculations
- **PyMuPDF/PyPDF2** - PDF text extraction

### Frontend
- **Streamlit** - Interactive web application framework
- **Requests** - HTTP client for backend communication

### RAG Engine
- **Chunking**: Overlapping text chunks (400 words with 50 overlap)
- **Embeddings**: nomic-embed-text via Ollama
- **Retrieval**: Cosine similarity with top-k selection
- **State Management**: Pickle-based persistence

## How It Works

1. **Document Processing**
   - Upload PDF/TXT files
   - Extract text content
   - Split into overlapping chunks
   - Generate embeddings for each chunk

2. **Query Processing**
   - Generate embedding for user query
   - Find most similar document chunks using cosine similarity
   - Retrieve top-k relevant chunks

3. **Answer Generation**
   - Construct context from retrieved chunks
   - Send context and question to LLama 3.1 model
   - Generate natural language answer
   - Display answer with source citations

## Configuration

### Backend Configuration
- Embedding dimension: 768 (nomic-embed-text)
- Chunk size: 400 words
- Chunk overlap: 50 words
- Top-k retrieval: 5 chunks (configurable)

### Frontend Configuration
- Backend URL: `http://localhost:8000`
- Query timeout: 120 seconds
- Upload timeout: 60 seconds

## Troubleshooting

### Common Issues

1. **Backend not starting**
   - Check if Ollama is running: `ollama serve`
   - Verify models are pulled: `ollama list`
   - Check Python dependencies: `pip list`

2. **Frontend connection errors**
   - Ensure backend is running on port 8000
   - Check backend status: `curl http://localhost:8000/`

3. **Upload failures**
   - Check file size (should be < 100MB)
   - Verify PDF libraries are installed
   - Check backend logs for detailed errors

### Log Files
- Backend logs appear in the terminal where `main.py` is running
- Frontend logs appear in Streamlit's terminal output
- State files are saved as `.pkl` files in the backend directory

## Development

### Setting Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # Optional: code formatting and testing
```

### Running Tests
```bash
# Test backend endpoints
curl http://localhost:8000/test/rag
curl http://localhost:8000/test/ollama
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document public functions with docstrings


## Acknowledgments
- Built with FastAPI and Streamlit
- Uses Ollama for local LLM inference
- Inspired by modern RAG architectures

## Contact
For questions or issues, please check the project repository or open an issue in the issue tracker.
