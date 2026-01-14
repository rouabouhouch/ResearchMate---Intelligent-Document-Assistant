import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import ollama
import pickle
import atexit
import sys
import numpy as np  # ADD THIS


# Import your RAG Engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.rag_engine import RAGEngine

# ==================== FastAPI App ====================
app = FastAPI(title="ResearchMate API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
os.makedirs("uploads", exist_ok=True)

# State persistence
RAG_STATE_FILE = "rag_state.pkl"
DOCS_STATE_FILE = "docs_state.pkl"

# Initialize global components
documents = []  # Store document metadata
rag_engine = RAGEngine(embedding_dim=768)  # Using YOUR RAG Engine

def load_state():
    """Load RAG state from disk"""
    global documents, rag_engine
    
    # Load documents
    if os.path.exists(DOCS_STATE_FILE):
        try:
            with open(DOCS_STATE_FILE, "rb") as f:
                documents = pickle.load(f)
            print(f"📂 Loaded {len(documents)} documents")
        except Exception as e:
            print(f"⚠️ Failed to load documents: {e}")
    
    # Load RAG engine state
    if os.path.exists(RAG_STATE_FILE):
        try:
            with open(RAG_STATE_FILE, "rb") as f:
                state = pickle.load(f)
            
            # Restore RAG engine state
            rag_engine.chunks = state.get("chunks", [])
            rag_engine.embeddings = state.get("embeddings", 
                np.empty((0, rag_engine.embedding_dim), dtype=np.float32))
            rag_engine.embedding_cache = state.get("embedding_cache", {})
            
            print(f"📂 Loaded RAG state: {len(rag_engine.chunks)} chunks")
        except Exception as e:
            print(f"⚠️ Failed to load RAG state: {e}")
    else:
        print("📝 No previous RAG state found, starting fresh")

def save_state():
    """Save RAG state to disk"""
    try:
        # Save documents
        with open(DOCS_STATE_FILE, "wb") as f:
            pickle.dump(documents, f)
        
        # Save RAG engine state
        state = {
            "chunks": rag_engine.chunks,
            "embeddings": rag_engine.embeddings,
            "embedding_cache": rag_engine.embedding_cache
        }
        with open(RAG_STATE_FILE, "wb") as f:
            pickle.dump(state, f)
        
        print(f"💾 State saved: {len(documents)} docs, {len(rag_engine.chunks)} chunks")
    except Exception as e:
        print(f"⚠️ Failed to save state: {e}")

# Load existing state
load_state()

# Register save on exit
atexit.register(save_state)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 500

# ==================== API Endpoints ====================
@app.get("/")
def home():
    """API root endpoint"""
    return {
        "message": "ResearchMate API with RAG Engine",
        "version": "1.0.0",
        "documents_count": len(documents),
        "rag_stats": rag_engine.get_stats()
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document with RAG integration"""
    print(f"\n📤 Uploading file: {file.filename} ({file.content_type})")
    
    try:
        # Read file content
        content_bytes = await file.read()
        print(f"  File size: {len(content_bytes):,} bytes")
        
        # Extract text based on file type
        content = ""
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith('.txt') or file.content_type == 'text/plain':
            content = content_bytes.decode('utf-8', errors='ignore')
            print(f"  Text file, extracted {len(content):,} characters")
        
        elif filename_lower.endswith('.pdf') or file.content_type == 'application/pdf':
            try:
                # Try PyMuPDF (fitz) first
                import fitz
                pdf_doc = fitz.open(stream=content_bytes, filetype="pdf")
                for page in pdf_doc:
                    content += page.get_text()
                pdf_doc.close()
                print(f"  PDF processed with PyMuPDF, extracted {len(content):,} characters")
            except ImportError:
                try:
                    # Fallback to PyPDF2
                    import PyPDF2
                    from io import BytesIO
                    pdf_file = BytesIO(content_bytes)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
                    print(f"  PDF processed with PyPDF2, extracted {len(content):,} characters")
                except ImportError:
                    content = f"[PDF file: {file.filename} - install PyPDF2 or PyMuPDF for text extraction]"
                    print("  ⚠️  No PDF library installed")
        
        else:
            # For other files, try to decode as text
            try:
                content = content_bytes.decode('utf-8', errors='ignore')
                print(f"  Other file type, decoded {len(content):,} characters")
            except:
                content = f"[Binary file: {file.filename} - size: {len(content_bytes):,} bytes]"
                print("  ⚠️  Could not decode as text")
        
        # Store document metadata
        doc_id = len(documents)
        doc_metadata = {
            "id": doc_id,
            "filename": file.filename,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "size_bytes": len(content_bytes),
            "content_length": len(content),
            "uploaded_at": "now"
        }
        documents.append(doc_metadata)
        
        # Add to RAG engine (YOUR optimized version)
        chunks_count = rag_engine.add_document(
            doc_id=doc_id,
            text=content
        )
        
        # Save state immediately
        save_state()
        
        return {
            "status": "success",
            "filename": file.filename,
            "doc_id": doc_id,
            "content_length": len(content),
            "chunks_created": chunks_count,
            "total_chunks": len(rag_engine.chunks),
            "message": "Document uploaded and processed with RAG"
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Upload error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "filename": file.filename
        }

@app.get("/documents")
def get_documents_list():
    """List all uploaded documents with metadata"""
    return {
        "total_documents": len(documents),
        "documents": documents,
        "rag_stats": rag_engine.get_stats()
    }

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query across documents using RAG retrieval and Ollama"""
    print(f"\n❓ Query received: '{request.question}'")
    
    import time
    start_time = time.time()
    
    try:
        # 1. Retrieve relevant chunks using RAG (YOUR optimized version)
        retrieved_chunks = rag_engine.retrieve(
            query=request.question,
            top_k=request.top_k
        )
        
        print(f"📊 Retrieved {len(retrieved_chunks)} chunks")
        
        if not retrieved_chunks:
            print("⚠️  No relevant chunks found")
            context = ""
            sources = []
        else:
            # Build context from retrieved chunks
            context_parts = []
            sources = []
            for chunk in retrieved_chunks:
                # Find the filename for this doc_id
                filename = "Unknown"
                for doc in documents:
                    if doc["id"] == chunk["doc_id"]:
                        filename = doc["filename"]
                        break
                
                context_parts.append(f"[From {filename}]:\n{chunk['chunk']}")
                sources.append({
                    "filename": filename,
                    "score": chunk['score'],
                    "chunk_index": chunk['chunk_index']
                })
            
            context = "\n\n" + "\n\n".join(context_parts) + "\n\n"
        
        # 2. Prepare prompt for Ollama
        if context:
            prompt = f"""Based on these documents:

{context}

Question: {request.question}

Please answer based only on the documents above. If the information isn't there, say so.

Answer:"""
        else:
            prompt = f"""Question: {request.question}

Please provide a helpful answer based on your general knowledge.

Answer:"""
        
        print(f"📝 Prompt length: {len(prompt):,} characters")
        
        # 3. Call Ollama for generation
        print(f"🤖 Calling Ollama...")
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful research assistant. Be concise and accurate."
                },
                {"role": "user", "content": prompt}
            ],
            options={
                'temperature': request.temperature,
                'num_predict': request.max_tokens
            }
        )
        
        answer = response['message']['content']
        
        # 4. Calculate response time
        response_time = time.time() - start_time
        
        print(f"✅ Response generated in {response_time:.2f}s")
        print(f"📄 Answer length: {len(answer):,} characters")
        
        return {
            "status": "success",
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved_chunks),
            "response_time": f"{response_time:.2f}s",
            "context_used": len(context) > 0,
            "model": "llama3.1:8b"
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Query error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": str(e),
            "answer": "An error occurred while processing your query."
        }

@app.get("/rag/stats")
def get_rag_stats():
    """Get RAG engine statistics"""
    return rag_engine.get_stats()

@app.get("/rag/reset")
def reset_rag():
    """Reset RAG engine (for testing)"""
    global rag_engine, documents
    
    # Reset RAG engine
    rag_engine = RAGEngine(embedding_dim=768)
    documents = []
    
    # Remove state files
    for state_file in [RAG_STATE_FILE, DOCS_STATE_FILE]:
        if os.path.exists(state_file):
            os.remove(state_file)
    
    print("🔄 RAG engine reset")
    return {"status": "success", "message": "RAG engine reset"}

@app.get("/test/ollama")
def test_ollama():
    """Test Ollama connection"""
    try:
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{"role": "user", "content": "Say hello in one word."}]
        )
        
        return {
            "status": "success",
            "response": response['message']['content'],
            "model": "llama3.1:8b"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test/rag")
def test_rag():
    """Test RAG engine functionality"""
    try:
        # Add a test document
        test_doc_id = len(documents)
        test_text = "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Deep learning is a type of machine learning that uses neural networks with many layers."
        
        chunks_added = rag_engine.add_document(
            doc_id=test_doc_id,
            text=test_text
        )
        
        # Test retrieval
        retrieved = rag_engine.retrieve("What is machine learning?", top_k=2)
        
        return {
            "status": "success",
            "chunks_added": chunks_added,
            "retrieved_chunks": len(retrieved),
            "rag_stats": rag_engine.get_stats()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==================== Main Execution ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ResearchMate API with Optimized RAG Engine")
    print("=" * 60)
    print(f"\n📂 Using RAG Engine from: backend.rag_engine")
    print(f"📊 Current stats: {rag_engine.get_stats()}")
    
    print("\n📦 Prerequisites:")
    print("  1. Make sure Ollama is running")
    print("  2. Models pulled: ollama pull llama3.1:8b nomic-embed-text")
    
    print("\n🌐 API Endpoints:")
    print("  GET  /                 - API status")
    print("  POST /upload           - Upload documents")
    print("  POST /query            - Ask questions with RAG")
    print("  GET  /documents        - List uploaded docs")
    print("  GET  /rag/stats        - RAG engine statistics")
    print("  GET  /test/rag         - Test RAG engine")
    print("=" * 60)
    print("\nStarting server...")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="info"
    )