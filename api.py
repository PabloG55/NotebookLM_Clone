from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from sqlalchemy.orm import Session
from typing import List
import uuid

# Models and DB
from core.database import get_db, Notebook, Document, ChatMessage
from core.storage_manager import save_raw_file, get_chroma_db_dir, delete_notebook_storage
from core.vector_store import VectorStore
# Specific feature logic
from core.chunker import chunk_text
from core.ingestion import ingest_source

app = FastAPI(title="NotebookLM API Layer")

def verify_hf_user(x_hf_user: str = Header(None)) -> str:
    """Extracts HF user ID from headers. Sent by the Gradio frontend."""
    if not x_hf_user:
        raise HTTPException(status_code=401, detail="Missing X-HF-User header. Please log in.")
    return x_hf_user

@app.get("/api/notebooks")
def list_notebooks(hf_user_id: str = Depends(verify_hf_user), db: Session = Depends(get_db)):
    """Fetch all notebooks for the authenticated user"""
    notebooks = db.query(Notebook).filter(Notebook.hf_user_id == hf_user_id).all()
    return [{"id": nb.notebook_id, "title": nb.title} for nb in notebooks]

@app.post("/api/upload")
async def upload_document(
    notebook_name: str = Form(...),
    file: UploadFile = File(...),
    hf_user_id: str = Depends(verify_hf_user),
    db: Session = Depends(get_db)
):
    """
    Handles PDF ingest.
    1. Check if notebook exists or make new one
    2. Save raw file bytes
    3. Extract text
    4. Chunk and vectorize into ChromaDB
    5. Save Database metadata
    """
    if not file.filename.endswith((".pdf", ".pptx", ".txt")):
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    # 1. Find or create notebook
    notebook = db.query(Notebook).filter(Notebook.hf_user_id == hf_user_id, Notebook.title == notebook_name).first()
    if not notebook:
        notebook = Notebook(notebook_id=str(uuid.uuid4()), hf_user_id=hf_user_id, title=notebook_name)
        db.add(notebook)
        db.commit()

    # 2. Extract Raw Bytes & Vectorize
    raw_bytes = await file.read()
    raw_text = ingest_source(file.filename.split('.')[-1].lower(), raw_bytes)
    
    if not raw_text or len(raw_text.strip()) < 50:
         raise HTTPException(status_code=400, detail="Could not extract enough text from file.")

    chunks = chunk_text(raw_text)
    
    # 3. Store Vectors in specific Chromadb folder
    chroma_dir = get_chroma_db_dir(hf_user_id, notebook.notebook_id)
    vstore = VectorStore(chroma_dir)
    vstore.add_chunks(chunks)

    # 4. Save metadata to DB
    doc = Document(
        doc_id=str(uuid.uuid4()),
        notebook_id=notebook.notebook_id,
        filename=file.filename,
        file_type=file.filename.split('.')[-1].lower(),
        chunk_count=len(chunks)
    )
    db.add(doc)
    db.commit()

    # 5. Persist original file (optional, follows architecture tree)
    save_raw_file(hf_user_id, notebook.notebook_id, file.filename, raw_bytes)

    return {"status": "success", "notebook_id": notebook.notebook_id, "chunks": len(chunks)}


from pydantic import BaseModel
class ChatRequest(BaseModel):
    notebook_id: str
    message: str

@app.post("/api/chat")
def chat(request: ChatRequest, hf_user_id: str = Depends(verify_hf_user), db: Session = Depends(get_db)):
    """Handles chat request, verifies ownership, vectors searches, and asks LLM"""
    
    # Verify Notebook Ownership
    notebook = db.query(Notebook).filter(Notebook.notebook_id == request.notebook_id, Notebook.hf_user_id == hf_user_id).first()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found or unauthorized")

    # Fetch History from DB to build prompt
    history_records = db.query(ChatMessage).filter(ChatMessage.notebook_id == request.notebook_id).order_by(ChatMessage.created_at).all()
    history = [{"role": msg.role, "content": msg.content} for msg in history_records]
    
    # Connect to vector store
    chroma_dir = get_chroma_db_dir(hf_user_id, request.notebook_id)
    vstore = VectorStore(chroma_dir)
    
    from features.chat import build_rag_messages
    from core.groq_client import groq_stream
    
    messages = build_rag_messages(request.message, vstore, history)
    
    full_response = ""
    for token in groq_stream(messages, temperature=0.6, max_tokens=2048):
        full_response += token

    # Log new messages to database
    db.add(ChatMessage(message_id=str(uuid.uuid4()), notebook_id=request.notebook_id, role="user", content=request.message))
    db.add(ChatMessage(message_id=str(uuid.uuid4()), notebook_id=request.notebook_id, role="assistant", content=full_response))
    db.commit()

    return {"response": full_response}
