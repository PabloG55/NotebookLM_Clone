from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from sqlalchemy.orm import Session
from typing import List
import uuid
import json

# Models and DB
from core.database import get_db, Notebook, Document, ChatMessage, Artifact
from core.storage_manager import save_raw_file, get_chroma_db_dir, delete_notebook_storage, get_notebook_subdir
import os
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
    notebooks = db.query(Notebook).filter(Notebook.hf_user_id == hf_user_id).order_by(Notebook.created_at.desc()).all()
    return [{"id": nb.notebook_id, "title": nb.title} for nb in notebooks]

@app.get("/api/notebooks/{notebook_id}/files")
def get_notebook_files(notebook_id: str, hf_user_id: str = Depends(verify_hf_user), db: Session = Depends(get_db)):
    """Fetch physical absolute paths of all uploaded raw files to render in Gradio"""
    notebook = db.query(Notebook).filter(Notebook.notebook_id == notebook_id, Notebook.hf_user_id == hf_user_id).first()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
        
    raw_dir = get_notebook_subdir(hf_user_id, notebook.notebook_id, "files_raw")
    if not os.path.exists(raw_dir):
        return []
    
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
    return files

@app.get("/api/notebooks/{notebook_id}/chats")
def get_notebook_chats(notebook_id: str, hf_user_id: str = Depends(verify_hf_user), db: Session = Depends(get_db)):
    """Fetch all history to hydrate Gradio ChatComponent"""
    notebook = db.query(Notebook).filter(Notebook.notebook_id == notebook_id, Notebook.hf_user_id == hf_user_id).first()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
        
    history_records = db.query(ChatMessage).filter(ChatMessage.notebook_id == notebook_id).order_by(ChatMessage.created_at).all()
    history = [{"role": msg.role, "content": msg.content} for msg in history_records]
    return history

@app.get("/api/notebooks/{notebook_id}/artifacts")
def get_notebook_artifacts(notebook_id: str, hf_user_id: str = Depends(verify_hf_user), db: Session = Depends(get_db)):
    """Fetch all generated artifacts for the notebook to hydrate UI tabs"""
    notebook = db.query(Notebook).filter(Notebook.notebook_id == notebook_id, Notebook.hf_user_id == hf_user_id).first()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
        
    artifacts = db.query(Artifact).filter(Artifact.notebook_id == notebook_id).all()
    # Return as a dictionary mapped by artifact_type for easy frontend lookup
    return {a.artifact_type: a.content for a in artifacts}

@app.post("/api/upload")
async def upload_document(
    notebook_name: str = Form(None),
    notebook_id: str = Form(None),
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
    notebook = None
    if notebook_id:
        notebook = db.query(Notebook).filter(Notebook.notebook_id == notebook_id, Notebook.hf_user_id == hf_user_id).first()
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook ID provided but not found.")
    elif notebook_name:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == hf_user_id, Notebook.title == notebook_name).first()
        if not notebook:
            notebook = Notebook(notebook_id=str(uuid.uuid4()), hf_user_id=hf_user_id, title=notebook_name)
            db.add(notebook)
            db.commit()
    else:
        raise HTTPException(status_code=400, detail="Must provide either notebook_name or notebook_id")

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

class RenameRequest(BaseModel):
    notebook_id: str
    new_title: str

@app.post("/api/notebooks/rename")
def rename_notebook(request: RenameRequest, hf_user_id: str = Depends(verify_hf_user), db: Session = Depends(get_db)):
    """Renames an existing notebook"""
    notebook = db.query(Notebook).filter(Notebook.notebook_id == request.notebook_id, Notebook.hf_user_id == hf_user_id).first()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found or unauthorized")
        
    notebook.title = request.new_title
    db.commit()
    return {"status": "success", "new_title": notebook.title}

class GenerateRequest(BaseModel):
    notebook_id: str
    artifact_type: str
    params: dict = {}

@app.post("/api/generate")
async def generate_artifact(request: GenerateRequest, hf_user_id: str = Depends(verify_hf_user), db: Session = Depends(get_db)):
    """Handles async generation of Summaries, Podcasts, Quizzes, and Study Guides from the full notebook text"""
    
    if request.artifact_type == "podcast_audio":
        from features.podcast import generate_podcast_audio
        from fastapi.responses import Response
        import base64
        
        # Audio generation just takes the parsed lines directly, it doesn't need to read the notebook from DB
        parsed_lines = request.params.get("parsed_lines")
        if not parsed_lines:
            raise HTTPException(status_code=400, detail="parsed_lines required for audio generation")
            
        # Check if we already have the audio synthesized for this specific notebook
        # Note: In a real app we might hash the transcript to detect changes, 
        # but for this portfolio piece mapping explicitly to notebook_id is fine.
        existing_audio = db.query(Artifact).filter(Artifact.notebook_id == request.notebook_id, Artifact.artifact_type == "podcast_audio").first()
        if existing_audio:
            try:
                # Return the decoded bytes natively
                audio_bytes = base64.b64decode(existing_audio.content)
                return Response(content=audio_bytes, media_type="audio/mpeg")
            except Exception as e:
                print("Failed to decode cached audio:", e)
        
        audio_bytes = await generate_podcast_audio(parsed_lines)
        
        # Save securely mapped to the user's notebook in SQLite using Base64
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=request.notebook_id, artifact_type="podcast_audio", content=base64_audio))
        try:
            db.commit()
        except BaseException as e:
            db.rollback()
            print("Failed caching audio to DB:", str(e))
        
        return Response(content=audio_bytes, media_type="audio/mpeg")

    # Verify Notebook Ownership
    notebook = db.query(Notebook).filter(Notebook.notebook_id == request.notebook_id, Notebook.hf_user_id == hf_user_id).first()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found or unauthorized")
        
    # Check if Artifact is already cached in database
    # For parameterized types (like modes), we append the param to the artifact_type for caching uniqueness
    cache_key = request.artifact_type
    if request.artifact_type == "summary":
        cache_key += f"_{request.params.get('mode', 'Brief').lower()}"
    elif request.artifact_type == "podcast_script":
        cache_key += f"_{request.params.get('num_exchanges', 12)}"
    elif request.artifact_type == "quiz":
        cache_key += f"_{request.params.get('num_questions', 5)}"
        
    existing_artifact = db.query(Artifact).filter(Artifact.notebook_id == request.notebook_id, Artifact.artifact_type == cache_key).first()
    if existing_artifact:
        # It's cached! Return it natively.
        if request.artifact_type in ["podcast_script", "quiz"]:
            return json.loads(existing_artifact.content)
        return {"result": existing_artifact.content}

    # Not cached. Reconstruct text by querying ChromaDB for all chunks
    chroma_dir = get_chroma_db_dir(hf_user_id, request.notebook_id)
    vstore = VectorStore(chroma_dir)
    chunks = vstore.collection.get()["documents"]
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Notebook has no processed text")
        
    full_text = " ".join(chunks)

    if request.artifact_type == "summary":
        from features.summarizer import summarize
        mode = request.params.get("mode", "Brief").lower()
        res = summarize(full_text, mode=mode)
        db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=request.notebook_id, artifact_type=cache_key, content=res))
        db.commit()
        return {"result": res}
        
    elif request.artifact_type == "podcast_script":
        from features.podcast import generate_podcast_script, parse_podcast_script
        num_exchanges = int(request.params.get("num_exchanges", 12))
        script_md = generate_podcast_script(full_text, num_exchanges)
        parsed_lines = parse_podcast_script(script_md)
        out_dict = {"script": script_md, "parsed_lines": parsed_lines}
        db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=request.notebook_id, artifact_type=cache_key, content=json.dumps(out_dict)))
        db.commit()
        return out_dict
        
    elif request.artifact_type == "quiz":
        from features.quiz import generate_quiz
        num_questions = int(request.params.get("num_questions", 5))
        quiz_data = generate_quiz(full_text, num_questions)
        out_dict = {"quiz": quiz_data}
        db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=request.notebook_id, artifact_type=cache_key, content=json.dumps(out_dict)))
        db.commit()
        return out_dict
        
    elif request.artifact_type == "study_guide":
        from features.study_guide import generate_study_guide
        study_guide = generate_study_guide(full_text)
        db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=request.notebook_id, artifact_type=cache_key, content=study_guide))
        db.commit()
        return {"result": study_guide}
        
    raise HTTPException(status_code=400, detail="Unknown artifact type")
