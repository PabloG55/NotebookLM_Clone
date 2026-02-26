import os
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone

# Ensure the parent directory exists before creating the database
DB_PATH = os.environ.get("DATABASE_URL", "sqlite:///./data/database.sqlite")
if DB_PATH.startswith("sqlite:///"):
    # Extract local path from sqlite URL
    local_path = DB_PATH.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Notebook(Base):
    __tablename__ = "notebooks"

    notebook_id = Column(String, primary_key=True, index=True) # Will use UUID strings
    hf_user_id = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    documents = relationship("Document", back_populates="notebook", cascade="all, delete-orphan")
    messages = relationship("ChatMessage", back_populates="notebook", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, index=True)
    notebook_id = Column(String, ForeignKey("notebooks.notebook_id"), nullable=False)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    notebook = relationship("Notebook", back_populates="documents")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    message_id = Column(String, primary_key=True, index=True)
    notebook_id = Column(String, ForeignKey("notebooks.notebook_id"), nullable=False)
    role = Column(String, nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    notebook = relationship("Notebook", back_populates="messages")

# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
