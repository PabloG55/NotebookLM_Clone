import os
import shutil
import uuid

# Base path for all persistent data
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
USERS_DIR = os.path.join(DATA_ROOT, "users")

def _get_user_dir(hf_user_id: str) -> str:
    """Returns the base directory for a specific user: /data/users/<username>"""
    return os.path.join(USERS_DIR, hf_user_id)

def get_notebook_dir(hf_user_id: str, notebook_id: str) -> str:
    """Returns the base directory for a specific notebook: /data/users/<username>/notebooks/<uuid>"""
    return os.path.join(_get_user_dir(hf_user_id), "notebooks", notebook_id)

def get_notebook_subdir(hf_user_id: str, notebook_id: str, subname: str) -> str:
    """
    Returns and ensures existence of a specific subdirectory within a notebook.
    Examples of subname: 'files_raw', 'files_extracted', 'chroma', 'artifacts/quizzes'
    """
    path = os.path.join(get_notebook_dir(hf_user_id, notebook_id), subname)
    os.makedirs(path, exist_ok=True)
    return path

def save_raw_file(hf_user_id: str, notebook_id: str, filename: str, file_bytes: bytes) -> str:
    """Saves raw uploaded bytes into the files_raw directory."""
    dir_path = get_notebook_subdir(hf_user_id, notebook_id, "files_raw")
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "wb") as f:
        f.write(file_bytes)
    return filepath

def save_extracted_text(hf_user_id: str, notebook_id: str, filename: str, text: str) -> str:
    """Saves extracted text into the files_extracted directory."""
    dir_path = get_notebook_subdir(hf_user_id, notebook_id, "files_extracted")
    filepath = os.path.join(dir_path, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath

def delete_notebook_storage(hf_user_id: str, notebook_id: str) -> bool:
    """Recursively deletes a notebook's entire directory structure."""
    path = get_notebook_dir(hf_user_id, notebook_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    return False

def get_chroma_db_dir(hf_user_id: str, notebook_id: str) -> str:
    """Returns the persistent directory for this notebook's ChromaDB instance."""
    return get_notebook_subdir(hf_user_id, notebook_id, "chroma")
