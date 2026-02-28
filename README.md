---
title: ThinkBook 🧠 (NotebookLM Clone)
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.12.0
python_version: "3.12"
app_file: app.py
hf_oauth: true
---

# ThinkBook 🧠

ThinkBook is a powerful, self-hosted **NotebookLM Clone** that lets you ground an AI in your own data. Upload documents, ingest URLs, and transform your notes into summaries, interactive quizzes, or even a full 2-person podcast with audio.

Built with **FastAPI**, **Gradio**, **ChromaDB**, and powered by **Groq** for lightning-fast inference.

---

## 🚀 Key Features

- **Document Ingestion:** Support for PDF, PPTX, TXT, and Web URLs.
- **Vector Search:** RAG (Retrieval-Augmented Generation) powered by ChromaDB.
- **AI Chat:** Instant answers based strictly on your notebook content.
- **Podcast Generation:** Converts documents into a realistic 2-person dialogue with high-quality TTS audio.
- **Study Tools:** AI-generated quizzes with grading and full study guides with flashcards.
- **User Persistence:** Integrates with Hugging Face OAuth for per-user data isolation and persistent storage.

---

## 🛠️ Local Installation

### Prerequisites

- **Python 3.12** (Recommended).
- A **Groq API Key** (Get it at [console.groq.com](https://console.groq.com/)).

### Setup

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd NotebookLM_Clone
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment:**
   Copy the example environment file and add your keys:

   ```bash
   cp .env.example .env
   # Edit .env and paste your GROQ_API_KEY
   ```

5. **Start the Application:**
   Run the startup script which launches both the FastAPI backend and the Gradio frontend:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

   - **Frontend:** http://localhost:7860
   - **API Backend:** http://localhost:8000

---

## ☁️ Hugging Face Deployment

Thinking of deploying to a Space? Follow these steps:

1. **Create a new Space** with the **Gradio SDK**.
2. **Add Secrets:** Go to Settings -> Variables and secrets:
   - `GROQ_API_KEY`: Your Groq Key.
   - `DATA_ROOT`: `/data` (If using persistence).
3. **Persistent Storage:** Enable "Free 20GB" storage in the Space settings to save your data permanently.
4. **Push your code:**
   ```bash
   git push origin main
   ```

---

## 🏗️ Architecture

- **`api.py`**: FastAPI backend handling database operations, file storage, and vector search.
- **`gradio_app.py`**: Modern, multi-tab frontend that communicates with the API.
- **`core/`**: Shared logic for ingestion, chunking, and LLM clients.
- **`features/`**: High-level modules for summarization, podcasts, and quizzes.

---

## 📝 License

MIT
