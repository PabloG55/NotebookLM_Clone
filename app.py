"""
NotebookLM Clone — Premium Multi-User Version
Gradio interface with HuggingFace OAuth and SQLite persistence.
"""
import fix_gradio  # patches gradio_client bug
import gradio as gr
import os
import json
import uuid
import tempfile
from dotenv import load_dotenv
from sqlalchemy.orm import Session

load_dotenv()

# Internal Modules
from core.database import SessionLocal, Notebook, Document, Artifact, ChatMessage
from core.storage_manager import (
    save_raw_file, 
    get_chroma_db_dir, 
    delete_notebook_storage, 
    get_notebook_subdir
)
from core.ingestion import ingest_source
from core.chunker import chunk_text
from core.vector_store import VectorStore
from core.groq_client import groq_stream

# Features
from features.summarizer import summarize
from features.chat import build_rag_messages
from features.podcast import generate_podcast_script, parse_podcast_script, generate_podcast_audio
from features.quiz import generate_quiz, check_answer
from features.study_guide import generate_study_guide

MAX_QUIZ_Q = 10

# ══════════════════════════════════════════════════════════════
# DATABASE UTILS
# ══════════════════════════════════════════════════════════════

def get_db():
    db = SessionLocal()
    try:
        return db
    except:
        db.close()
        raise

# ══════════════════════════════════════════════════════════════
# NOTEBOOK MANAGEMENT
# ══════════════════════════════════════════════════════════════

def fetch_notebooks(profile: gr.OAuthProfile | None):
    if not profile:
        return gr.Dropdown(choices=[], value=None)
    db = get_db()
    try:
        notebooks = db.query(Notebook).filter(Notebook.hf_user_id == profile.username).order_by(Notebook.created_at.desc()).all()
        choices = [nb.title for nb in notebooks]
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    finally:
        db.close()

def process_source(notebook_name, source_type, file_obj, url_text, is_append, profile: gr.OAuthProfile | None):
    if not profile:
        return "❌ Please log in with Hugging Face first.", gr.Dropdown()
    
    name = notebook_name.strip()
    if not name:
        return "❌ Please enter a notebook name.", gr.Dropdown()
    
    db = get_db()
    try:
        # Check if notebook exists
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == name).first()
        if is_append:
            if not notebook:
                return f"❌ '{name}' does not exist. Cannot append.", gr.Dropdown()
            nb_id = notebook.notebook_id
        else:
            if notebook:
                return f"❌ '{name}' already exists. Use a different name.", gr.Dropdown()
            # Create Notebook Record
            nb_id = str(uuid.uuid4())
            notebook = Notebook(notebook_id=nb_id, hf_user_id=profile.username, title=name)
            db.add(notebook)
            db.commit()

        # Process Content
        all_text = []
        source_name = ""

        if source_type == "Files (PDF / PPTX / TXT)":
            if not file_obj:
                return "❌ Please upload at least one file.", gr.Dropdown()
            files = file_obj if isinstance(file_obj, list) else [file_obj]
            for f in files:
                try:
                    fname = f.name.lower()
                    ftype = "pdf" if fname.endswith(".pdf") else ("pptx" if fname.endswith((".pptx", ".ppt")) else "txt")
                    with open(f.name, "rb") as fh:
                        raw_bytes = fh.read()
                    text = ingest_source(ftype, raw_bytes)
                    if text and len(text.strip()) > 20:
                        all_text.append(text)
                        # Save metadata
                        doc = Document(doc_id=str(uuid.uuid4()), notebook_id=nb_id, filename=os.path.basename(f.name), file_type=ftype)
                        db.add(doc)
                        save_raw_file(profile.username, nb_id, os.path.basename(f.name), raw_bytes)
                except Exception as e:
                    print(f"Skipping {f.name}: {e}")
            source_name = "Uploaded Files"
        else:
            if not url_text.strip():
                return "❌ Please enter a URL.", gr.Dropdown()
            raw_text = ingest_source("url", url_text.strip())
            if raw_text:
                all_text.append(raw_text)
                doc = Document(doc_id=str(uuid.uuid4()), notebook_id=nb_id, filename=url_text.strip(), file_type="url")
                db.add(doc)
                save_raw_file(profile.username, nb_id, "source_url.txt", url_text.strip().encode())
            source_name = url_text.strip()

        combined_text = "\n\n---\n\n".join(all_text)
        if not combined_text or len(combined_text.strip()) < 50:
            if not is_append:
                db.delete(notebook)
                db.commit()
            return "❌ Could not extract enough text.", gr.Dropdown()

        # Vectorize
        chunks = chunk_text(combined_text)
        chroma_dir = get_chroma_db_dir(profile.username, nb_id)
        store = VectorStore(chroma_dir)
        store.add_chunks(chunks, source_filename=source_name)
        
        db.commit()
        
        # Refresh list
        notebooks = db.query(Notebook).filter(Notebook.hf_user_id == profile.username).order_by(Notebook.created_at.desc()).all()
        choices = [nb.title for nb in notebooks]
        action = "appended to" if is_append else "added!"
        return f"✅ **{name}** {action} {len(chunks)} chunks processed.", gr.Dropdown(choices=choices, value=name)
    except Exception as e:
        db.rollback()
        return f"❌ Error: {e}", gr.Dropdown()
    finally:
        db.close()

def delete_notebook(notebook_name, profile: gr.OAuthProfile | None):
    if not profile or not notebook_name:
        return gr.Dropdown(), "❌ Unauthorized."
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == notebook_name).first()
        if notebook:
            nb_id = notebook.notebook_id
            db.delete(notebook)
            db.commit()
            delete_notebook_storage(profile.username, nb_id)
        
        notebooks = db.query(Notebook).filter(Notebook.hf_user_id == profile.username).order_by(Notebook.created_at.desc()).all()
        choices = [nb.title for nb in notebooks]
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None), "🗑️ Deleted."
    finally:
        db.close()

def rename_notebook(old_name, new_name, profile: gr.OAuthProfile | None):
    if not profile or not old_name:
        return gr.Dropdown(), "❌ Unauthorized."
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == old_name).first()
        if not notebook:
            return gr.Dropdown(), "❌ Notebook not found."
        notebook.title = new_name.strip()
        db.commit()
        
        notebooks = db.query(Notebook).filter(Notebook.hf_user_id == profile.username).order_by(Notebook.created_at.desc()).all()
        choices = [nb.title for nb in notebooks]
        return gr.Dropdown(choices=choices, value=new_name.strip()), gr.update(value=""), f"✅ Renamed to **{new_name.strip()}**"
    finally:
        db.close()

def get_notebook_info(notebook_name, profile: gr.OAuthProfile | None):
    if not profile or not notebook_name:
        return "_Please log in to see notebook stats_"
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == notebook_name).first()
        if not notebook:
            return "_Notebook not found_"
        
        chroma_dir = get_chroma_db_dir(profile.username, notebook.notebook_id)
        store = VectorStore(chroma_dir)
        count = store.collection.count()
        return f"📊 **{notebook_name}** · {count} context chunks indexed."
    finally:
        db.close()

# ══════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════

def chat_response(message, history, notebook_name, profile: gr.OAuthProfile | None):
    if not profile:
        return history + [{"role": "assistant", "content": "❌ Please log in first."}], ""
    if not notebook_name:
        return history + [{"role": "assistant", "content": "❌ Select a notebook first."}], ""
    
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == notebook_name).first()
        if not notebook: return history, ""
        
        # Load history from DB if Gradio history is empty
        if not history:
            msgs = db.query(ChatMessage).filter(ChatMessage.notebook_id == notebook.notebook_id).order_by(ChatMessage.created_at).all()
            history = [{"role": m.role, "content": m.content} for m in msgs]

        chroma_dir = get_chroma_db_dir(profile.username, notebook.notebook_id)
        store = VectorStore(chroma_dir)
        
        messages = build_rag_messages(message, store, history)
        full_response = ""
        for token in groq_stream(messages):
            full_response += token
            
        # Persistence
        db.add(ChatMessage(message_id=str(uuid.uuid4()), notebook_id=notebook.notebook_id, role="user", content=message))
        db.add(ChatMessage(message_id=str(uuid.uuid4()), notebook_id=notebook.notebook_id, role="assistant", content=full_response))
        db.commit()
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": full_response})
        return history, ""
    finally:
        db.close()

async def generate_audio_ui(lines_state, notebook_name, profile: gr.OAuthProfile | None):
    if not lines_state or not profile or not notebook_name:
        return None, "❌ Generate the podcast script first."
    
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == notebook_name).first()
        existing = db.query(Artifact).filter(Artifact.notebook_id == notebook.notebook_id, Artifact.artifact_type == "podcast_audio").first()
        
        import base64
        if existing:
            audio_bytes = base64.b64decode(existing.content)
        else:
            audio_bytes = await generate_podcast_audio(lines_state)
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=notebook.notebook_id, 
                            artifact_type="podcast_audio", content=base64_audio))
            db.commit()
            
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name, "✅ Audio ready!"
    except Exception as e:
        return None, f"❌ Audio error: {e}"
    finally:
        db.close()

def submit_quiz_ui(quiz_json, *answers):
    try:
        quiz = json.loads(quiz_json)
    except:
        return "❌ No quiz loaded."
    
    results = ""
    correct_count = 0
    for i, q in enumerate(quiz):
        user_ans = answers[i] if i < len(answers) else ""
        if not user_ans:
            results += f"**Q{i+1}:** ⚠️ Not answered\n\n"
            continue
        letter = user_ans[0] # Take first letter (A, B, C, or D)
        is_correct, explanation = check_answer(q, letter)
        if is_correct:
            correct_count += 1
            results += f"**Q{i+1}:** ✅ Correct! ({q['answer']})\n💡 _{explanation}_\n\n"
        else:
            results += f"**Q{i+1}:** ❌ Chose **{letter}**, correct: **{q['answer']}**\n💡 _{explanation}_\n\n"
            
    pct = int((correct_count / len(quiz)) * 100)
    results += f"\n---\n### Score: {correct_count}/{len(quiz)} ({pct}%)"
    return results

def get_full_text(notebook):
    chroma_dir = get_chroma_db_dir(notebook.hf_user_id, notebook.notebook_id)
    store = VectorStore(chroma_dir)
    chunks = store.collection.get()["documents"]
    return "\n\n".join(chunks)

def generate_summary_ui(notebook_name, mode, profile: gr.OAuthProfile | None):
    if not profile or not notebook_name: return "❌ Unauthorized."
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == notebook_name).first()
        cache_key = f"summary_{mode.lower()}"
        existing = db.query(Artifact).filter(Artifact.notebook_id == notebook.notebook_id, Artifact.artifact_type == cache_key).first()
        if existing: return existing.content
        
        text = get_full_text(notebook)
        res = summarize(text, mode=mode.lower())
        db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=notebook.notebook_id, artifact_type=cache_key, content=res))
        db.commit()
        return res
    finally:
        db.close()

def generate_podcast_ui(notebook_name, num_exchanges, profile: gr.OAuthProfile | None):
    if not profile or not notebook_name: return "❌ Unauthorized.", None
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == notebook_name).first()
        cache_key = f"podcast_script_{num_exchanges}"
        existing = db.query(Artifact).filter(Artifact.notebook_id == notebook.notebook_id, Artifact.artifact_type == cache_key).first()
        
        if existing:
            data = json.loads(existing.content)
            script_md, lines = data["script"], data["lines"]
        else:
            text = get_full_text(notebook)
            script_md = generate_podcast_script(text, int(num_exchanges))
            lines = parse_podcast_script(script_md)
            db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=notebook.notebook_id, 
                            artifact_type=cache_key, content=json.dumps({"script": script_md, "lines": lines})))
            db.commit()
            
        formatted = ""
        for speaker, line in lines:
            icon = "🎤" if speaker == "Alex" else "🎓"
            formatted += f"{icon} **{speaker}:** {line}\n\n"
        return formatted, lines
    finally:
        db.close()

def gen_quiz_ui(notebook_name, num_q, profile: gr.OAuthProfile | None):
    empty = [gr.update(visible=False) for _ in range(MAX_QUIZ_Q)]
    if not profile or not notebook_name: return ("❌ Login required", "[]", "", "", *empty)
    
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == notebook_name).first()
        cache_key = f"quiz_{num_q}"
        existing = db.query(Artifact).filter(Artifact.notebook_id == notebook.notebook_id, Artifact.artifact_type == cache_key).first()
        
        if existing:
            quiz = json.loads(existing.content)
        else:
            text = get_full_text(notebook)
            quiz = generate_quiz(text, num_questions=int(num_q))
            db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=notebook.notebook_id, artifact_type=cache_key, content=json.dumps(quiz)))
            db.commit()
            
        radio_updates = []
        for i in range(MAX_QUIZ_Q):
            if i < len(quiz):
                q = quiz[i]
                radio_updates.append(gr.update(choices=[f"A: {q['options']['A']}", f"B: {q['options']['B']}", f"C: {q['options']['C']}", f"D: {q['options']['D']}"], value=None, visible=True))
            else:
                radio_updates.append(gr.update(visible=False))
        return ("✅ Quiz ready!", json.dumps(quiz), render_quiz_md(quiz), "", *radio_updates)
    finally:
        db.close()

def render_quiz_md(quiz):
    out = ""
    for i, q in enumerate(quiz):
        out += f"**Q{i+1}. {q['question']}**\n"
        for l, opt in q['options'].items(): out += f"- **{l}:** {opt}\n"
        out += "\n"
    return out

def get_study_guide_ui(notebook_name, profile: gr.OAuthProfile | None):
    if not profile or not notebook_name: return "❌ Unauthorized."
    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == notebook_name).first()
        existing = db.query(Artifact).filter(Artifact.notebook_id == notebook.notebook_id, Artifact.artifact_type == "study_guide").first()
        if existing: return existing.content
        
        text = get_full_text(notebook)
        res = generate_study_guide(text)
        db.add(Artifact(artifact_id=str(uuid.uuid4()), notebook_id=notebook.notebook_id, artifact_type="study_guide", content=res))
        db.commit()
        return res
    finally:
        db.close()

def load_notebook_data(nb_name, profile: gr.OAuthProfile | None):
    empty_radios = [gr.update(visible=False, value=None) for _ in range(MAX_QUIZ_Q)]
    default_outputs = [
        "No notebook selected.",  # nb_info_md
        None,                     # nb_files_view
        [],                       # chatbot
        "",                       # sum_out
        "",                       # pod_script_out
        None,                     # pod_lines_state
        "",                       # quiz_display_md
        "{}",                     # quiz_json_box
        "",                       # study_out
        None,                     # audio_out
        ""                        # quiz_res_md
    ] + empty_radios

    if not nb_name or not profile:
        return tuple(default_outputs)

    db = get_db()
    try:
        notebook = db.query(Notebook).filter(Notebook.hf_user_id == profile.username, Notebook.title == nb_name).first()
        if not notebook:
            default_outputs[0] = "❌ Notebook not found."
            return tuple(default_outputs)
            
        nb_id = notebook.notebook_id
        
        chroma_dir = get_chroma_db_dir(profile.username, nb_id)
        store = VectorStore(chroma_dir)
        count = store.collection.count()
        info_md = f"📊 **{nb_name}** · {count} context chunks indexed."

        docs = db.query(Document).filter(Document.notebook_id == nb_id).all()
        raw_dir = get_notebook_subdir(profile.username, nb_id, "files_raw")
        files = []
        import os
        for d in docs:
            fpath = os.path.join(raw_dir, d.filename)
            if os.path.exists(fpath):
                files.append(fpath)
        if not files:
            files = None
        
        msgs = db.query(ChatMessage).filter(ChatMessage.notebook_id == nb_id).order_by(ChatMessage.created_at).all()
        history = [{"role": m.role, "content": m.content} for m in msgs]
        
        artifacts = db.query(Artifact).filter(Artifact.notebook_id == nb_id).all()
        art_dict = {a.artifact_type: a.content for a in artifacts}
        
        sum_val = art_dict.get("summary_brief", art_dict.get("summary_descriptive", ""))
        
        pod_key = next((k for k in art_dict.keys() if k.startswith("podcast_script_")), None)
        pod_script_val = ""
        pod_lines_val = None
        if pod_key:
            try:
                pod_data = json.loads(art_dict[pod_key])
                pod_script_val = pod_data.get("script", "")
                pod_lines_val = pod_data.get("lines", [])
            except:
                pass
                
        quiz_key = next((k for k in art_dict.keys() if k.startswith("quiz_")), None)
        quiz_val = []
        if quiz_key:
            try:
                quiz_val = json.loads(art_dict[quiz_key])
            except:
                pass
                
        quiz_json_val = json.dumps(quiz_val) if quiz_val else "{}"
        quiz_display = render_quiz_md(quiz_val) if quiz_val else ""
        
        quiz_radios = []
        for i in range(MAX_QUIZ_Q):
            if i < len(quiz_val):
                q = quiz_val[i]
                quiz_radios.append(gr.update(choices=[f"A: {q['options']['A']}", f"B: {q['options']['B']}", f"C: {q['options']['C']}", f"D: {q['options']['D']}"], value=None, visible=True))
            else:
                quiz_radios.append(gr.update(visible=False, value=None))

        study_val = art_dict.get("study_guide", "")
        
        audio_out_val = None
        audio_base64 = art_dict.get("podcast_audio")
        if audio_base64:
            import base64
            import tempfile
            try:
                audio_bytes = base64.b64decode(audio_base64)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tmp.write(audio_bytes)
                tmp.close()
                audio_out_val = tmp.name
            except Exception as e:
                print(f"Error loading audio: {e}")

        return (
            info_md,
            files,
            history,
            sum_val,
            pod_script_val,
            pod_lines_val,
            quiz_display,
            quiz_json_val,
            study_val,
            audio_out_val,
            "",
            *quiz_radios
        )
    finally:
        db.close()

# ══════════════════════════════════════════════════════════════
# UI DEFINITION
# ══════════════════════════════════════════════════════════════

css = """
#title { text-align: center; }
#title h1 { background: linear-gradient(90deg, #388bfd, #56d364); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.4rem; font-weight: 800; }
footer { display: none !important; }
"""

with gr.Blocks(title="ThinkBook 🧠", css=css) as demo:
    with gr.Row():
        gr.Markdown("# 🧠 ThinkBook\nMulti-user NotebookLM clone with full persistence.", elem_id="title")
        login_btn = gr.LoginButton()
    
    with gr.Row():
        active_nb = gr.Dropdown(choices=[], label="📚 Active Notebook", interactive=True, scale=4)
        with gr.Column(scale=2):
            nb_info_md = gr.Markdown("_Login to start_")
            with gr.Row():
                rename_in = gr.Textbox(placeholder="New name...", show_label=False, scale=3)
                rename_btn = gr.Button("✏️ Rename", size="sm", scale=1)
                delete_btn = gr.Button("🗑️ Delete", size="sm", scale=1, variant="stop")

    demo.load(fetch_notebooks, outputs=active_nb)

    # Note: the main hydration call is at the end of the file

    with gr.Tabs():
        # Notebooks Tab
        with gr.TabItem("📁 Management"):
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("### ➕ Create New Notebook")
                    nb_name = gr.Textbox(label="Notebook Name", placeholder="e.g. Biology Notes")
                    src_type1 = gr.Radio(["Files (PDF / PPTX / TXT)", "URL"], label="Source Type", value="Files (PDF / PPTX / TXT)")
                    file_in1 = gr.File(label="Upload Files", file_count="multiple")
                    url_in1 = gr.Textbox(label="URL", visible=False)
                    def toggle(t): return gr.update(visible=t=="Files (PDF / PPTX / TXT)", value=None), gr.update(visible=t=="URL", value="")
                    src_type1.change(toggle, inputs=src_type1, outputs=[file_in1, url_in1])
                    add_btn = gr.Button("🚀 Create Notebook", variant="primary")
                    add_status = gr.Markdown("_Add a notebook to begin_")
                    
                    def clear_file(): return None
                    
                with gr.Column(variant="panel"):
                    gr.Markdown("### 📎 Append to Active Notebook")
                    gr.Markdown("_Adds sources to the notebook currently selected in the top bar._")
                    src_type2 = gr.Radio(["Files (PDF / PPTX / TXT)", "URL"], label="Source Type", value="Files (PDF / PPTX / TXT)")
                    file_in2 = gr.File(label="Upload Files", file_count="multiple")
                    url_in2 = gr.Textbox(label="URL", visible=False)
                    src_type2.change(toggle, inputs=src_type2, outputs=[file_in2, url_in2])

                    append_btn = gr.Button("📎 Process & Append", variant="primary")
                    append_status = gr.Markdown()
                    

            with gr.Row():
                nb_files_view = gr.File(label="Files Currently in Notebook (Read-Only)", interactive=False)

        # Chat Tab
        with gr.TabItem("💬 Chat"):
            chatbot = gr.Chatbot(height=450, type="messages")
            with gr.Row():
                chat_in = gr.Textbox(placeholder="Ask about your document...", scale=5, show_label=False)
                send_btn = gr.Button("Send ➤", variant="primary")
            send_btn.click(chat_response, [chat_in, chatbot, active_nb], [chatbot, chat_in])
            chat_in.submit(chat_response, [chat_in, chatbot, active_nb], [chatbot, chat_in])

        # Feature Tabs... 
        with gr.TabItem("📝 Summary"):
            sum_mode = gr.Radio(["Brief", "Descriptive"], value="Brief", label="Style")
            sum_btn = gr.Button("✨ Generate")
            sum_out = gr.Markdown()
            def load_sum(): return "⏳ Generating Summary..."
            sum_btn.click(load_sum, None, sum_out).then(generate_summary_ui, [active_nb, sum_mode], sum_out)

        with gr.TabItem("🎙️ Podcast"):
            exchanges_sl = gr.Slider(8, 20, value=12, step=1, label="Exchanges")
            pod_btn = gr.Button("🎙️ Generate Script")
            pod_script_out = gr.Markdown()
            pod_lines_state = gr.State()
            def load_pod(): return "⏳ Generating Podcast Script...", None
            pod_btn.click(load_pod, None, [pod_script_out, pod_lines_state]).then(generate_podcast_ui, [active_nb, exchanges_sl], [pod_script_out, pod_lines_state])
            
            audio_btn = gr.Button("🔊 Generate Audio")
            audio_status = gr.Markdown()
            audio_out = gr.Audio(label="🎧 Listen")
            def load_audio(): return None, "⏳ Generating Audio (may take a minute)..."
            audio_btn.click(load_audio, None, [audio_out, audio_status]).then(generate_audio_ui, [pod_lines_state, active_nb], [audio_out, audio_status])

        with gr.TabItem("🧪 Quiz"):
            num_q_sl = gr.Slider(3, MAX_QUIZ_Q, value=5, step=1, label="Questions")
            quiz_gen_btn = gr.Button("🎲 Generate Quiz")
            quiz_json_box = gr.Textbox(visible=False, value="{}")
            quiz_status_md = gr.Markdown()
            quiz_display_md = gr.Markdown()
            ans_radios = [gr.Radio(choices=["A", "B", "C", "D"], label=f"Q{i+1}", visible=False) for i in range(MAX_QUIZ_Q)]
            submit_btn = gr.Button("✅ Submit Answers")
            quiz_res_md = gr.Markdown()
            
            def load_quiz(): return "⏳ Generating Quiz...", "{}", "", "", gr.update(visible=False)
            quiz_gen_btn.click(
                load_quiz, None, [quiz_status_md, quiz_json_box, quiz_display_md, quiz_res_md] + [ans_radios[0]]
            ).then(
                gen_quiz_ui, [active_nb, num_q_sl], [quiz_status_md, quiz_json_box, quiz_display_md, quiz_res_md] + ans_radios
            )
            submit_btn.click(submit_quiz_ui, [quiz_json_box] + ans_radios, quiz_res_md)

        with gr.TabItem("📚 Study Guide"):
            study_btn = gr.Button("📚 Generate")
            study_out = gr.Markdown()
            def load_study(): return "⏳ Generating Study Guide..."
            study_btn.click(load_study, None, study_out).then(get_study_guide_ui, [active_nb], study_out)

    
    # === WIRING HOISTS ===
    def _do_append(nb, st, fi, url, profile: gr.OAuthProfile | None): return process_source(nb, st, fi, url, True, profile)
    def _do_add(nb, st, fi, url, profile: gr.OAuthProfile | None): return process_source(nb, st, fi, url, False, profile)
    
    rename_btn.click(rename_notebook, [active_nb, rename_in], [active_nb, rename_in, nb_info_md])
    delete_btn.click(delete_notebook, [active_nb], [active_nb, nb_info_md])
    
    # Refresh notebook data logic
    active_nb.change(
        load_notebook_data, 
        inputs=[active_nb], 
        outputs=[nb_info_md, nb_files_view, chatbot, sum_out, pod_script_out, pod_lines_state, quiz_display_md, quiz_json_box, study_out, audio_out, quiz_res_md] + ans_radios
    )

    # Note: add_btn action requires the `notebook_name` input
    add_btn.click(_do_add, [nb_name, src_type1, file_in1, url_in1], [add_status, active_nb]).then(
        clear_file, None, file_in1
    ).then(
        load_notebook_data, inputs=[active_nb], outputs=[nb_info_md, nb_files_view, chatbot, sum_out, pod_script_out, pod_lines_state, quiz_display_md, quiz_json_box, study_out, audio_out, quiz_res_md] + ans_radios
    )
    
    append_btn.click(_do_append, [active_nb, src_type2, file_in2, url_in2], [append_status, active_nb]).then(
        clear_file, None, file_in2
    ).then(
        load_notebook_data, inputs=[active_nb], outputs=[nb_info_md, nb_files_view, chatbot, sum_out, pod_script_out, pod_lines_state, quiz_display_md, quiz_json_box, study_out, audio_out, quiz_res_md] + ans_radios
    )

if __name__ == "__main__":
    demo.launch()