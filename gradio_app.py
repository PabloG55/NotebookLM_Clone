"""
ThinkBook - NotebookLM Clone (Frontend Client)
Provides the Gradio UI and communicates with the FastAPI backend via HTTP requests.
"""
import fix_gradio
import gradio as gr
import requests
import json
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

def get_headers(profile: gr.OAuthProfile | None) -> dict:
    if not profile:
        return {}
    return {"X-HF-User": profile.username}

def fetch_notebooks(profile: gr.OAuthProfile | None):
    if not profile:
        return gr.Dropdown(choices=[], value=None)
    try:
        res = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile))
        if res.status_code == 200:
            notebooks = res.json()
            choices = [nb["title"] for nb in notebooks]
            # In a full impl we map names to IDs, but for Simplicity keeping titles here
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        print(f"Error fetching notebooks: {e}")
    return gr.Dropdown(choices=[], value=None)


def process_source(notebook_name, source_type, file_objs, url_text, profile: gr.OAuthProfile | None):
    if not profile:
        return "❌ Please log in with Hugging Face first.", gr.Dropdown()
        
    name = notebook_name.strip()
    if not name:
        return "❌ Please enter a notebook name.", gr.Dropdown()

    try:
        if source_type in ["PDF", "PPTX", "TXT"]:
            if not file_objs:
                return "❌ Please upload a file.", gr.Dropdown()
            
            # Use the first file for MVP presentation
            file_obj = file_objs[0] if isinstance(file_objs, list) else file_objs
            
            # Send to FastAPI
            with open(file_obj.name, "rb") as f:
                files = {"file": (os.path.basename(file_obj.name), f, "application/octet-stream")}
                data = {"notebook_name": name}
                
                res = requests.post(
                    f"{API_BASE_URL}/api/upload",
                    headers=get_headers(profile),
                    data=data,
                    files=files
                )
                
            if res.status_code == 200:
                body = res.json()
                return f"✅ **{name}** added! {body.get('chunks', 0)} chunks processed.", fetch_notebooks(profile)
            else:
                return f"❌ Server Error: {res.text}", fetch_notebooks(profile)
        else:
            return "❌ URL ingest not implemented in API yet.", fetch_notebooks(profile)
            
    except Exception as e:
        return f"❌ Error: {str(e)}", fetch_notebooks(profile)


def chat_response(message, history, notebook_name, profile: gr.OAuthProfile | None):
    if not message.strip():
        yield history, ""
        return

    history = history or []

    if not profile:
        history.append({"role": "assistant", "content": "❌ Please log in first."})
        yield history, ""
        return

    if not notebook_name:
        history.append({"role": "assistant", "content": "❌ Please select a notebook first."})
        yield history, ""
        return

    # To connect names to IDs properly we'd fetch the notebook list here
    # Assuming notebook_name is actually the notebook_id or handled by the backend
    try:
        # First get the notebook ID from the name
        res_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile))
        notebook_id = None
        if res_nbs.status_code == 200:
            for nb in res_nbs.json():
                if nb["title"] == notebook_name:
                    notebook_id = nb["id"]
                    break
        
        if not notebook_id:
            history.append({"role": "assistant", "content": "❌ Notebook not found on server."})
            yield history, ""
            return

        payload = {
            "notebook_id": notebook_id,
            "message": message
        }
        res = requests.post(f"{API_BASE_URL}/api/chat", headers=get_headers(profile), json=payload)
        
        if res.status_code == 200:
            ans = res.json().get("response", "")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ans})
            yield history, ""
        else:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Error: {res.text}"})
            yield history, ""

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error: {e}"})
        yield history, ""

def clear_chat():
    return [], ""

# ==========================================
# UI Build
# ==========================================
with gr.Blocks(title="ThinkBook 🧠") as demo:
    # Header
    with gr.Row():
        gr.Markdown(
            "# 🧠 ThinkBook\nUpload any document · Chat · Summarize · Podcast · Quiz · Study Guide",
            elem_id="title"
        )
        login_btn = gr.LoginButton()

    # Global notebook selector bar
    with gr.Row():
        active_nb = gr.Dropdown(choices=[], label="📚 Active Notebook", interactive=True, scale=4)
        nb_info_md = gr.Markdown("_Login and select a notebook_")

    def get_notebook_info(nb_name):
        return f"Selected: **{nb_name}**" if nb_name else "No notebook selected."
        
    active_nb.change(get_notebook_info, inputs=active_nb, outputs=nb_info_md)

    gr.Markdown("---")

    with gr.Tabs():
        # ── TAB 1: NOTEBOOKS ─────────────────────────────────────────────────
        with gr.TabItem("📁 Notebooks"):
            gr.Markdown("### ➕ Add New Notebook")
            with gr.Row():
                with gr.Column():
                    nb_name = gr.Textbox(label="Notebook Name", placeholder="e.g. Biology Notes")
                    src_type = gr.Radio(["PDF", "PPTX", "TXT", "URL"], label="Source Type", value="PDF")
                    file_in = gr.File(label="Upload Files", file_types=[".pdf",".pptx",".ppt",".txt",".md"], file_count="multiple")
                    url_in = gr.Textbox(label="URL", placeholder="https://...", visible=False)

                    def toggle(t):
                        return gr.File(visible=t != "URL", file_count="multiple"), gr.Textbox(visible=t == "URL")
                    src_type.change(toggle, inputs=src_type, outputs=[file_in, url_in])

                    add_btn = gr.Button("🚀 Process & Add", variant="primary")

                with gr.Column():
                    add_status = gr.Markdown("_Upload a source to begin._")

            add_btn.click(process_source, inputs=[nb_name, src_type, file_in, url_in], outputs=[add_status, active_nb])

        # ── TAB 2: CHAT ──────────────────────────────────────────────────────
        with gr.TabItem("💬 Chat"):
            gr.Markdown("### Ask anything about your document")
            chatbot = gr.Chatbot(label="ThinkBook AI", height=450)
            with gr.Row():
                chat_in = gr.Textbox(placeholder="Ask a question...", label="", scale=5, show_label=False)
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clr_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            send_btn.click(chat_response, inputs=[chat_in, chatbot, active_nb], outputs=[chatbot, chat_in])
            chat_in.submit(chat_response, inputs=[chat_in, chatbot, active_nb], outputs=[chatbot, chat_in])
            clr_btn.click(clear_chat, outputs=[chatbot, chat_in])

    # Trigger load when page opens to fetch profile and notebooks
    demo.load(fetch_notebooks, inputs=None, outputs=active_nb)

if __name__ == "__main__":
    demo.launch()
