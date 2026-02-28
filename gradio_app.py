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

def fetch_notebooks_with_selection(profile: gr.OAuthProfile | None, selected_title: str | None = None):
    if not profile:
        return gr.Dropdown(choices=[], value=None)
    try:
        res = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile))
        if res.status_code == 200:
            notebooks = res.json()
            choices = [nb["title"] for nb in notebooks]
            target_val = selected_title if selected_title in choices else (choices[0] if choices else None)
            return gr.Dropdown(choices=choices, value=target_val)
    except Exception as e:
        print(f"Error fetching notebooks: {e}")
    return gr.Dropdown(choices=[], value=None)

def fetch_notebooks(profile: gr.OAuthProfile | None):
    return fetch_notebooks_with_selection(profile)


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
                
                # Resolve ID if this notebook exists (needed for Append flow)
                res_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile))
                if res_nbs.status_code == 200:
                    notebook_id = next((nb["id"] for nb in res_nbs.json() if nb["title"] == name), None)
                    if notebook_id:
                        data["notebook_id"] = notebook_id
                
                res = requests.post(
                    f"{API_BASE_URL}/api/upload",
                    headers=get_headers(profile),
                    data=data,
                    files=files
                )
                
            if res.status_code == 200:
                body = res.json()
                return f"✅ **{name}** added! {body.get('chunks', 0)} chunks processed.", fetch_notebooks_with_selection(profile, name)
            else:
                return f"❌ Server Error: {res.text}", fetch_notebooks_with_selection(profile, name)
        else:
            return "❌ URL ingest not implemented in API yet.", fetch_notebooks_with_selection(profile, name)
            
    except Exception as e:
        return f"❌ Error: {str(e)}", fetch_notebooks_with_selection(profile, name)


def chat_response(message, history, notebook_name, profile: gr.OAuthProfile | None):
    if not message.strip():
        yield history, ""
        return

    history = history or []

    if not profile:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ Please log in first."})
        yield history, ""
        return

    if not notebook_name:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ Please select a notebook first."})
        yield history, ""
        return

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
            history.append({"role": "user", "content": message})
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

def generate_summary(notebook_name, mode, profile: gr.OAuthProfile | None):
    if not profile or not notebook_name: return "❌ Log in and select a notebook."
    res_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile)).json()
    nb_id = next((nb["id"] for nb in res_nbs if nb["title"] == notebook_name), None)
    if not nb_id: return "❌ Notebook not found."
    
    res = requests.post(f"{API_BASE_URL}/api/generate", headers=get_headers(profile), json={"notebook_id": nb_id, "artifact_type": "summary", "params": {"mode": mode}})
    return res.json().get("result", f"❌ Error: {res.text}")

def generate_podcast(notebook_name, exchanges, profile: gr.OAuthProfile | None):
    if not profile or not notebook_name: return "❌ Log in and select a notebook.", None
    res_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile)).json()
    nb_id = next((nb["id"] for nb in res_nbs if nb["title"] == notebook_name), None)
    if not nb_id: return "❌ Notebook not found.", None
    
    res = requests.post(f"{API_BASE_URL}/api/generate", headers=get_headers(profile), json={"notebook_id": nb_id, "artifact_type": "podcast_script", "params": {"num_exchanges": exchanges}})
    if res.status_code != 200: return f"❌ Error: {res.text}", None
    d = res.json()
    return d.get("script", ""), d.get("parsed_lines", [])

def generate_audio(parsed_lines, profile: gr.OAuthProfile | None):
    if not parsed_lines: return None, "❌ No script generated yet."
    if not profile: return None, "❌ Log in to hear audio."
    # We fake the notebook ID here just to pass the auth structure since the audio endpoint uses the script
    res = requests.post(f"{API_BASE_URL}/api/generate", headers=get_headers(profile), json={"notebook_id": "dummy", "artifact_type": "podcast_audio", "params": {"parsed_lines": parsed_lines}})
    if res.status_code != 200: return None, f"❌ Error: {res.text}"
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(res.content)
        return f.name, "✅ Audio ready!"

MAX_QUIZ_Q = 10
def gen_quiz(notebook_name, num_q, profile: gr.OAuthProfile | None):
    radios = [gr.update(visible=False, interactive=True) for _ in range(MAX_QUIZ_Q)]
    if not profile or not notebook_name: return "❌ Log in/Select MB", "{}", "", "" , *radios
    res_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile)).json()
    nb_id = next((nb["id"] for nb in res_nbs if nb["title"] == notebook_name), None)
    if not nb_id: return "❌ Notebook not found", "{}", "", "", *radios
    
    res = requests.post(f"{API_BASE_URL}/api/generate", headers=get_headers(profile), json={"notebook_id": nb_id, "artifact_type": "quiz", "params": {"num_questions": num_q}})
    if res.status_code != 200: return f"❌ Error: {res.text}", "{}", "", "", *radios
    
    quiz_data = res.json().get("quiz", [])
    md = ""
    for i, q in enumerate(quiz_data[:MAX_QUIZ_Q]):
        md += f"**{i+1}. {q['question']}**\n\n"
        for k, v in q['options'].items():
             md += f"- **{k})** {v}\n"
        md += "\n---\n"
        radios[i] = gr.update(visible=True, choices=["A", "B", "C", "D"], label=f"Q{i+1}")
        
    return "✅ Quiz Ready!", json.dumps(quiz_data), md, "", *radios

def submit_quiz(quiz_json, *answers):
    quiz_data = json.loads(quiz_json)
    if not quiz_data: return "❌ No quiz active."
    score = 0
    md = "### Results\n"
    for i, q in enumerate(quiz_data):
        if i >= len(answers) or not answers[i]: continue
        user_ans = answers[i]
        correct_ans = q["answer"]
        if user_ans == correct_ans:
            score += 1
            md += f"✅ **Q{i+1} Correct!** ({user_ans})\n{q.get('explanation', '')}\n\n"
        else:
            md += f"❌ **Q{i+1} Incorrect.** You chose {user_ans}, correct was {correct_ans}.\n{q.get('explanation', '')}\n\n"
    md = f"**Final Score:** {score}/{len(quiz_data)}\n\n" + md
    return md

def generate_study_guide(notebook_name, profile: gr.OAuthProfile | None):
    if not profile or not notebook_name: return "❌ Log in and select a notebook."
    res_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile)).json()
    nb_id = next((nb["id"] for nb in res_nbs if nb["title"] == notebook_name), None)
    if not nb_id: return "❌ Notebook not found."
    
    res = requests.post(f"{API_BASE_URL}/api/generate", headers=get_headers(profile), json={"notebook_id": nb_id, "artifact_type": "study_guide"})
    return res.json().get("result", f"❌ Error: {res.text}")

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
        with gr.Column(scale=2):
            nb_info_md = gr.Markdown("_Login and select a notebook_")
            with gr.Row():
                rename_in = gr.Textbox(placeholder="New name...", show_label=False, scale=3)
                rename_btn = gr.Button("✏️ Rename", size="sm", scale=1)

    # Function moved to end of blocks to reference file_in and chatbot

    def rename_notebook_ui(notebook_name, new_name, profile: gr.OAuthProfile | None):
        if not profile or not notebook_name or not new_name.strip(): 
            return gr.update(), gr.update(), "❌ Invalid operation."
        res_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile)).json()
        nb_id = next((nb["id"] for nb in res_nbs if nb["title"] == notebook_name), None)
        if not nb_id: 
            return gr.update(), gr.update(), "❌ Notebook not found."
        
        res = requests.post(f"{API_BASE_URL}/api/notebooks/rename", headers=get_headers(profile), json={"notebook_id": nb_id, "new_title": new_name.strip()})
        if res.status_code == 200:
            new_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile)).json()
            titles = [n["title"] for n in new_nbs]
            return gr.update(choices=titles, value=new_name.strip()), gr.update(value=""), f"✅ Renamed to **{new_name.strip()}**"
        return gr.update(), gr.update(), f"❌ Error: {res.text}"

    rename_btn.click(rename_notebook_ui, inputs=[active_nb, rename_in], outputs=[active_nb, rename_in, nb_info_md])

    gr.Markdown("---")

    with gr.Tabs():
        # ── TAB 1: NOTEBOOKS ─────────────────────────────────────────────────
        with gr.TabItem("📁 Notebooks"):
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("### ➕ Create New Notebook")
                    nb_name = gr.Textbox(label="Notebook Name", placeholder="e.g. Biology Notes")
                    src_type1 = gr.Radio(["PDF", "PPTX", "TXT", "URL"], label="Source Type", value="PDF")
                    file_in1 = gr.File(label="Upload Files", file_types=[".pdf",".pptx",".ppt",".txt",".md"], file_count="multiple")
                    url_in1 = gr.Textbox(label="URL", placeholder="https://...", visible=False)

                    def toggle(t):
                        return gr.File(visible=t != "URL", file_count="multiple"), gr.Textbox(visible=t == "URL")
                    src_type1.change(toggle, inputs=src_type1, outputs=[file_in1, url_in1])

                    add_btn = gr.Button("🚀 Create Notebook", variant="primary")
                    add_status = gr.Markdown("_Upload a source to begin._")
                    
                    def clear_file(): return None
                    
                    add_btn.click(
                        process_source, 
                        inputs=[nb_name, src_type1, file_in1, url_in1], 
                        outputs=[add_status, active_nb]
                    ).then(
                        clear_file, None, file_in1
                    ).then(
                        load_notebook_data, 
                        inputs=[active_nb], 
                        outputs=[nb_info_md, nb_files_view, chatbot, sum_out, pod_script_out, pod_lines_state, quiz_display_md, quiz_json_box, study_out]
                    )

                with gr.Column(variant="panel"):
                    gr.Markdown("### 📎 Append to Active Notebook")
                    gr.Markdown("_Adds sources to the notebook currently selected in the top bar._")
                    src_type2 = gr.Radio(["PDF", "PPTX", "TXT", "URL"], label="Source Type", value="PDF")
                    file_in2 = gr.File(label="Upload Files", file_types=[".pdf",".pptx",".ppt",".txt",".md"], file_count="multiple")
                    url_in2 = gr.Textbox(label="URL", placeholder="https://...", visible=False)
                    
                    src_type2.change(toggle, inputs=src_type2, outputs=[file_in2, url_in2])

                    append_btn = gr.Button("📎 Process & Append", variant="primary")
                    append_status = gr.Markdown()
                    append_btn.click(
                        process_source, 
                        inputs=[active_nb, src_type2, file_in2, url_in2], 
                        outputs=[append_status, active_nb]
                    ).then(
                        clear_file, None, file_in2
                    ).then(
                        load_notebook_data, 
                        inputs=[active_nb], 
                        outputs=[nb_info_md, nb_files_view, chatbot, sum_out, pod_script_out, pod_lines_state, quiz_display_md, quiz_json_box, study_out]
                    )
            
            with gr.Row():
                nb_files_view = gr.File(label="Files Currently in Notebook (Read-Only)", interactive=False)

        # ── TAB 2: CHAT ──────────────────────────────────────────────────────
        with gr.TabItem("💬 Chat"):
            gr.Markdown("### Ask anything about your document")
            chatbot = gr.Chatbot(label="ThinkBook AI", height=450, value=[])
            with gr.Row():
                chat_in = gr.Textbox(placeholder="Ask a question...", label="", scale=5, show_label=False)
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clr_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            chat_in.submit(chat_response, inputs=[chat_in, chatbot, active_nb], outputs=[chatbot, chat_in])
            send_btn.click(chat_response, inputs=[chat_in, chatbot, active_nb], outputs=[chatbot, chat_in])
            clr_btn.click(clear_chat, inputs=None, outputs=[chatbot, chat_in])

        # ── TAB 3: SUMMARY ───────────────────────────────────────────────────
        with gr.TabItem("📝 Summary"):
            gr.Markdown("### Generate a document summary")
            with gr.Row():
                sum_mode = gr.Radio(
                    ["Brief", "Descriptive"], value="Brief", label="Style",
                    info="Brief = 4-6 sentences · Descriptive = full structured breakdown",
                )
                sum_btn = gr.Button("✨ Generate", variant="primary")
            sum_out = gr.Markdown()
            def load_sum(): return "⏳ Generating Summary..."
            sum_btn.click(load_sum, None, sum_out).then(generate_summary, inputs=[active_nb, sum_mode], outputs=sum_out)

        # ── TAB 4: PODCAST ───────────────────────────────────────────────────
        with gr.TabItem("🎙️ Podcast"):
            gr.Markdown("""
### 2-person podcast from your document
🎤 **Alex** — Curious host (US accent) &nbsp;|&nbsp; 🎓 **Dr. Sam** — Expert guest (UK accent)
            """)
            with gr.Row():
                exchanges_sl = gr.Slider(8, 20, value=12, step=1, label="Exchanges")
                pod_btn = gr.Button("🎙️ Generate Script", variant="primary")

            pod_script_out = gr.Markdown()
            pod_lines_state = gr.State(None)

            with gr.Row():
                audio_btn = gr.Button("🔊 Generate Audio", variant="secondary")
                audio_status = gr.Markdown()
            audio_out = gr.Audio(label="🎧 Listen", type="filepath")

            def load_pod(): return "⏳ Generating Script...", None
            pod_btn.click(load_pod, None, [pod_script_out, pod_lines_state]).then(generate_podcast, inputs=[active_nb, exchanges_sl], outputs=[pod_script_out, pod_lines_state])
            
            def load_audio(): return None, "⏳ Synthesizing Audio..."
            audio_btn.click(load_audio, None, [audio_out, audio_status]).then(generate_audio, inputs=pod_lines_state, outputs=[audio_out, audio_status])

        # ── TAB 5: QUIZ ──────────────────────────────────────────────────────
        with gr.TabItem("🧪 Quiz"):
            gr.Markdown("### Test your knowledge")
            with gr.Row():
                num_q_sl = gr.Slider(3, MAX_QUIZ_Q, value=5, step=1, label="Questions")
                quiz_gen_btn = gr.Button("🎲 Generate Quiz", variant="primary")

            quiz_status_md = gr.Markdown()
            quiz_display_md = gr.Markdown()
            quiz_json_box = gr.Textbox(visible=False, value="{}")

            answer_radios = []
            for i in range(MAX_QUIZ_Q):
                r = gr.Radio(choices=["A", "B", "C", "D"], label=f"Q{i+1}", visible=False, interactive=True)
                answer_radios.append(r)

            submit_btn = gr.Button("✅ Submit Answers", variant="primary")
            quiz_results_md = gr.Markdown()

            def load_quiz(): return "⏳ Generating Quiz...", "{}", "", ""
            quiz_gen_btn.click(
                load_quiz, None, [quiz_status_md, quiz_json_box, quiz_display_md, quiz_results_md]
            ).then(
                gen_quiz,
                inputs=[active_nb, num_q_sl],
                outputs=[quiz_status_md, quiz_json_box, quiz_display_md, quiz_results_md] + answer_radios,
            )
            submit_btn.click(
                submit_quiz,
                inputs=[quiz_json_box] + answer_radios,
                outputs=quiz_results_md,
            )

        # ── TAB 6: STUDY GUIDE ───────────────────────────────────────────────
        with gr.TabItem("📚 Study Guide"):
            gr.Markdown("### Key concepts, definitions, flashcards & summary")
            study_btn = gr.Button("📚 Generate Study Guide", variant="primary")
            study_out = gr.Markdown()
            def load_study(): return "⏳ Generating Study Guide..."
            study_btn.click(load_study, None, study_out).then(generate_study_guide, inputs=[active_nb], outputs=study_out)

    def load_notebook_data(nb_name, profile: gr.OAuthProfile | None):
        # Default empties for 9 UI components
        if not nb_name or not profile:
            return "No notebook selected.", None, [], "", "", None, "", "{}", ""
            
        res_nbs = requests.get(f"{API_BASE_URL}/api/notebooks", headers=get_headers(profile)).json()
        nb_id = next((nb["id"] for nb in res_nbs if nb["title"] == nb_name), None)
        
        if not nb_id:
            return "❌ Notebook not found.", None, [], "", "", None, "", "{}", ""
            
        # Fetch uploaded files
        res_files = requests.get(f"{API_BASE_URL}/api/notebooks/{nb_id}/files", headers=get_headers(profile))
        files = res_files.json() if res_files.status_code == 200 else None
        
        # Fetch chat history
        res_chats = requests.get(f"{API_BASE_URL}/api/notebooks/{nb_id}/chats", headers=get_headers(profile))
        chats = res_chats.json() if res_chats.status_code == 200 else []
        
        # Fetch generated artifacts
        res_artifacts = requests.get(f"{API_BASE_URL}/api/notebooks/{nb_id}/artifacts", headers=get_headers(profile))
        artifacts = res_artifacts.json() if res_artifacts.status_code == 200 else {}
        
        sum_val = next((v for k, v in artifacts.items() if k.startswith("summary")), "")
        pod_script_val = next((json.loads(v).get("script", "") for k, v in artifacts.items() if k.startswith("podcast_script")), "")
        pod_lines_val = next((json.loads(v).get("parsed_lines", []) for k, v in artifacts.items() if k.startswith("podcast_script")), None)
        quiz_val = next((json.loads(v).get("quiz", []) for k, v in artifacts.items() if k.startswith("quiz")), [])
        
        quiz_json_val = json.dumps(quiz_val) if quiz_val else "{}"
        quiz_display = ""
        if quiz_val:
            for i, q in enumerate(quiz_val):
                quiz_display += f"**Q{i+1}: {q.get('question', '')}**\n"
                for j, opt in enumerate(q.get('options', [])):
                     quiz_display += f"- {chr(65+j)}: {opt}\n"
                quiz_display += "\n"
                
        study_val = next((v for k, v in artifacts.items() if k.startswith("study_guide")), "")
        
        return f"Selected: **{nb_name}**", files, chats, sum_val, pod_script_val, pod_lines_val, quiz_display, quiz_json_val, study_val

    active_nb.change(
        load_notebook_data, 
        inputs=[active_nb], 
        outputs=[nb_info_md, nb_files_view, chatbot, sum_out, pod_script_out, pod_lines_state, quiz_display_md, quiz_json_box, study_out]
    )

    # Trigger load when page opens to fetch profile and notebooks
    demo.load(fetch_notebooks, inputs=None, outputs=active_nb)

if __name__ == "__main__":
    demo.launch()
