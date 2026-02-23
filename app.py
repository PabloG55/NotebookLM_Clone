"""
ThinkBook - NotebookLM Clone
Gradio interface — works on HuggingFace Spaces (Gradio SDK).
All core/features logic unchanged; only UI layer replaced.
"""
import gradio as gr
import os
import json
import tempfile
from dotenv import load_dotenv

load_dotenv()

from core.ingestion import ingest_source
from core.chunker import chunk_text
from core.vector_store import VectorStore
from features.summarizer import summarize
from features.chat import build_rag_messages
from features.podcast import generate_podcast_script, parse_podcast_script, generate_podcast_audio
from features.quiz import generate_quiz, check_answer
from features.study_guide import generate_study_guide
from core.groq_client import groq_stream

# Global state — persists within one session
NOTEBOOKS: dict = {}

MAX_QUIZ_Q = 10


# ══════════════════════════════════════════════════════════════
# NOTEBOOK MANAGEMENT
# ══════════════════════════════════════════════════════════════

def process_source(notebook_name, source_type, file_obj, url_text):
    global NOTEBOOKS
    name = notebook_name.strip()
    if not name:
        return "❌ Please enter a notebook name.", gr.Dropdown(choices=list(NOTEBOOKS.keys()))
    if name in NOTEBOOKS:
        return f"❌ '{name}' already exists. Use a different name.", gr.Dropdown(choices=list(NOTEBOOKS.keys()))
    try:
        if source_type in ["PDF", "PPTX", "TXT"]:
            if file_obj is None:
                return "❌ Please upload a file.", gr.Dropdown(choices=list(NOTEBOOKS.keys()))
            with open(file_obj.name, "rb") as f:
                raw_bytes = f.read()
            raw_text = ingest_source(source_type.lower(), raw_bytes)
        else:
            if not url_text.strip():
                return "❌ Please enter a URL.", gr.Dropdown(choices=list(NOTEBOOKS.keys()))
            raw_text = ingest_source("url", url_text.strip())

        if not raw_text or len(raw_text.strip()) < 50:
            return "❌ Could not extract enough text.", gr.Dropdown(choices=list(NOTEBOOKS.keys()))

        chunks = chunk_text(raw_text)
        store = VectorStore()
        store.add_chunks(chunks)
        NOTEBOOKS[name] = {"text": raw_text, "store": store}
        choices = list(NOTEBOOKS.keys())
        return f"✅ **{name}** added! {len(chunks)} chunks · {len(raw_text.split()):,} words.", gr.Dropdown(choices=choices, value=name)
    except Exception as e:
        return f"❌ Error: {e}", gr.Dropdown(choices=list(NOTEBOOKS.keys()))


def delete_notebook(notebook_name):
    global NOTEBOOKS
    if notebook_name and notebook_name in NOTEBOOKS:
        del NOTEBOOKS[notebook_name]
    choices = list(NOTEBOOKS.keys())
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None), "🗑️ Deleted."


def get_notebook_info(notebook_name):
    if not notebook_name or notebook_name not in NOTEBOOKS:
        return "No notebook selected."
    text = NOTEBOOKS[notebook_name]["text"]
    return f"📊 **{notebook_name}** · {len(text.split()):,} words"


# ══════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════

def chat_response(message, history, notebook_name):
    if not message.strip():
        return history, ""
    if not notebook_name or notebook_name not in NOTEBOOKS:
        history = history or []
        history.append((message, "❌ Please select a notebook first from the Notebooks tab."))
        return history, ""

    store = NOTEBOOKS[notebook_name]["store"]
    prior = []
    for user_msg, bot_msg in (history or []):
        prior.append({"role": "user", "content": user_msg})
        prior.append({"role": "assistant", "content": bot_msg})

    from features.chat import build_rag_messages
    messages = build_rag_messages(message, store, prior)

    full_response = ""
    for token in groq_stream(messages, temperature=0.6, max_tokens=2048):
        full_response += token

    history = history or []
    history.append((message, full_response))
    return history, ""


def clear_chat():
    return [], ""


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

def generate_summary(notebook_name, mode):
    if not notebook_name or notebook_name not in NOTEBOOKS:
        return "❌ Please select a notebook first."
    try:
        return summarize(NOTEBOOKS[notebook_name]["text"], mode=mode.lower())
    except Exception as e:
        return f"❌ Error: {e}"


# ══════════════════════════════════════════════════════════════
# PODCAST
# ══════════════════════════════════════════════════════════════

def generate_podcast(notebook_name, num_exchanges):
    if not notebook_name or notebook_name not in NOTEBOOKS:
        return "❌ Please select a notebook first.", None
    try:
        script = generate_podcast_script(NOTEBOOKS[notebook_name]["text"], int(num_exchanges))
        lines = parse_podcast_script(script)
        if not lines:
            return "❌ Could not parse script. Try again.", None
        formatted = ""
        for speaker, line in lines:
            icon = "🎤" if speaker == "Alex" else "🎓"
            formatted += f"{icon} **{speaker}:** {line}\n\n"
        return formatted, lines
    except Exception as e:
        return f"❌ Error: {e}", None


def generate_audio(lines_state):
    if not lines_state:
        return None, "❌ Generate the podcast script first."
    try:
        audio_bytes = generate_podcast_audio(lines_state)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name, "✅ Audio ready!"
    except Exception as e:
        return None, f"❌ Audio error: {e}"


# ══════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════

def render_quiz_md(quiz):
    if not quiz:
        return ""
    out = ""
    for i, q in enumerate(quiz):
        out += f"**Q{i+1}. {q['question']}**\n"
        for letter, option in q.get("options", {}).items():
            out += f"- **{letter}:** {option}\n"
        out += "\n"
    return out


def gen_quiz(notebook_name, num_q):
    if not notebook_name or notebook_name not in NOTEBOOKS:
        return "❌ Select a notebook first.", "{}", "", "" , *[gr.Radio(value=None, visible=False)]*MAX_QUIZ_Q

    try:
        quiz = generate_quiz(NOTEBOOKS[notebook_name]["text"], num_questions=int(num_q))
        quiz_json = json.dumps(quiz)
        n = int(num_q)
        radio_updates = [
            gr.Radio(
                choices=[f"A: {q['options'].get('A','')}", f"B: {q['options'].get('B','')}", 
                         f"C: {q['options'].get('C','')}", f"D: {q['options'].get('D','')}"],
                label=f"Q{i+1}",
                value=None,
                visible=(i < n),
                interactive=True
            ) if i < n else gr.Radio(visible=False, value=None)
            for i, q in enumerate(quiz[:MAX_QUIZ_Q])
        ]
        # pad if quiz shorter than MAX
        while len(radio_updates) < MAX_QUIZ_Q:
            radio_updates.append(gr.Radio(visible=False, value=None))

        return "✅ Quiz ready! Select your answers below.", quiz_json, render_quiz_md(quiz), "", *radio_updates
    except Exception as e:
        return f"❌ Error: {e}", "{}", "", "", *[gr.Radio(visible=False, value=None)]*MAX_QUIZ_Q


def submit_quiz(quiz_json, *answers):
    try:
        quiz = json.loads(quiz_json)
    except Exception:
        return "❌ No quiz loaded."
    if not quiz:
        return "❌ No quiz loaded."

    results = ""
    correct_count = 0
    for i, q in enumerate(quiz):
        user_ans = answers[i] if i < len(answers) else ""
        if not user_ans:
            results += f"**Q{i+1}:** ⚠️ Not answered\n\n"
            continue
        letter = user_ans[0]
        is_correct, explanation = check_answer(q, letter)
        if is_correct:
            correct_count += 1
            results += f"**Q{i+1}:** ✅ Correct! ({q['answer']})\n💡 _{explanation}_\n\n"
        else:
            results += f"**Q{i+1}:** ❌ You chose **{letter}**, correct: **{q['answer']}**\n💡 _{explanation}_\n\n"

    pct = int((correct_count / len(quiz)) * 100)
    grade = "🏆 Excellent!" if pct >= 80 else ("📚 Good effort!" if pct >= 60 else "📖 Keep studying!")
    results += f"\n---\n### Score: {correct_count}/{len(quiz)} ({pct}%) {grade}"
    return results


# ══════════════════════════════════════════════════════════════
# STUDY GUIDE
# ══════════════════════════════════════════════════════════════

def get_study_guide(notebook_name):
    if not notebook_name or notebook_name not in NOTEBOOKS:
        return "❌ Please select a notebook first."
    try:
        return generate_study_guide(NOTEBOOKS[notebook_name]["text"])
    except Exception as e:
        return f"❌ Error: {e}"


# ══════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════

css = """
#title { text-align: center; padding: 20px 0 10px 0; }
#title h1 {
    background: linear-gradient(90deg, #388bfd, #56d364);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
}
#title p { color: #8b949e; margin: 4px 0 0 0; }
footer { display: none !important; }
"""

with gr.Blocks(
    css=css,
    title="ThinkBook 🧠",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green", neutral_hue="slate"),
) as demo:

    # Header
    gr.Markdown(
        "# 🧠 ThinkBook\nUpload any document · Chat · Summarize · Podcast · Quiz · Study Guide",
        elem_id="title",
    )

    # Global notebook selector bar
    with gr.Row():
        active_nb = gr.Dropdown(choices=[], label="📚 Active Notebook", interactive=True, scale=4)
        nb_info_md = gr.Markdown("_No notebook loaded yet_")

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
                    file_in = gr.File(label="Upload File", file_types=[".pdf",".pptx",".ppt",".txt",".md"])
                    url_in = gr.Textbox(label="URL", placeholder="https://...", visible=False)

                    def toggle(t):
                        return gr.File(visible=t != "URL"), gr.Textbox(visible=t == "URL")
                    src_type.change(toggle, inputs=src_type, outputs=[file_in, url_in])

                    add_btn = gr.Button("🚀 Process & Add", variant="primary")

                with gr.Column():
                    add_status = gr.Markdown("_Upload a source to begin._")
                    gr.Markdown("---")
                    gr.Markdown("### 🗑️ Delete Active Notebook")
                    del_btn = gr.Button("Delete Selected Notebook", variant="stop")
                    del_status = gr.Markdown("")

            add_btn.click(process_source, inputs=[nb_name, src_type, file_in, url_in], outputs=[add_status, active_nb])
            del_btn.click(delete_notebook, inputs=active_nb, outputs=[active_nb, del_status])

        # ── TAB 2: CHAT ──────────────────────────────────────────────────────
        with gr.TabItem("💬 Chat"):
            gr.Markdown("### Ask anything about your document")
            chatbot = gr.Chatbot(label="ThinkBook AI", height=450, bubble_full_width=False)
            with gr.Row():
                chat_in = gr.Textbox(placeholder="Ask a question...", label="", scale=5, show_label=False)
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clr_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            send_btn.click(chat_response, inputs=[chat_in, chatbot, active_nb], outputs=[chatbot, chat_in])
            chat_in.submit(chat_response, inputs=[chat_in, chatbot, active_nb], outputs=[chatbot, chat_in])
            clr_btn.click(clear_chat, outputs=[chatbot, chat_in])

        # ── TAB 3: SUMMARY ───────────────────────────────────────────────────
        with gr.TabItem("📝 Summary"):
            gr.Markdown("### Generate a document summary")
            with gr.Row():
                sum_mode = gr.Radio(["Brief", "Descriptive"], value="Brief", label="Style",
                                    info="Brief = 4-6 sentences · Descriptive = full structured breakdown")
                sum_btn = gr.Button("✨ Generate", variant="primary")
            sum_out = gr.Markdown()
            sum_btn.click(generate_summary, inputs=[active_nb, sum_mode], outputs=sum_out)

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

            pod_btn.click(generate_podcast, inputs=[active_nb, exchanges_sl], outputs=[pod_script_out, pod_lines_state])
            audio_btn.click(generate_audio, inputs=pod_lines_state, outputs=[audio_out, audio_status])

        # ── TAB 5: QUIZ ──────────────────────────────────────────────────────
        with gr.TabItem("🧪 Quiz"):
            gr.Markdown("### Test your knowledge")
            with gr.Row():
                num_q_sl = gr.Slider(3, MAX_QUIZ_Q, value=5, step=1, label="Questions")
                quiz_gen_btn = gr.Button("🎲 Generate Quiz", variant="primary")

            quiz_status_md = gr.Markdown()
            quiz_display_md = gr.Markdown()
            quiz_json_box = gr.Textbox(visible=False, value="{}")

            # Radio buttons for answers — one per possible question
            answer_radios = []
            for i in range(MAX_QUIZ_Q):
                r = gr.Radio(choices=["A","B","C","D"], label=f"Q{i+1}", visible=False, interactive=True)
                answer_radios.append(r)

            submit_btn = gr.Button("✅ Submit Answers", variant="primary")
            quiz_results_md = gr.Markdown()

            quiz_gen_btn.click(
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
            study_btn.click(get_study_guide, inputs=active_nb, outputs=study_out)

    gr.Markdown("<center><small>Powered by Groq · FAISS · Gradio</small></center>")

if __name__ == "__main__":
    demo.launch()
