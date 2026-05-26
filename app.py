import re
import time
from collections import defaultdict

import gradio as gr
import rag_pie

# ── Rate limiting ──────────────────────────────────────────────────────────
RATE_LIMIT_MAX  = 20
RATE_LIMIT_SECS = 60
MAX_QUERY_LEN   = 500

_ip_requests: dict = defaultdict(list)


def _check_rate_limit(ip: str) -> bool:
    now    = time.time()
    cutoff = now - RATE_LIMIT_SECS
    _ip_requests[ip] = [t for t in _ip_requests[ip] if t > cutoff]
    if len(_ip_requests[ip]) >= RATE_LIMIT_MAX:
        return False
    _ip_requests[ip].append(now)
    return True


def _sanitize(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:MAX_QUERY_LEN].strip()


# ── Styles ────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Montserrat:wght@300;400;500&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #F4EFE6 !important;
    font-family: 'Montserrat', sans-serif !important;
    color: #2A2A2A !important;
}

/* ── Header ── */
#header {
    background: #1C3D2E;
    border-bottom: 2px solid #B8922E;
    padding: 1.1rem 2.5rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    margin-bottom: 0 !important;
}
#header-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #F4EFE6;
    letter-spacing: 0.06em;
    line-height: 1;
}
#header-sub {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.65rem;
    font-weight: 400;
    color: #7DA882;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}
#header-divider {
    width: 1px;
    height: 30px;
    background: #B8922E;
    opacity: 0.45;
    flex-shrink: 0;
}

/* ── Main layout ── */
#main-row {
    gap: 1.2rem !important;
    padding: 1.2rem 1.6rem !important;
    align-items: flex-start !important;
}

/* ── Left panel ── */
#left-panel {
    background: #FFFFFF;
    border: 1px solid #D9D2C5;
    border-radius: 2px;
    padding: 1.2rem 1.3rem;
    min-height: 560px;
}
#panel-heading {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #B8922E;
    margin-bottom: 0.9rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid #E8E2D8;
}
#doc-status {
    font-size: 0.72rem;
    color: #6B6B6B;
    line-height: 1.7;
}
#doc-status strong {
    color: #1C3D2E;
    font-weight: 500;
}
.upload-btn button {
    width: 100% !important;
    background: transparent !important;
    color: #B8922E !important;
    border: 1px solid #B8922E !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.55rem !important;
    margin-top: 1rem !important;
    transition: background 0.15s, color 0.15s;
}
.upload-btn button:hover {
    background: #B8922E !important;
    color: #FFFFFF !important;
}

/* ── Chat panel ── */
#chat-panel {
    flex: 1;
}
.gr-chatbot, [data-testid="chatbot"] {
    background: #FFFFFF !important;
    border: 1px solid #D9D2C5 !important;
    border-radius: 2px !important;
    font-family: 'Montserrat', sans-serif !important;
}
[data-testid="chatbot"] .user > div,
.gr-chatbot .message.user > div {
    background: #1C3D2E !important;
    color: #F4EFE6 !important;
    border-radius: 10px 10px 2px 10px !important;
    font-size: 0.83rem !important;
    line-height: 1.55 !important;
    padding: 0.65rem 0.9rem !important;
}
[data-testid="chatbot"] .bot > div,
.gr-chatbot .message.bot > div {
    background: #F9F6F1 !important;
    color: #2A2A2A !important;
    border-left: 3px solid #B8922E !important;
    border-radius: 10px 10px 10px 2px !important;
    font-size: 0.83rem !important;
    line-height: 1.65 !important;
    padding: 0.65rem 0.9rem !important;
}

/* ── Input row ── */
#input-row {
    margin-top: 0.7rem !important;
    gap: 0.6rem !important;
    align-items: center !important;
}
#input-row textarea, #input-row input[type="text"] {
    background: #FFFFFF !important;
    border: 1px solid #C8BFB0 !important;
    border-radius: 2px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.83rem !important;
    color: #2A2A2A !important;
    padding: 0.55rem 0.8rem !important;
}
#input-row textarea:focus, #input-row input[type="text"]:focus {
    border-color: #1C3D2E !important;
    box-shadow: 0 0 0 2px rgba(28,61,46,0.1) !important;
    outline: none !important;
}
button.primary, .gr-button-primary {
    background: #1C3D2E !important;
    color: #F4EFE6 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.1rem !important;
}
button.primary:hover { background: #143023 !important; }
button.secondary, .gr-button-secondary {
    background: transparent !important;
    color: #6B6B6B !important;
    border: 1px solid #C8BFB0 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
}
button.secondary:hover { border-color: #1C3D2E !important; color: #1C3D2E !important; }

/* ── Misc ── */
.gr-box, .gr-panel, .gr-block { border: none !important; background: transparent !important; }
footer { display: none !important; }
"""

HEADER_HTML = """
<div id="header">
  <div>
    <div id="header-title">The Members&#39; Room</div>
  </div>
  <div id="header-divider"></div>
  <div id="header-sub">Document Intelligence &nbsp;·&nbsp; Private Access</div>
</div>
"""

EMPTY_STATUS_HTML = """
<div id="doc-status">
  <div style="color:#B8922E; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
              margin-bottom:0.5rem; font-weight:500;">No document loaded</div>
  Attach a PDF using the button below to begin.
</div>
"""


def _build_status_html(doc_types: list, file_names: list) -> str:
    types_str = " &nbsp;·&nbsp; ".join(f"<strong>{t}</strong>" for t in sorted(doc_types))
    files_str = "<br>".join(f"&nbsp;&nbsp;{n}" for n in file_names)
    return (
        '<div id="doc-status">'
        '<div style="color:#1C3D2E; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;'
        '            margin-bottom:0.6rem; font-weight:500;">Document loaded</div>'
        f'<div style="margin-bottom:0.5rem;">{files_str}</div>'
        f'<div style="margin-top:0.4rem; font-size:0.7rem; color:#6B6B6B;">'
        f'Types detected:<br><span style="color:#1C3D2E;">{types_str}</span></div>'
        '</div>'
    )


# ── Handlers ─────────────────────────────────────────────────────────────

def process_pdfs(pdf_files, history, state):
    if not pdf_files:
        return history, state, EMPTY_STATUS_HTML

    paths = [f.name for f in pdf_files] if isinstance(pdf_files, list) else [pdf_files.name]
    file_names = [p.split("\\")[-1].split("/")[-1] for p in paths]

    try:
        pages = rag_pie.load_pdfs(paths)
        index = rag_pie.build_index(pages)
        engine = rag_pie.build_rag_pipeline(index)
    except Exception as e:
        err_msg = f"Error loading documents: {e}"
        return history + [{"role": "assistant", "content": err_msg}], state, EMPTY_STATUS_HTML

    state = {"index": index, "engine": engine}

    doc_types = sorted({m["doc_type"] for m in rag_pie.pdf_metadata_store})
    n = len(paths)
    welcome = (
        f"**{n} file{'s' if n > 1 else ''} loaded** — {', '.join(file_names)}\n\n"
        f"Documents identified: {', '.join(doc_types)}\n\n"
        "Ready — ask me anything about your documents."
    )

    status_html = _build_status_html(doc_types, file_names)
    return history + [{"role": "assistant", "content": welcome}], state, status_html


def answer_query(query, history, state, request: gr.Request):
    ip = request.client.host if request else "unknown"
    query = _sanitize(query)

    if not query:
        yield history, state, ""
        return

    if not _check_rate_limit(ip):
        yield history + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": "Too many requests — please wait a moment and try again."},
        ], state, ""
        return

    if not state or "index" not in state:
        yield history + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": "Please attach a PDF using the button on the left."},
        ], state, ""
        return

    predicted_type = rag_pie.predict_query_doc_type(query)
    matched        = rag_pie.retrieve_files_by_doc_type(predicted_type) if predicted_type else []
    query_engine   = (
        rag_pie.build_filtered_engine(state["index"], predicted_type)
        if matched else state["engine"]
    )

    streaming_response = query_engine.query(query)

    partial = ""
    history = history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": ""},
    ]
    for token in streaming_response.response_gen:
        partial += token
        history[-1]["content"] = partial
        yield history, state, ""

    sources = [
        f"p.{m.get('page_start')}–{m.get('page_end')} · {m.get('doc_type')} · {m.get('file_name')}"
        for node in streaming_response.source_nodes
        for m in [node.metadata]
    ]
    if sources:
        history[-1]["content"] = partial + "\n\n*Sources: " + " | ".join(sources) + "*"

    yield history, state, ""


# ── Layout ───────────────────────────────────────────────────────────────

with gr.Blocks(title="The Members' Room", css=CSS) as demo:
    gr.HTML(HEADER_HTML)

    state = gr.State({})

    with gr.Row(elem_id="main-row", equal_height=False):

        # Left panel — document info + upload
        with gr.Column(scale=1, min_width=210, elem_id="left-panel"):
            gr.HTML('<div id="panel-heading">Documents</div>')
            doc_status = gr.HTML(EMPTY_STATUS_HTML)
            upload_btn = gr.UploadButton(
                "Attach PDF",
                file_types=[".pdf"],
                file_count="multiple",
                variant="secondary",
                elem_classes=["upload-btn"],
            )

        # Right panel — chat
        with gr.Column(scale=4, elem_id="chat-panel"):
            chatbot = gr.Chatbot(
                height=520,
                label="",
                show_label=False,
                bubble_full_width=False,
            )
            with gr.Row(elem_id="input-row"):
                query_box = gr.Textbox(
                    placeholder="Ask about your documents...",
                    label="",
                    scale=5,
                    container=False,
                    lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, size="sm")

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", size="sm", scale=0)

    # Wiring
    upload_btn.upload(
        process_pdfs,
        inputs=[upload_btn, chatbot, state],
        outputs=[chatbot, state, doc_status],
    )
    send_btn.click(
        answer_query,
        inputs=[query_box, chatbot, state],
        outputs=[chatbot, state, query_box],
    )
    query_box.submit(
        answer_query,
        inputs=[query_box, chatbot, state],
        outputs=[chatbot, state, query_box],
    )
    clear_btn.click(lambda: ([], {}), None, [chatbot, state])


if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Base())
