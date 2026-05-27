import re
import time
from collections import defaultdict

import gradio as gr
import rag_pie

# ── Rate limiting ─────────────────────────────────────────────────────────────
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


# ── Styles ────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Montserrat:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #F4EFE6 !important;
    font-family: 'Montserrat', sans-serif !important;
    color: #2A2A2A !important;
}

/* ── Header ── */
#app-header {
    background: #1C3D2E;
    border-bottom: 2px solid #B8922E;
    padding: 1rem 2.4rem;
    display: flex;
    align-items: center;
    gap: 1.6rem;
}
#app-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.55rem;
    font-weight: 600;
    color: #F4EFE6;
    letter-spacing: 0.05em;
    line-height: 1.1;
}
#app-subtitle {
    font-size: 0.62rem;
    font-weight: 400;
    color: #7DA882;
    letter-spacing: 0.24em;
    text-transform: uppercase;
}
#header-rule {
    width: 1px;
    height: 28px;
    background: #B8922E;
    opacity: 0.4;
    flex-shrink: 0;
}

/* ── Outer wrapper ── */
#outer-wrap {
    padding: 1.1rem 1.6rem 1.6rem !important;
    gap: 1.2rem !important;
    align-items: flex-start !important;
}

/* ── Left panel ── */
#left-col {
    background: #FFFFFF;
    border: 1px solid #DDD6CA;
    padding: 1.1rem 1.2rem 1.4rem;
    min-height: 580px;
    display: flex;
    flex-direction: column;
    gap: 0;
}
.panel-label {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #B8922E;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #EAE4DA;
    margin-bottom: 0.9rem;
}
#doc-status {
    font-size: 0.73rem;
    line-height: 1.75;
    color: #6B6B6B;
    flex: 1;
}
.upload-wrap button {
    width: 100% !important;
    margin-top: 1.1rem !important;
    background: transparent !important;
    color: #B8922E !important;
    border: 1px solid #B8922E !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.52rem 0 !important;
    transition: background 0.15s, color 0.15s;
}
.upload-wrap button:hover {
    background: #B8922E !important;
    color: #FFF !important;
}

/* ── Chat panel ── */
.gr-chatbot, [data-testid="chatbot"] {
    background: #FFFFFF !important;
    border: 1px solid #DDD6CA !important;
    border-radius: 2px !important;
}
[data-testid="chatbot"] .user > div,
.gr-chatbot .message.user > div {
    background: #1C3D2E !important;
    color: #F4EFE6 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    border-radius: 10px 10px 2px 10px !important;
    padding: 0.6rem 0.85rem !important;
}
[data-testid="chatbot"] .bot > div,
.gr-chatbot .message.bot > div {
    background: #F9F6F1 !important;
    color: #2A2A2A !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
    border-left: 3px solid #B8922E !important;
    border-radius: 10px 10px 10px 2px !important;
    padding: 0.6rem 0.85rem !important;
}

/* ── Input area ── */
#input-area {
    margin-top: 0.65rem !important;
    gap: 0.55rem !important;
    align-items: flex-end !important;
}
#input-area textarea {
    background: #FFFFFF !important;
    border: 1px solid #C8BFB0 !important;
    border-radius: 2px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.82rem !important;
    color: #2A2A2A !important;
    resize: none !important;
    min-height: 44px !important;
}
#input-area textarea:focus {
    border-color: #1C3D2E !important;
    box-shadow: 0 0 0 2px rgba(28,61,46,0.09) !important;
    outline: none !important;
}
button.primary, .gr-button-primary {
    background: #1C3D2E !important;
    color: #F4EFE6 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.58rem 1rem !important;
    white-space: nowrap !important;
}
button.primary:hover { background: #143023 !important; }
button.secondary, .gr-button-secondary {
    background: transparent !important;
    color: #8A8074 !important;
    border: 1px solid #C8BFB0 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
}
button.secondary:hover { color: #1C3D2E !important; border-color: #1C3D2E !important; }

/* hide Gradio chrome */
.gr-box, .gr-panel, .gr-block { border: none !important; background: transparent !important; }
footer { display: none !important; }
"""

HEADER_HTML = """
<div id="app-header">
  <div>
    <div id="app-title">The Members&#39;&nbsp;Room</div>
    <div id="app-subtitle">Document Intelligence &nbsp;·&nbsp; Private</div>
  </div>
  <div id="header-rule"></div>
</div>
"""

_STATUS_EMPTY = """
<div id="doc-status" style="color:#9A9180;">
  No document loaded.<br><br>
  Attach a PDF below — it will be parsed automatically.
</div>
"""

_STATUS_LOADING = """
<div id="doc-status" style="color:#B8922E;">
  Parsing document&hellip;<br><br>
  <span style="color:#9A9180;font-size:0.68rem;">
    Classifying pages and building index.
  </span>
</div>
"""


def _status_html(doc_types: list, file_names: list) -> str:
    files = "".join(
        f'<div style="color:#1C3D2E;font-weight:500;margin-bottom:2px;">{n}</div>'
        for n in file_names
    )
    types = " &nbsp;·&nbsp; ".join(
        f'<span style="color:#1C3D2E;font-weight:500;">{t}</span>'
        for t in sorted(doc_types)
    )
    return (
        f'<div id="doc-status">'
        f'<div style="margin-bottom:0.7rem;">{files}</div>'
        f'<div style="font-size:0.68rem;color:#8A8074;letter-spacing:0.05em;">'
        f'DETECTED TYPES</div>'
        f'<div style="margin-top:0.25rem;font-size:0.72rem;line-height:1.9;">{types}</div>'
        f'</div>'
    )


# ── Handlers ──────────────────────────────────────────────────────────────────

def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def process_pdfs(pdf_files, history, state):
    """Called the moment files are attached — parses and indexes immediately."""
    if not pdf_files:
        return history, state, _STATUS_EMPTY

    paths      = [f.name for f in pdf_files] if isinstance(pdf_files, list) else [pdf_files.name]
    file_names = [p.replace("\\", "/").split("/")[-1] for p in paths]

    try:
        docs   = rag_pie.load_pdfs(paths)
        index  = rag_pie.build_index(docs)
        engine = rag_pie.build_rag_pipeline(index)
    except Exception as exc:
        return (
            history + [_msg("assistant", f"Error loading document: {exc}")],
            state,
            _STATUS_EMPTY,
        )

    state     = {"index": index, "engine": engine}
    doc_types = sorted({m["doc_type"] for m in rag_pie.pdf_metadata_store})
    n         = len(paths)

    msg = (
        f"**{n} file{'s' if n > 1 else ''} loaded** — {', '.join(file_names)}\n\n"
        f"Document types found: {', '.join(doc_types)}\n\n"
        "Ready. Ask me anything about your documents."
    )
    return history + [_msg("assistant", msg)], state, _status_html(doc_types, file_names)


def answer_query(query, history, state, request: gr.Request):
    ip    = request.client.host if request else "unknown"
    query = _sanitize(query)

    if not query:
        yield history, state, ""
        return

    if not _check_rate_limit(ip):
        yield history + [
            _msg("user", query),
            _msg("assistant", "Too many requests — please wait before trying again."),
        ], state, ""
        return

    if not state or "index" not in state:
        yield history + [
            _msg("user", query),
            _msg("assistant", "Please attach a PDF first using the button on the left."),
        ], state, ""
        return

    # Show thinking placeholder immediately so the user sees activity
    history = history + [_msg("user", query), _msg("assistant", "…")]
    yield history, state, ""

    predicted_type = rag_pie.predict_query_doc_type(query)
    matched        = rag_pie.retrieve_files_by_doc_type(predicted_type) if predicted_type else []
    engine         = (
        rag_pie.build_filtered_engine(state["index"], predicted_type)
        if matched else state["engine"]
    )

    response = engine.query(query)
    answer   = str(response).strip()

    sources = [
        f"p.{m.get('page_start')}–{m.get('page_end')} · {m.get('doc_type')} · {m.get('file_name')}"
        for node in response.source_nodes
        for m in [node.metadata]
    ]
    if sources:
        answer += "\n\n*Sources: " + " | ".join(sources) + "*"

    history[-1] = _msg("assistant", answer)
    yield history, state, ""


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML(HEADER_HTML)

    _state = gr.State({})

    with gr.Row(elem_id="outer-wrap", equal_height=False):

        # ── Left: document panel ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=200, elem_id="left-col"):
            gr.HTML('<div class="panel-label">Documents</div>')
            doc_status = gr.HTML(_STATUS_EMPTY)
            with gr.Group(elem_classes=["upload-wrap"]):
                upload_btn = gr.UploadButton(
                    "Attach PDF",
                    file_types=[".pdf"],
                    file_count="multiple",
                    variant="secondary",
                )

        # ── Right: chat ───────────────────────────────────────────────────
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                value=[],
                height=520,
                label="",
                show_label=False,
            )
            with gr.Row(elem_id="input-area"):
                query_box = gr.Textbox(
                    placeholder="Ask about your documents…",
                    label="",
                    scale=5,
                    container=False,
                    lines=1,
                    max_lines=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
            with gr.Row():
                clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm", scale=0)

    # ── Event wiring ─────────────────────────────────────────────────────────
    upload_btn.upload(
        process_pdfs,
        inputs=[upload_btn, chatbot, _state],
        outputs=[chatbot, _state, doc_status],
    )
    send_btn.click(
        answer_query,
        inputs=[query_box, chatbot, _state],
        outputs=[chatbot, _state, query_box],
    )
    query_box.submit(
        answer_query,
        inputs=[query_box, chatbot, _state],
        outputs=[chatbot, _state, query_box],
    )
    clear_btn.click(
        lambda: ([], {}, _STATUS_EMPTY),
        inputs=None,
        outputs=[chatbot, _state, doc_status],
    )


if __name__ == "__main__":
    demo.launch(share=True)
