import gradio as gr
import rag_pie

COUNTRY_CLUB_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Lato:wght@300;400&display=swap');

body, .gradio-container {
    background-color: #FAF7F2 !important;
    font-family: 'Lato', sans-serif !important;
}

h1, h2, h3, .gr-markdown h1 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    color: #1B4332 !important;
    letter-spacing: 0.04em;
}

.gr-markdown p {
    color: #3D3D3D;
    font-size: 0.95rem;
}

/* Panel backgrounds */
.gr-box, .gr-panel, .gr-block, .gr-form {
    background-color: #FAF7F2 !important;
    border: 1px solid #D4C5A9 !important;
    border-radius: 8px !important;
}

/* Primary buttons — gold */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #C9A044, #B8892E) !important;
    color: #FAF7F2 !important;
    border: none !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.06em !important;
    border-radius: 4px !important;
}

.gr-button-primary:hover, button.primary:hover {
    background: linear-gradient(135deg, #B8892E, #A07820) !important;
}

/* Secondary buttons */
.gr-button-secondary, button.secondary {
    background: transparent !important;
    color: #1B4332 !important;
    border: 1px solid #1B4332 !important;
    font-family: 'Lato', sans-serif !important;
    letter-spacing: 0.06em !important;
    border-radius: 4px !important;
}

/* Inputs and textboxes */
input[type="text"], textarea, .gr-textbox textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #C9A044 !important;
    border-radius: 4px !important;
    color: #2C2C2C !important;
    font-family: 'Lato', sans-serif !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: #1B4332 !important;
    box-shadow: 0 0 0 2px rgba(27, 67, 50, 0.15) !important;
}

/* Chatbot */
.gr-chatbot {
    background-color: #FFFFFF !important;
    border: 1px solid #D4C5A9 !important;
    border-radius: 8px !important;
}

/* Chat bubbles */
.gr-chatbot .message.user {
    background-color: #1B4332 !important;
    color: #FAF7F2 !important;
    border-radius: 12px 12px 2px 12px !important;
}

.gr-chatbot .message.bot {
    background-color: #F0EBE1 !important;
    color: #2C2C2C !important;
    border-radius: 12px 12px 12px 2px !important;
    border-left: 3px solid #C9A044 !important;
}

/* Radio buttons */
.gr-radio label {
    color: #1B4332 !important;
    font-family: 'Lato', sans-serif !important;
}

/* Status box */
#status-box textarea {
    color: #1B4332 !important;
    font-style: italic;
}

/* Divider line under header */
.header-rule {
    border: none;
    border-top: 1px solid #C9A044;
    margin: 0.25rem 0 1.25rem 0;
}
"""

HEADER_HTML = """
<div style="padding: 1.5rem 0 0.5rem 0;">
  <h1 style="font-family: 'Cormorant Garamond', Georgia, serif; color: #1B4332;
             font-size: 2.2rem; font-weight: 600; margin: 0; letter-spacing: 0.04em;">
    Mortgage Document Assistant
  </h1>
  <div style="border-top: 1px solid #C9A044; margin: 0.4rem 0 0.5rem 0;"></div>
  <p style="font-family: 'Lato', sans-serif; color: #5A5A5A; font-size: 0.9rem;
            margin: 0; letter-spacing: 0.02em;">
    Upload a mortgage PDF and ask questions about its contents.
  </p>
</div>
"""


def process_pdf(pdf_file, state):
    if pdf_file is None:
        return "Please upload a Mortgage Blob PDF file.", state, []

    pages = rag_pie.load_pdf(pdf_file.name)
    index = rag_pie.build_index(pages)
    engine = rag_pie.build_rag_pipeline(index)

    state = {"index": index, "engine": engine}

    doc_types = sorted({m["doc_type"] for m in rag_pie.pdf_metadata_store})
    status = "Ready! Found: " + ", ".join(doc_types)

    return status, state, []


def answer_query(query, history, state):
    if not query.strip():
        return history, state, ""

    if not state or "index" not in state:
        return history + [("Please upload a PDF first.", None)], state, ""

    predicted_type = rag_pie.predict_query_doc_type(query)
    matched = rag_pie.retrieve_files_by_doc_type(predicted_type)
    query_engine = rag_pie.build_filtered_engine(state["index"], predicted_type) if matched else state["engine"]

    response = query_engine.query(query)

    sources = [
        f"p.{m.get('page_start')}-{m.get('page_end')} | {m.get('doc_type')}"
        for node in response.source_nodes
        for m in [node.metadata]
    ]

    answer = str(response)
    if sources:
        answer += "\n\nSources: " + " · ".join(sources)

    return history + [(query, answer)], state, ""


with gr.Blocks(title="Mortgage Document Assistant", css=COUNTRY_CLUB_CSS) as demo:
    gr.HTML(HEADER_HTML)

    state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Document Setup")

            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])

            process_btn = gr.Button("Process Document", variant="primary")

            status_box = gr.Textbox(
                label="Status",
                elem_id="status-box",
                interactive=False,
                placeholder="Upload a PDF to get started...",
            )

        with gr.Column(scale=2):
            gr.Markdown("### Conversation")

            chatbot = gr.Chatbot(height=420, label="")

            with gr.Row():
                query_box = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    label="",
                    scale=4,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("Clear Chat", variant="secondary")

    process_btn.click(
        process_pdf,
        inputs=[pdf_upload, state],
        outputs=[status_box, state, chatbot],
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
    clear_btn.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Base())
