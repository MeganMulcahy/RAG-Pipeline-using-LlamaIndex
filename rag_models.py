import os
from getpass import getpass

def initialize_models():
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    model_name = os.getenv("GOOGLE_MODEL")
    if not model_name:
        raise RuntimeError(
            "GOOGLE_MODEL is required. Set GOOGLE_MODEL in your .env or environment."
        )

    llm = GoogleGenAI(
        model=model_name,
        max_tokens=100,
        system_prompt="""
    You are a strict extraction engine.

    Rules:
    - Output exactly ONE sentence.
    - Maximum 15 words.
    - No explanations.
    - No markdown.
    - No bullet points.
    - No introductions or conclusions.
    - No extra context.
    - Answer directly using retrieved information only.
    """
    )
    Settings.llm = llm

    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not embedding_model_name:
        raise RuntimeError(
            "EMBEDDING_MODEL_NAME is required. Set EMBEDDING_MODEL_NAME to a local or cached Hugging Face model path."
        )

    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    Settings.embed_model = embed_model

    return llm, embed_model
