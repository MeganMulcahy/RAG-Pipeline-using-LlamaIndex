import subprocess
import sys

REQUIRED_PACKAGES = [
    "llama-index",
    "google-genai",
    "llama-index-embeddings-huggingface",
    "llama-index-retrievers-bm25",
    "llama-index-llms-llama-cpp",
    "llama-index-llms-ollama",
    "pymupdf",
    "pymupdf4llm",
    "unstructured[pdf]",
    "pytesseract",
    "pillow",
    "llama-index-readers-file",
    "gradio",
]


def install_packages(packages):
    """Install missing Python packages using the current interpreter."""
    command = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    print("Installing packages:", " ".join(packages))
    subprocess.check_call(command)


def main():
    try:
        install_packages(REQUIRED_PACKAGES)
        print("All required packages are installed.")
    except subprocess.CalledProcessError as error:
        print("Failed to install required packages.")
        print("Error:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()