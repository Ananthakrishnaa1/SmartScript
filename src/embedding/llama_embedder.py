from langchain_ollama import OllamaEmbeddings

class LLamaEmbedder:
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.embedder = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )