from langchain_ollama import ChatOllama
from config.settings import settings  # Changed from relative to absolute import

def initialize_llm():
    return ChatOllama(
        model=settings.MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.TEMPERATURE,
        streaming=True
    )