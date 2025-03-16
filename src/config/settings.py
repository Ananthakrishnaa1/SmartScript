from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"  # Default value
    MODEL_NAME: str = "llama3.1:8b"                  # Default value
    TEMPERATURE: float = 0.3                         # Default value
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "test"               # Default value
    
    class Config:
        # Look for .env file in project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_file = os.path.join(root_dir, '.env')

settings = Settings()