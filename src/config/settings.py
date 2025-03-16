from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str
    MODEL_NAME: str
    TEMPERATURE: float
    
    class Config:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_file = os.path.join(root_dir, '.env')

settings = Settings()