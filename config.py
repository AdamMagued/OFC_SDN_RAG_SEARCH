import os
from typing import Optional

class Config:
    # Ollama settings
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:latest")
    
    # Data paths
    SDN_CSV_PATH: str = os.getenv("SDN_CSV_PATH", "./data/sdn.csv")
    ALT_CSV_PATH: str = os.getenv("ALT_CSV_PATH", "./data/alt.csv")
    
    # API settings
    API_URL: str = os.getenv("API_URL", "http://localhost:8000")
    
    # Server settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    UI_HOST: str = os.getenv("UI_HOST", "0.0.0.0") 
    UI_PORT: int = int(os.getenv("UI_PORT", "8001"))

config = Config()