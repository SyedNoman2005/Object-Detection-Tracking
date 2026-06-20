import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Real-Time AI Surveillance System"
    API_V1_STR: str = "/api/v1"
    
    # Model configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "yolov8n.pt")
    DEVICE: str = os.getenv("DEVICE", "cpu")  # 'cuda' or 'cpu'
    
    # Tracking configuration
    COUNTING_LINE_Y_RATIO: float = 0.5  # Position of the counting line
    
    # Storage configuration
    MEDIA_ROOT: str = os.getenv("MEDIA_ROOT", "media")
    UPLOAD_DIR: str = os.path.join(MEDIA_ROOT, "uploads")
    OUTPUT_DIR: str = os.path.join(MEDIA_ROOT, "outputs")
    
    # DB configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    
    # Redis for Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
