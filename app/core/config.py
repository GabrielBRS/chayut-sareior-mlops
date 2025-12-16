import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    PROJECT_NAME: str = "Ortzion AI - MLOps API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_STORAGE_PATH: str = os.path.join(BASE_DIR, "models_store")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()