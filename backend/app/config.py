from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Marine Debris Sonar API"
    DATABASE_URL: str = "postgresql+psycopg://postgres:change_me_in_env@localhost:5432/marine_debris_db"
    ML_SERVICE_URL: str = "http://127.0.0.1:8001"    
    class Config:
        env_file = ".env"

settings = Settings()
