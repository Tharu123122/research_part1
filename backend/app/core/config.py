import os

from pydantic import BaseModel


class Settings(BaseModel):
    service_name: str = "oral-cancer-risk-api"
    cors_origins: list[str] = [
        "http://localhost:8081",
        "http://localhost:19006",
        "http://localhost:3000",
    ]
    disclaimer: str = "This is AI-assisted screening support and not a medical diagnosis."
    mongodb_uri: str | None = os.getenv("MONGODB_URI")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "oral_cancer_screening")
    predictions_collection: str = os.getenv("MONGODB_PREDICTIONS_COLLECTION", "predictions")
    connection_timeout_ms: int = int(os.getenv("MONGODB_TIMEOUT_MS", "5000"))


settings = Settings()
