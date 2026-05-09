from pydantic import BaseModel


class Settings(BaseModel):
    service_name: str = "oral-cancer-risk-api"
    cors_origins: list[str] = [
        "http://localhost:8081",
        "http://localhost:19006",
        "http://localhost:3000",
    ]
    disclaimer: str = "This is AI-assisted screening support and not a medical diagnosis."


settings = Settings()
