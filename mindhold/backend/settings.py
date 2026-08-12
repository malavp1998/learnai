"""
Centralized, validated config — replaces scattered `os.environ["X"]` calls
across db.py/embeddings.py/chat.py.

Why this matters: with raw `os.environ[...]`, a missing key blows up deep
inside whichever function first needs it (e.g. mid-request, inside a
Groq client constructor) with a generic KeyError. `pydantic-settings`
reads all required env vars ONCE at import time and fails immediately
with a clear message listing exactly which ones are missing — a startup
check instead of a runtime surprise.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives in the project root (mindhold/), one level above this file's
# backend/ directory — resolved relative to this file so it's found
# regardless of the working directory the server is launched from.
ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    groq_api_key: str
    jina_api_key: str
    database_url: str


settings = Settings()
