from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://webhook:webhook@localhost:5432/webhookdb"

    # ── LLM provider selection ────────────────────────────────────────────────
    # Options: "groq" (default, free), "gemini" (free), "anthropic" (paid)
    llm_provider: str = "groq"

    # Groq — free at console.groq.com
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Alternative Groq models:
    #   qwen-qwq-32b          (Qwen, very good reasoning)
    #   llama-3.1-8b-instant  (faster, lighter)
    #   mixtral-8x7b-32768    (good at structured output)

    # Google Gemini — free at aistudio.google.com
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Anthropic — paid (original)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"


settings = Settings()
