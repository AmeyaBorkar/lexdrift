from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "sqlite+aiosqlite:///./lexdrift.db"

    # SEC EDGAR
    sec_user_agent: str = "LexDrift research@example.com"
    sec_rate_limit: int = 10

    # Celery / Redis
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # NLP
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    finbert_model: str = "ProsusAI/finbert"
    use_finbert_sentiment: bool = True
    spacy_model: str = "en_core_web_sm"
    embedding_chunk_size: int = 2000
    embedding_chunk_overlap: int = 400

    # Phrase discovery
    ngram_min_freq: int = 2
    ngram_top_k: int = 30
    priority_phrases_path: str = "data/default_phrases.json"

    # LLM (Groq — free tier, Llama 3.3 70B)
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_enabled: bool = True

    # Alerts
    drift_threshold: float = 0.15
    sentiment_stddev_multiplier: float = 2.0


settings = Settings()
