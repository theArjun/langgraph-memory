import os
from pathlib import Path

from dotenv import load_dotenv

root_path = Path(__file__).parent.parent
load_dotenv(dotenv_path=root_path / ".env", override=True)


def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")


def get_database_url():
    return os.getenv("DATABASE_URL")


__all__ = [
    "get_openai_api_key",
    "get_database_url",
]
