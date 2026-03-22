import os
from pathlib import Path

from dotenv import load_dotenv

root_path = Path(__file__).parent.parent
load_dotenv(dotenv_path=root_path / ".env", override=True)


def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")


__all__ = [
    "get_openai_api_key",
]
