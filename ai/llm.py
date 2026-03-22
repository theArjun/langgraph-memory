from langchain_openai import ChatOpenAI

from .config import get_openai_api_key
from .models import LLMModels

llm = ChatOpenAI(model=LLMModels.GPT_4O, api_key=get_openai_api_key())

__all__ = [
    "llm",
]
