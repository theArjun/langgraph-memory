from langchain.embeddings import init_embeddings

from .config import get_openai_api_key
from .models import EmbeddingModels, Providers

embeddings = init_embeddings(
    model=EmbeddingModels.TEXT_EMBEDDING_3_SMALL,
    provider=Providers.OPENAI,
    api_key=get_openai_api_key(),
)

__all__ = [
    "embeddings",
]
