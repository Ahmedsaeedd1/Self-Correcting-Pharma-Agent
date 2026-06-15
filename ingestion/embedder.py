import os
from typing import List
from google import genai
from langchain_core.embeddings import Embeddings

from core.config import EMBEDDING_MODEL

class GeminiEmbeddings(Embeddings):
    """
    Thin LangChain-compatible wrapper around the google-genai SDK.
    Uses the stable v1 REST endpoint, unlike langchain-google-genai which
    defaults to v1beta where text-embedding-004 is not available.
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        # The genai.Client picks up GOOGLE_API_KEY from the environment automatically
        self.client = genai.Client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings using the Gemini model.
        """
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single string query.
        """
        return self.embed_documents([text])[0]
