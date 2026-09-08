import os

from dotenv import load_dotenv
from groq import Groq

from .base import LLMClient

load_dotenv()


class GroqLLM(LLMClient):
    def __init__(self, model: str, max_tokens: int = 1024):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (checked environment and .env file)")
        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content
