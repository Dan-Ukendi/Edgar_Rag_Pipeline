import requests

from .base import LLMClient


class OllamaLLM(LLMClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", max_tokens: int = 1024):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": self._max_tokens},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def is_reachable(self) -> bool:
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=2)
            return response.ok
        except requests.RequestException:
            return False
