from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi3:mini"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running: https://ollama.ai"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama error {e.response.status_code}: {e.response.text}")

    def is_available(self) -> bool:
        try:
            httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return True
        except Exception:
            return False
