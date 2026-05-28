from __future__ import annotations

import os


class ClaudeClient:
    """LLM provider backed by the Anthropic Claude API."""

    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Claude provider requires the anthropic SDK.\n"
                "Install with: pip install 'pocket-gm[web]'"
            )
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY in your "
                "environment or pass api_key explicitly."
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=(
                "You are a precise assistant for a tabletop RPG Game Master. "
                "Answer only from the provided sources. Never invent information."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def is_available(self) -> bool:
        return bool(self._api_key)
