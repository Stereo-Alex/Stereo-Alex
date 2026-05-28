from __future__ import annotations

from typing import Protocol

from pocket_gm.core.config import LLMConfig


class LLMClient(Protocol):
    """Common interface implemented by every LLM provider."""

    def generate(self, prompt: str, temperature: float = ..., max_tokens: int = ...) -> str: ...
    def is_available(self) -> bool: ...


def build_llm_client(cfg: LLMConfig) -> LLMClient:
    """Construct the LLM client selected by ``cfg.provider``.

    Supported providers:
      - "ollama" (default): local Ollama server at ``cfg.base_url``
      - "claude": Anthropic Claude API (reads ANTHROPIC_API_KEY from env)
    """
    provider = (cfg.provider or "ollama").lower()

    if provider == "ollama":
        from pocket_gm.synthesis.ollama_client import OllamaClient
        return OllamaClient(base_url=cfg.base_url, model=cfg.model)

    if provider in ("claude", "anthropic"):
        from pocket_gm.synthesis.claude_client import ClaudeClient
        # The default cfg.model ("phi3:mini") is an Ollama name; only forward
        # the configured model to Claude when it actually looks like one.
        model = cfg.model if cfg.model.startswith("claude") else "claude-haiku-4-5-20251001"
        return ClaudeClient(model=model)

    raise ValueError(
        f"Unknown LLM provider '{cfg.provider}'. Supported: 'ollama', 'claude'."
    )
