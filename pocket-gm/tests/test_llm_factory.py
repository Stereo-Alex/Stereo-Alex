from __future__ import annotations

import pytest

from pocket_gm.core.config import LLMConfig
from pocket_gm.synthesis.llm import build_llm_client


def test_default_provider_builds_ollama_client():
    cfg = LLMConfig()  # provider defaults to "ollama"
    client = build_llm_client(cfg)
    from pocket_gm.synthesis.ollama_client import OllamaClient
    assert isinstance(client, OllamaClient)
    assert client.model == "phi3:mini"


def test_unknown_provider_raises():
    cfg = LLMConfig(provider="banana")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm_client(cfg)


def test_claude_provider_builds_claude_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = LLMConfig(provider="claude", model="claude-haiku-4-5-20251001")
    client = build_llm_client(cfg)
    from pocket_gm.synthesis.claude_client import ClaudeClient
    assert isinstance(client, ClaudeClient)
    assert client.is_available()


def test_claude_provider_falls_back_when_model_is_ollama_name(monkeypatch):
    """A leftover Ollama model name must not be forwarded to the Claude API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = LLMConfig(provider="claude", model="phi3:mini")
    client = build_llm_client(cfg)
    assert client.model.startswith("claude")


def test_claude_provider_without_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = LLMConfig(provider="claude")
    with pytest.raises(RuntimeError, match="No Anthropic API key"):
        build_llm_client(cfg)
