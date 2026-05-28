from pathlib import Path

import yaml

from pocket_gm.core.config import Config, load_config, save_config


def test_default_config():
    cfg = load_config(Path("/nonexistent/path/config.yaml"))
    assert cfg.embedding.model == "all-MiniLM-L6-v2"
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.relevance_threshold == 0.35
    assert cfg.llm.model == "phi3:mini"
    assert cfg.whisper.model_size == "base"


def test_round_trip(tmp_path):
    cfg = Config()
    cfg.llm.model = "mistral:7b-instruct"
    cfg.retrieval.top_k = 8
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.llm.model == "mistral:7b-instruct"
    assert loaded.retrieval.top_k == 8


def test_shipped_example_config_loads():
    """The example config the README tells users to copy must load cleanly."""
    example = Path(__file__).parent.parent / "config.example.yaml"
    cfg = load_config(example)
    assert cfg.whisper.diarize is False
    assert cfg.chunking.sessions.time_window_seconds == 120


def test_unknown_keys_are_ignored(tmp_path):
    """A config with an extra/future key should not crash every command."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({
        "whisper": {"model_size": "small", "some_future_knob": True},
        "retrieval": {"top_k": 7, "totally_unknown": "x"},
    }))
    cfg = load_config(path)
    assert cfg.whisper.model_size == "small"
    assert cfg.retrieval.top_k == 7


def test_diarize_and_time_window_round_trip(tmp_path):
    cfg = Config()
    cfg.whisper.diarize = True
    cfg.chunking.sessions.time_window_seconds = 90
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.whisper.diarize is True
    assert loaded.chunking.sessions.time_window_seconds == 90
