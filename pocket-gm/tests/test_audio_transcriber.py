import json
import pytest
from pathlib import Path
from pocket_gm.ingestion.audio_transcriber import Transcript, TranscriptSegment, save_transcript, load_transcript_json


def make_transcript() -> Transcript:
    return Transcript(
        segments=[
            TranscriptSegment(start=0.0, end=5.2, text="The party enters the fort."),
            TranscriptSegment(start=5.2, end=10.8, text="They fight the guards."),
            TranscriptSegment(start=10.8, end=18.0, text="The Stag Lord appears on the ramparts."),
        ],
        language="en",
    )


def test_full_text_joins_segments():
    t = make_transcript()
    assert "The party enters the fort." in t.full_text
    assert "Stag Lord" in t.full_text


def test_save_and_load_roundtrip(tmp_path):
    t = make_transcript()
    save_transcript(t, tmp_path, "session_01_2025-01-01")

    txt = (tmp_path / "session_01_2025-01-01.txt").read_text()
    assert "Stag Lord" in txt

    loaded = load_transcript_json(tmp_path / "session_01_2025-01-01.json")
    assert len(loaded.segments) == 3
    assert loaded.segments[0].start == 0.0
    assert loaded.segments[2].text == "The Stag Lord appears on the ramparts."
    assert loaded.language == "en"


def test_save_creates_both_files(tmp_path):
    t = make_transcript()
    txt_path, json_path = save_transcript(t, tmp_path, "test")
    assert txt_path.exists()
    assert json_path.exists()


def test_load_json_segments_have_timestamps(tmp_path):
    t = make_transcript()
    _, json_path = save_transcript(t, tmp_path, "test")
    loaded = load_transcript_json(json_path)
    assert loaded.segments[1].start == 5.2
    assert loaded.segments[1].end == 10.8
