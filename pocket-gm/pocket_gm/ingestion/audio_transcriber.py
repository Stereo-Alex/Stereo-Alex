from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "unknown"


@dataclass
class Transcript:
    segments: list[TranscriptSegment]
    language: str

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


def transcribe(
    audio_path: Path,
    model_size: str = "base",
    language: str = "en",
    device: str = "cpu",
    diarize: bool = False,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper is required: pip install faster-whisper")

    if diarize:
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            raise ImportError(
                "Speaker diarization requires pyannote.audio.\n"
                "Install it with: pip install pyannote.audio\n"
                "You also need to accept the HuggingFace model licence at:\n"
                "  https://huggingface.co/pyannote/speaker-diarization\n"
                "Alternatively, set diarize=False (the default) to skip diarization."
            )

    model = WhisperModel(model_size, device=device, compute_type="int8")
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language if language != "auto" else None,
        word_timestamps=True,
    )

    segments = [
        TranscriptSegment(start=seg.start, end=seg.end, text=seg.text)
        for seg in segments_iter
    ]

    return Transcript(segments=segments, language=info.language)


def save_transcript(transcript: Transcript, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    txt_path.write_text(transcript.full_text, encoding="utf-8")

    data = {
        "language": transcript.language,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in transcript.segments
        ],
    }
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return txt_path, json_path


def load_transcript_json(json_path: Path) -> Transcript:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = [TranscriptSegment(**s) for s in data["segments"]]
    return Transcript(segments=segments, language=data.get("language", "en"))
