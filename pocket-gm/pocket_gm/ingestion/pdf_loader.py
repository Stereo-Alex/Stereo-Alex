from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawChunk:
    text: str
    page: int
    heading: str
    filename: str


def load_pdf(path: Path) -> list[RawChunk]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("pymupdf is required: pip install pymupdf")

    doc = fitz.open(str(path))
    chunks: list[RawChunk] = []
    current_heading = ""

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_texts: list[tuple[str, bool]] = []  # (text, is_heading)

        for block in blocks:
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = span.get("size", 0)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)
                    # Heading heuristic: larger font or bold + short line
                    is_heading = (size >= 12 and is_bold) or (size >= 14)
                    page_texts.append((text, is_heading))

        # Build page text, tracking headings
        page_lines: list[str] = []
        for text, is_heading in page_texts:
            if is_heading and len(text) < 120:
                current_heading = text
            page_lines.append(text)

        if page_lines:
            chunks.append(RawChunk(
                text=" ".join(page_lines),
                page=page_num,
                heading=current_heading,
                filename=path.name,
            ))

    doc.close()
    return chunks
