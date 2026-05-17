from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ObsidianNote:
    path: Path
    title: str
    body: str                    # cleaned, plain-text body ready for chunking
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)


# ── Regex patterns ────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_WIKILINK_EMBED_RE = re.compile(r"!\[\[([^\]|#]+?)(?:[|#][^\]]*)?\]\]")   # ![[note]] or ![[note|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:\|([^\]]+))?\]\]")           # [[note]] or [[note|alias]]
_CALLOUT_HEADER_RE = re.compile(r"^>\s*\[![\w-]+\][+-]?\s*(.*)$", re.MULTILINE)
_CALLOUT_BODY_RE = re.compile(r"^>\s?", re.MULTILINE)
_DATAVIEW_BLOCK_RE = re.compile(r"```dataview.*?```", re.DOTALL | re.IGNORECASE)
_DATAVIEW_INLINE_RE = re.compile(r"`=\s*[^`]+`")
_INLINE_TAG_RE = re.compile(r"(?<!\S)#([\w/-]+)")                           # #tag but not markdown headings
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_ADMONITION_BLOCK_RE = re.compile(r"```ad-\w+.*?```", re.DOTALL | re.IGNORECASE)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter, return (metadata_dict, body_without_frontmatter)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    try:
        import yaml
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        meta = {}

    body = text[m.end():]
    return meta, body


def _resolve_embed(note_name: str, vault_index: dict[str, Path], visited: set[str], depth: int) -> str:
    """Inline the content of an embedded note, up to 2 levels deep to avoid cycles."""
    if depth > 2 or note_name in visited:
        return f"[embedded: {note_name}]"

    path = vault_index.get(note_name.lower())
    if not path:
        return f"[embedded: {note_name}]"

    visited = visited | {note_name}
    raw = path.read_text(encoding="utf-8", errors="replace")
    _, body = _parse_frontmatter(raw)
    return _clean_body(body, vault_index, visited, depth + 1)


def _clean_body(text: str, vault_index: dict[str, Path], visited: set[str], depth: int = 0) -> str:
    # Strip HTML comments
    text = _HTML_COMMENT_RE.sub("", text)

    # Strip dataview blocks (they're queries, not content)
    text = _DATAVIEW_BLOCK_RE.sub("", text)
    text = _DATAVIEW_INLINE_RE.sub("", text)

    # Strip admonition plugin blocks — keep their inner text
    def expand_admonition(m: re.Match) -> str:
        inner = m.group(0)
        # Extract lines that aren't the fence or the type header
        lines = inner.splitlines()[1:-1]  # strip ```ad-xxx and closing ```
        if lines and re.match(r"title:\s*", lines[0], re.IGNORECASE):
            lines = lines[1:]
        return "\n".join(lines)
    text = _ADMONITION_BLOCK_RE.sub(expand_admonition, text)

    # Expand embedded notes ![[note]] inline
    def expand_embed(m: re.Match) -> str:
        note_name = m.group(1).strip()
        # Skip image/binary embeds
        if re.search(r"\.(png|jpg|jpeg|gif|svg|pdf|mp3|mp4|wav)$", note_name, re.IGNORECASE):
            return ""
        return _resolve_embed(note_name, vault_index, visited, depth)
    text = _WIKILINK_EMBED_RE.sub(expand_embed, text)

    # Convert callout headers: "> [!NOTE] Title" → "Title"
    text = _CALLOUT_HEADER_RE.sub(lambda m: m.group(1).strip(), text)
    # Strip blockquote markers from callout bodies
    text = _CALLOUT_BODY_RE.sub("", text)

    # Replace wikilinks with display text: [[Note|alias]] → alias, [[Note]] → Note
    text = _WIKILINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)

    # Strip inline tags (#tag → keep text without #, or drop entirely)
    # Keep the word so "the #Stag-Lord is..." becomes "the Stag-Lord is..."
    text = _INLINE_TAG_RE.sub(lambda m: m.group(1).replace("/", " "), text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _build_vault_index(vault_path: Path) -> dict[str, Path]:
    """Map lowercase note stem → Path for wikilink resolution."""
    return {p.stem.lower(): p for p in vault_path.rglob("*.md")}


def load_obsidian_vault(vault_path: Path) -> list[ObsidianNote]:
    """Load all notes from an Obsidian vault directory."""
    vault_index = _build_vault_index(vault_path)
    notes: list[ObsidianNote] = []

    for path in sorted(vault_path.rglob("*.md")):
        # Skip Obsidian system folders
        if any(part.startswith(".") for part in path.parts):
            continue

        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, raw_body = _parse_frontmatter(raw)

        tags: list[str] = []
        if isinstance(meta.get("tags"), list):
            tags = [str(t) for t in meta["tags"]]
        elif isinstance(meta.get("tags"), str):
            tags = [meta["tags"]]

        aliases: list[str] = []
        if isinstance(meta.get("aliases"), list):
            aliases = [str(a) for a in meta["aliases"]]
        elif isinstance(meta.get("aliases"), str):
            aliases = [meta["aliases"]]

        title = meta.get("title") or path.stem

        body = _clean_body(raw_body, vault_index, visited={path.stem.lower()})

        if not body.strip():
            continue

        notes.append(ObsidianNote(
            path=path,
            title=title,
            body=body,
            tags=tags,
            aliases=aliases,
            frontmatter=meta,
        ))

    return notes


def load_obsidian_file(file_path: Path, vault_path: Path | None = None) -> ObsidianNote:
    """Load a single Obsidian note, optionally with vault context for wikilink resolution."""
    vault_index = _build_vault_index(vault_path) if vault_path else {file_path.stem.lower(): file_path}
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    meta, raw_body = _parse_frontmatter(raw)

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]

    aliases = meta.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]

    title = meta.get("title") or file_path.stem
    body = _clean_body(raw_body, vault_index, visited={file_path.stem.lower()})

    return ObsidianNote(
        path=file_path,
        title=title,
        body=body,
        tags=list(tags),
        aliases=list(aliases),
        frontmatter=meta,
    )
