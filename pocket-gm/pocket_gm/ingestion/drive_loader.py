from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


# ── Folder ID extraction ──────────────────────────────────────────────────────

_FOLDER_ID_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
_FILE_ID_RE   = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")


def parse_drive_id(url_or_id: str) -> str:
    """Extract a Drive resource ID from a URL or return the raw ID."""
    for pattern in (_FOLDER_ID_RE, _FILE_ID_RE):
        m = pattern.search(url_or_id)
        if m:
            return m.group(1)
    # Assume it's already a bare ID
    return url_or_id.strip()


# ── Auth ──────────────────────────────────────────────────────────────────────

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _get_credentials(token_path: Path, credentials_path: Path):
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
    except ImportError:
        raise ImportError(
            "Google Drive support requires extra dependencies.\n"
            "Install with: pip install 'pocket-gm[gdrive]'"
        )

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Google credentials file not found: {credentials_path}\n"
                    "Download it from https://console.cloud.google.com/ → APIs & Services → Credentials"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), _SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())

    return creds


def build_service(token_path: Path, credentials_path: Path):
    try:
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "Google Drive support requires extra dependencies.\n"
            "Install with: pip install 'pocket-gm[gdrive]'"
        )
    creds = _get_credentials(token_path, credentials_path)
    return build("drive", "v3", credentials=creds)


# ── Manifest (MD5 cache) ──────────────────────────────────────────────────────

@dataclass
class ManifestEntry:
    file_id: str
    name: str
    md5: str
    local_path: str


def _load_manifest(manifest_path: Path) -> dict[str, ManifestEntry]:
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        data = json.load(f)
    return {k: ManifestEntry(**v) for k, v in data.items()}


def _save_manifest(manifest_path: Path, manifest: dict[str, ManifestEntry]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({k: vars(v) for k, v in manifest.items()}, f, indent=2)


# ── Listing ───────────────────────────────────────────────────────────────────

_SUPPORTED_MIME = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    # Google Docs → export as plain text
    "application/vnd.google-apps.document",
}

_AUDIO_MIME = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/mp4", "audio/m4a", "audio/flac", "audio/ogg",
    "video/mp4",
}


def list_folder(service, folder_id: str, include_audio: bool = False) -> list[dict]:
    """Return all supported files in a Drive folder (non-recursive)."""
    results = []
    page_token = None

    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, md5Checksum, size)",
            pageToken=page_token,
            pageSize=100,
        ).execute()

        for f in resp.get("files", []):
            mime = f.get("mimeType", "")
            if mime in _SUPPORTED_MIME:
                results.append(f)
            elif include_audio and mime in _AUDIO_MIME:
                results.append(f)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


# ── Downloading ───────────────────────────────────────────────────────────────

def download_file(service, file_id: str, mime_type: str, dest: Path) -> None:
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError:
        raise ImportError("Install with: pip install 'pocket-gm[gdrive]'")

    import io

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Google Docs → export as plain text
    if mime_type == "application/vnd.google-apps.document":
        data = service.files().export_media(
            fileId=file_id, mimeType="text/plain"
        ).execute()
        dest.write_bytes(data if isinstance(data, bytes) else data.encode())
        return

    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest.write_bytes(buf.getvalue())


# ── High-level sync ───────────────────────────────────────────────────────────

@dataclass
class SyncResult:
    downloaded: list[Path]
    skipped: int
    errors: list[str]


def sync_folder(
    service,
    folder_id: str,
    cache_dir: Path,
    manifest_path: Path,
    include_audio: bool = False,
) -> SyncResult:
    """Download new/changed files from a Drive folder to local cache."""
    manifest = _load_manifest(manifest_path)
    files = list_folder(service, folder_id, include_audio=include_audio)

    downloaded: list[Path] = []
    skipped = 0
    errors: list[str] = []

    for f in files:
        file_id = f["id"]
        name = f["name"]
        mime = f.get("mimeType", "")
        remote_md5 = f.get("md5Checksum", "")

        # Determine local filename
        if mime == "application/vnd.google-apps.document":
            local_name = Path(name).stem + ".txt"
        else:
            local_name = name
        dest = cache_dir / local_name

        # Skip if unchanged
        existing = manifest.get(file_id)
        if existing and existing.md5 == remote_md5 and dest.exists():
            skipped += 1
            continue

        try:
            download_file(service, file_id, mime, dest)
            manifest[file_id] = ManifestEntry(
                file_id=file_id,
                name=local_name,
                md5=remote_md5,
                local_path=str(dest),
            )
            downloaded.append(dest)
        except Exception as e:
            errors.append(f"{name}: {e}")

    _save_manifest(manifest_path, manifest)
    return SyncResult(downloaded=downloaded, skipped=skipped, errors=errors)
