from __future__ import annotations

import io
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _make_googleapiclient_mock():
    """Inject a minimal mock of googleapiclient into sys.modules."""
    mock_module = ModuleType("googleapiclient")
    mock_http = ModuleType("googleapiclient.http")
    mock_http.MediaIoBaseDownload = MagicMock()
    mock_module.http = mock_http
    sys.modules.setdefault("googleapiclient", mock_module)
    sys.modules.setdefault("googleapiclient.http", mock_http)
    return mock_http.MediaIoBaseDownload


_MediaIoBaseDownload = _make_googleapiclient_mock()


from pocket_gm.ingestion.drive_loader import (  # noqa: E402
    ManifestEntry,
    _load_manifest,
    _save_manifest,
    download_file,
    list_folder,
    parse_drive_id,
    sync_folder,
)


# ── parse_drive_id ────────────────────────────────────────────────────────────

def test_parse_drive_id_folder_url():
    url = "https://drive.google.com/drive/folders/1AbCdEfGhIjK?usp=sharing"
    assert parse_drive_id(url) == "1AbCdEfGhIjK"


def test_parse_drive_id_file_url():
    url = "https://drive.google.com/file/d/1AbCdEfGhIjK/view"
    assert parse_drive_id(url) == "1AbCdEfGhIjK"


def test_parse_drive_id_bare_id():
    assert parse_drive_id("  1AbCdEfGhIjK  ") == "1AbCdEfGhIjK"


def test_parse_drive_id_with_dashes_underscores():
    url = "https://drive.google.com/drive/folders/1A_B-CdEfGhIjK"
    assert parse_drive_id(url) == "1A_B-CdEfGhIjK"


# ── Manifest ──────────────────────────────────────────────────────────────────

def test_load_manifest_missing_file(tmp_path):
    result = _load_manifest(tmp_path / "nonexistent.json")
    assert result == {}


def test_save_and_load_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    entry = ManifestEntry(
        file_id="file123",
        name="kingmaker.pdf",
        md5="abc123",
        local_path=str(tmp_path / "kingmaker.pdf"),
    )
    _save_manifest(manifest_path, {"file123": entry})

    loaded = _load_manifest(manifest_path)
    assert "file123" in loaded
    assert loaded["file123"].md5 == "abc123"
    assert loaded["file123"].name == "kingmaker.pdf"


def test_save_manifest_creates_parent_dirs(tmp_path):
    manifest_path = tmp_path / "nested" / "dir" / "manifest.json"
    _save_manifest(manifest_path, {})
    assert manifest_path.exists()


# ── list_folder ───────────────────────────────────────────────────────────────

def _make_service(files: list[dict], next_page_token: str | None = None):
    """Build a minimal mock Drive service."""
    service = MagicMock()
    resp = {"files": files}
    if next_page_token:
        resp["nextPageToken"] = next_page_token
    service.files().list().execute.return_value = resp
    return service


def test_list_folder_pdf():
    files = [{"id": "1", "name": "sourcebook.pdf", "mimeType": "application/pdf", "md5Checksum": "aa"}]
    service = _make_service(files)
    result = list_folder(service, "folder123")
    assert len(result) == 1
    assert result[0]["name"] == "sourcebook.pdf"


def test_list_folder_excludes_unsupported():
    files = [
        {"id": "1", "name": "image.png", "mimeType": "image/png"},
        {"id": "2", "name": "notes.txt", "mimeType": "text/plain", "md5Checksum": "bb"},
    ]
    service = _make_service(files)
    result = list_folder(service, "folder123")
    assert len(result) == 1
    assert result[0]["name"] == "notes.txt"


def test_list_folder_audio_excluded_by_default():
    files = [{"id": "1", "name": "session1.mp3", "mimeType": "audio/mpeg"}]
    service = _make_service(files)
    result = list_folder(service, "folder123", include_audio=False)
    assert result == []


def test_list_folder_audio_included_when_flagged():
    files = [{"id": "1", "name": "session1.mp3", "mimeType": "audio/mpeg"}]
    service = _make_service(files)
    result = list_folder(service, "folder123", include_audio=True)
    assert len(result) == 1


def test_list_folder_google_doc():
    files = [{"id": "1", "name": "My Notes", "mimeType": "application/vnd.google-apps.document"}]
    service = _make_service(files)
    result = list_folder(service, "folder123")
    assert len(result) == 1


def test_list_folder_pagination():
    """Two pages of results should be combined."""
    service = MagicMock()
    page1 = {
        "files": [{"id": "1", "name": "a.pdf", "mimeType": "application/pdf"}],
        "nextPageToken": "tok",
    }
    page2 = {"files": [{"id": "2", "name": "b.pdf", "mimeType": "application/pdf"}]}
    service.files().list().execute.side_effect = [page1, page2]
    result = list_folder(service, "folder123")
    assert len(result) == 2


# ── download_file ─────────────────────────────────────────────────────────────

def test_download_file_binary(tmp_path):
    dest = tmp_path / "file.pdf"
    service = MagicMock()

    fake_content = b"%PDF-1.4 fake content"
    buf_mock = MagicMock(spec=io.BytesIO)
    buf_mock.getvalue.return_value = fake_content

    mock_downloader = MagicMock()
    mock_downloader.next_chunk.side_effect = [(None, False), (None, True)]
    _MediaIoBaseDownload.return_value = mock_downloader

    with patch("io.BytesIO", return_value=buf_mock):
        download_file(service, "file_id", "application/pdf", dest)

    assert dest.exists()
    assert dest.read_bytes() == fake_content


def test_download_google_doc(tmp_path):
    dest = tmp_path / "notes.txt"
    service = MagicMock()
    service.files().export_media().execute.return_value = b"Exported text content"

    download_file(service, "doc_id", "application/vnd.google-apps.document", dest)

    assert dest.exists()
    assert dest.read_bytes() == b"Exported text content"


def test_download_google_doc_string_response(tmp_path):
    """export_media may return str in some SDK versions."""
    dest = tmp_path / "notes.txt"
    service = MagicMock()
    service.files().export_media().execute.return_value = "Exported text content"

    download_file(service, "doc_id", "application/vnd.google-apps.document", dest)

    assert dest.read_bytes() == b"Exported text content"


# ── sync_folder ───────────────────────────────────────────────────────────────

def _mock_service_for_sync(files):
    service = MagicMock()
    service.files().list().execute.return_value = {"files": files}
    return service


def test_sync_folder_downloads_new_files(tmp_path):
    files = [{"id": "f1", "name": "rules.pdf", "mimeType": "application/pdf", "md5Checksum": "aaa"}]
    service = _mock_service_for_sync(files)
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"

    with patch("pocket_gm.ingestion.drive_loader.download_file") as mock_dl:
        def fake_download(svc, fid, mime, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake pdf")
        mock_dl.side_effect = fake_download

        result = sync_folder(service, "folder_id", cache_dir, manifest_path)

    assert len(result.downloaded) == 1
    assert result.skipped == 0
    assert result.errors == []


def test_sync_folder_skips_unchanged(tmp_path):
    files = [{"id": "f1", "name": "rules.pdf", "mimeType": "application/pdf", "md5Checksum": "aaa"}]
    service = _mock_service_for_sync(files)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dest = cache_dir / "rules.pdf"
    dest.write_bytes(b"fake pdf")

    manifest_path = tmp_path / "manifest.json"
    existing_entry = ManifestEntry(file_id="f1", name="rules.pdf", md5="aaa", local_path=str(dest))
    _save_manifest(manifest_path, {"f1": existing_entry})

    with patch("pocket_gm.ingestion.drive_loader.download_file") as mock_dl:
        result = sync_folder(service, "folder_id", cache_dir, manifest_path)

    mock_dl.assert_not_called()
    assert result.skipped == 1
    assert result.downloaded == []


def test_sync_folder_redownloads_changed_md5(tmp_path):
    files = [{"id": "f1", "name": "rules.pdf", "mimeType": "application/pdf", "md5Checksum": "bbb"}]
    service = _mock_service_for_sync(files)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dest = cache_dir / "rules.pdf"
    dest.write_bytes(b"old content")

    manifest_path = tmp_path / "manifest.json"
    old_entry = ManifestEntry(file_id="f1", name="rules.pdf", md5="aaa", local_path=str(dest))
    _save_manifest(manifest_path, {"f1": old_entry})

    with patch("pocket_gm.ingestion.drive_loader.download_file") as mock_dl:
        def fake_download(svc, fid, mime, dest_path):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"new content")
        mock_dl.side_effect = fake_download

        result = sync_folder(service, "folder_id", cache_dir, manifest_path)

    assert len(result.downloaded) == 1
    assert result.skipped == 0


def test_sync_folder_records_error_on_download_failure(tmp_path):
    files = [{"id": "f1", "name": "broken.pdf", "mimeType": "application/pdf", "md5Checksum": "ccc"}]
    service = _mock_service_for_sync(files)
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"

    with patch("pocket_gm.ingestion.drive_loader.download_file", side_effect=Exception("network error")):
        result = sync_folder(service, "folder_id", cache_dir, manifest_path)

    assert result.errors == ["broken.pdf: network error"]
    assert result.downloaded == []


def test_sync_folder_google_doc_renamed_to_txt(tmp_path):
    files = [{"id": "d1", "name": "My Notes", "mimeType": "application/vnd.google-apps.document", "md5Checksum": ""}]
    service = _mock_service_for_sync(files)
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"

    with patch("pocket_gm.ingestion.drive_loader.download_file") as mock_dl:
        def fake_download(svc, fid, mime, dest_path):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"doc content")
        mock_dl.side_effect = fake_download

        result = sync_folder(service, "folder_id", cache_dir, manifest_path)

    assert len(result.downloaded) == 1
    assert result.downloaded[0].name == "My Notes.txt"


def test_sync_folder_saves_updated_manifest(tmp_path):
    files = [{"id": "f1", "name": "rules.pdf", "mimeType": "application/pdf", "md5Checksum": "zzz"}]
    service = _mock_service_for_sync(files)
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"

    with patch("pocket_gm.ingestion.drive_loader.download_file") as mock_dl:
        def fake_download(svc, fid, mime, dest_path):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"data")
        mock_dl.side_effect = fake_download

        sync_folder(service, "folder_id", cache_dir, manifest_path)

    manifest = _load_manifest(manifest_path)
    assert "f1" in manifest
    assert manifest["f1"].md5 == "zzz"
