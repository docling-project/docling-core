"""Tests for the audio and video of tracks, and their place in DocLang archives."""

import logging
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from docling_core.types.doc import DocItemLabel, DoclingDocument, MediaRef, TrackItem, TrackSource
from docling_core.utils.settings import settings


def _add_track(doc: DoclingDocument, audio: MediaRef | None = None, video: MediaRef | None = None) -> TrackItem:
    track = doc.add_track(audio=audio, video=video)
    doc.add_text(label=DocItemLabel.TEXT, text="hi", parent=track, source=TrackSource(start_time=0, end_time=1))
    return track


def _members(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return sorted(name for name in zf.namelist() if name.startswith(("audio/", "video/")))


@pytest.fixture
def media_files(tmp_path: Path) -> dict[str, Path]:
    files = {"mp3": tmp_path / "talk.mp3", "mp4": tmp_path / "clip.mp4", "bin": tmp_path / "clip.bin"}
    for path in files.values():
        path.write_bytes(b"not decoded")
    return files


def test_media_ref_checks(media_files):
    assert MediaRef.from_file(media_files["mp3"]).mimetype == "audio/mpeg"
    with pytest.raises(ValueError, match="Cannot tell"):
        MediaRef.from_file(media_files["bin"])
    with pytest.raises(ValidationError, match="Unsupported media type"):
        MediaRef(mimetype="text/plain", uri=Path("a.txt"))
    with pytest.raises(ValidationError, match="data: URIs"):
        MediaRef(mimetype="audio/mpeg", uri="data:audio/mpeg;base64,AAAA")
    with pytest.raises(ValidationError, match="Not an audio"):
        TrackItem(self_ref="#/tracks/0", audio=MediaRef(mimetype="video/mp4", uri=Path("a.mp4")))
    with pytest.raises(ValidationError, match="Not a video"):
        TrackItem(self_ref="#/tracks/0", video=MediaRef(mimetype="audio/mpeg", uri=Path("a.mp3")))


def test_archive_round_trip(tmp_path, media_files):
    doc = DoclingDocument(name="media")
    # a loose timed item is an implicit track, which counts in the numbering
    doc.add_text(label=DocItemLabel.TEXT, text="loose", source=TrackSource(start_time=0, end_time=1))
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="untimed")
    _add_track(doc, audio=MediaRef.from_file(media_files["mp3"]))
    # the archive name takes the extension of the media type
    _add_track(doc, video=MediaRef.from_file(media_files["bin"], mimetype="video/mp4"))

    archive = tmp_path / "out.dclx"
    doc.save_as_doclang_archive(archive)
    assert _members(archive) == ["audio/2.mp3", "video/3.mp4"]

    back = DoclingDocument.load_from_doclang_archive(archive, artifacts_dir=tmp_path / "art")
    media = [(track.audio, track.video) for track in back.tracks]
    assert [(a and a.mimetype, v and v.mimetype) for a, v in media] == [
        (None, None),
        ("audio/mpeg", None),
        (None, "video/mp4"),
    ]
    assert media[1][0].uri.read_bytes() == b"not decoded"

    # media from a loaded archive is written again
    again = tmp_path / "again.dclx"
    back.save_as_doclang_archive(again)
    assert _members(again) == ["audio/2.mp3", "video/3.mp4"]


def test_local_paths_need_opt_in(tmp_path, media_files, caplog):
    doc = DoclingDocument(name="media")
    _add_track(doc, audio=MediaRef.from_file(media_files["mp3"]))
    # a document loaded from JSON carries its paths as content only
    loaded = DoclingDocument.model_validate_json(doc.model_dump_json())
    assert loaded.tracks[0].audio.uri == media_files["mp3"]

    with caplog.at_level(logging.WARNING):
        loaded.save_as_doclang_archive(tmp_path / "default.dclx")
    assert _members(tmp_path / "default.dclx") == []
    assert "allow_local_media" in caplog.text

    loaded.save_as_doclang_archive(tmp_path / "opt_in.dclx", allow_local_media=True)
    assert _members(tmp_path / "opt_in.dclx") == ["audio/1.mp3"]

    orig = settings.allow_media_file_uri
    try:
        settings.allow_media_file_uri = True
        loaded.save_as_doclang_archive(tmp_path / "setting.dclx")
    finally:
        settings.allow_media_file_uri = orig
    assert _members(tmp_path / "setting.dclx") == ["audio/1.mp3"]


def test_remote_media_is_not_fetched(tmp_path):
    doc = DoclingDocument(name="media")
    _add_track(doc, audio=MediaRef(mimetype="audio/mpeg", uri="https://example.com/a.mp3"))
    doc.save_as_doclang_archive(tmp_path / "out.dclx", allow_local_media=True)
    assert _members(tmp_path / "out.dclx") == []


def test_archive_media_that_does_not_fit_is_ignored(tmp_path, caplog):
    doc = DoclingDocument(name="media")
    _add_track(doc)
    archive = tmp_path / "out.dclx"
    doc.save_as_doclang_archive(archive)
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("audio/2.mp3", b"no such track")
        zf.writestr("audio/x.mp3", b"no number")
        zf.writestr("audio/1.exe", b"not media")
        zf.writestr("video/1.mp3", b"not a video")

    with caplog.at_level(logging.WARNING):
        back = DoclingDocument.load_from_doclang_archive(archive, artifacts_dir=tmp_path / "art")
    assert back.tracks[0].audio is None
    assert back.tracks[0].video is None
    assert caplog.text.count("Ignoring") == 4
