# SPDX-License-Identifier: MIT
"""Local file copies keep their ownership contract without buffering whole inputs."""

import errno
import os
import stat
import tempfile
import tracemalloc
from io import BytesIO
from pathlib import Path

import pytest

from docling_core.utils.file import resolve_file_source, resolve_source_to_path


@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("name", ["document.pdf", "原始文档.pdf"])
@pytest.mark.parametrize("content", [b"", b"%PDF-1.7\n" + b"data" * 300_001], ids=["empty", "multiple_chunks"])
def test_local_copy_is_independent(tmp_path, as_string, name, content):
    """New copies preserve bytes and can be edited or deleted independently."""
    source = tmp_path / name
    source.write_bytes(content)
    workdir = tmp_path / "输出" / "nested"

    result = resolve_source_to_path(str(source) if as_string else source, workdir=workdir)

    assert result == workdir / name
    assert result.read_bytes() == content
    assert not result.samefile(source)
    result.write_bytes(b"changed copy")
    assert source.read_bytes() == content
    result.unlink()
    assert source.read_bytes() == content


@pytest.mark.parametrize("legacy", [False, True])
def test_automatic_temporary_copy_persists(tmp_path, monkeypatch, legacy):
    """The caller owns the returned temporary copy, including through the old API."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    source = tmp_path / "document.pdf"
    source.write_bytes(b"original")

    if legacy:
        with pytest.warns(DeprecationWarning):
            result = resolve_file_source(source)
    else:
        result = resolve_source_to_path(source)

    assert result.parent != source.parent
    assert result.parent.parent == tmp_path
    assert result.read_bytes() == b"original"
    assert not result.samefile(source)
    result.unlink()
    result.parent.rmdir()
    assert source.read_bytes() == b"original"


def test_existing_copy_is_overwritten_in_place(tmp_path):
    """Overwrite still truncates the file and keeps the existing file identity."""
    source = tmp_path / "document.pdf"
    source.write_bytes(b"short")
    workdir = tmp_path / "output"
    workdir.mkdir()
    destination = workdir / source.name
    destination.write_bytes(b"a much longer previous copy")
    before = destination.stat()

    assert resolve_source_to_path(source, workdir=workdir) == destination

    assert destination.read_bytes() == b"short"
    assert destination.stat().st_ino == before.st_ino
    assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IMODE(before.st_mode)
    assert source.read_bytes() == b"short"


@pytest.mark.parametrize("alias", ["same_path", "hardlink", "symlink"])
def test_source_destination_alias_keeps_content(tmp_path, alias):
    """A source alias must not be truncated before its content is read."""
    source = tmp_path / "document.pdf"
    source.write_bytes(b"source content")
    if alias == "same_path":
        destination = source
    else:
        workdir = tmp_path / "output"
        workdir.mkdir()
        destination = workdir / source.name
        try:
            if alias == "hardlink":
                destination.hardlink_to(source)
            else:
                destination.symlink_to(source)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM, errno.ENOTSUP, errno.ENOSYS) or (
                os.name == "nt" and exc.winerror == 1314
            ):
                pytest.skip(f"This environment cannot create {alias}: {exc}")
            raise
    os.utime(source, (1_000_000_000, 1_000_000_000))

    result = resolve_source_to_path(source, workdir=destination.parent)

    assert result == destination
    assert result.samefile(source)
    assert source.read_bytes() == b"source content"
    assert source.stat().st_mtime > 1_000_000_000


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only attribute behavior")
def test_read_only_same_path_still_raises(tmp_path):
    """The same-path case must not become a no-op that hides a write failure."""
    source = tmp_path / "document.pdf"
    source.write_bytes(b"source content")
    source.chmod(stat.S_IREAD)
    try:
        with pytest.raises(PermissionError):
            resolve_source_to_path(source, workdir=tmp_path)
        assert source.read_bytes() == b"source content"
    finally:
        source.chmod(stat.S_IWRITE | stat.S_IREAD)


@pytest.mark.parametrize("directory_source", [False, True])
def test_unreadable_source_does_not_create_workdir(tmp_path, directory_source):
    """Opening an absent file or directory must fail before destination creation."""
    source = tmp_path / "source"
    if directory_source:
        source.mkdir()
    workdir = tmp_path / "not-created"

    with pytest.raises(OSError):
        resolve_source_to_path(source, workdir=workdir)

    assert not workdir.exists()


@pytest.mark.parametrize("existing_destination", [False, True])
def test_source_read_error_does_not_leave_damaged_copy(tmp_path, monkeypatch, existing_destination):
    """A mid-read I/O error must not replace a previous copy or expose a partial new copy."""
    source = tmp_path / "document.pdf"
    source.write_bytes(b"original source")
    workdir = tmp_path / "output"
    workdir.mkdir()
    destination = workdir / source.name
    if existing_destination:
        destination.write_bytes(b"previous copy")

    class FailingReader(BytesIO):
        """Simulate a file that fails after its first bounded read."""

        def read(self, size=-1):
            if size < 0 or self.tell():
                raise OSError("source read failed")
            return super().read(size)

    original_open = Path.open

    def open_with_read_failure(path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            return FailingReader(b"first readable block")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_with_read_failure)
        with pytest.raises(OSError, match="source read failed"):
            resolve_source_to_path(source, workdir=workdir)

    if existing_destination:
        assert destination.read_bytes() == b"previous copy"
    else:
        assert not destination.exists()
    assert source.read_bytes() == b"original source"


def test_new_local_copy_has_bounded_python_memory(tmp_path):
    """Copying a large local document must not allocate its full contents in Python."""
    source = tmp_path / "large.pdf"
    content = b"document bytes\n" * (1024 * 1024)
    source.write_bytes(content)

    tracemalloc.start()
    try:
        result = resolve_source_to_path(source, workdir=tmp_path / "output")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.read_bytes() == content
    assert peak < 4 * 1024 * 1024


def _open_with_simulated_failure(source, failure, error):
    """Inject an OS-style I/O error while retaining real files and close behavior."""
    original_open = Path.open

    class FailingFile:
        def __init__(self, stream, is_source, is_destination):
            self.stream = stream
            self.is_source = is_source
            self.is_destination = is_destination

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *exc):
            result = self.stream.__exit__(*exc)
            if (self.is_source and failure == "source_close") or (
                self.is_destination and failure == "destination_close"
            ):
                raise error
            return result

        def read(self, size=-1):
            if self.is_source and failure == "source_read" and (size < 0 or self.stream.tell()):
                raise error
            return self.stream.read(size)

        def write(self, data):
            if self.is_destination and failure == "destination_write":
                self.stream.write(data[:7])
                self.stream.flush()
                raise error
            return self.stream.write(data)

    def open_file(path, mode="r", *args, **kwargs):
        is_source = path == source and mode == "rb"
        is_destination = path != source and path.name == source.name and mode in ("wb", "xb")
        if is_destination and failure == "destination_open":
            raise error
        stream = original_open(path, mode, *args, **kwargs)
        return FailingFile(stream, is_source, is_destination)

    return open_file


@pytest.mark.parametrize("automatic_workdir", [False, True], ids=["explicit", "automatic"])
@pytest.mark.parametrize(
    "failure", ["source_read", "source_close", "destination_open", "destination_write", "destination_close"]
)
def test_failed_local_copy_cleans_only_owned_paths(tmp_path, monkeypatch, automatic_workdir, failure):
    """Simulated I/O failures remove new copies without deleting caller-owned files."""
    source = tmp_path / "document.pdf"
    source.write_bytes(b"original source")
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary_root))
    workdir = tmp_path / "caller-owned"
    workdir.mkdir()
    sentinel = workdir / "keep.txt"
    sentinel.write_bytes(b"keep this file")
    error = OSError(errno.EIO, f"simulated {failure} failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", _open_with_simulated_failure(source, failure, error))
        with pytest.raises(OSError) as raised:
            resolve_source_to_path(source, workdir=None if automatic_workdir else workdir)

    assert raised.value is error
    assert source.read_bytes() == b"original source"
    assert sentinel.read_bytes() == b"keep this file"
    assert not (workdir / source.name).exists()
    assert list(temporary_root.iterdir()) == []


@pytest.mark.parametrize("cleanup", ["unlink", "rmdir"])
def test_cleanup_failure_preserves_original_write_error(tmp_path, monkeypatch, caplog, cleanup):
    """A simulated cleanup denial must not hide the disk-full error being handled."""
    source = tmp_path / "document.pdf"
    source.write_bytes(b"original source")
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary_root))
    original_cleanup = getattr(Path, cleanup)
    write_error = OSError(errno.ENOSPC, "simulated disk full")
    cleanup_error = PermissionError(errno.EACCES, "simulated cleanup denied")

    def fail_cleanup(path, *args, **kwargs):
        if (cleanup == "unlink" and path != source and path.name == source.name) or (
            cleanup == "rmdir" and path.parent == temporary_root
        ):
            raise cleanup_error
        return original_cleanup(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", _open_with_simulated_failure(source, "destination_write", write_error))
        patch.setattr(Path, cleanup, fail_cleanup)
        with pytest.raises(OSError) as raised:
            resolve_source_to_path(source)

    assert raised.value is write_error
    assert raised.value.errno == errno.ENOSPC
    assert source.read_bytes() == b"original source"
    expected_warning = "incomplete local copy" if cleanup == "unlink" else "temporary directory"
    assert expected_warning in caplog.text
    assert "simulated cleanup denied" in caplog.text
