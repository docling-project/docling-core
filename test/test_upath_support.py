"""Tests for UPath/fsspec support for cloud storage compatibility."""

import uuid
from pathlib import Path

import pytest

from docling_core.types.doc import DoclingDocument, ImageRefMode
from docling_core.types.doc.utils import is_remote_path, relative_path

# Check if universal-pathlib is available
try:
    from upath import UPath

    HAS_UPATH = True
except ImportError:
    HAS_UPATH = False


@pytest.fixture
def memory_base():
    """Provide a unique memory:// base path for each test."""
    return UPath(f"memory://test-{uuid.uuid4()}")


# =============================================================================
# Tests for is_remote_path()
# =============================================================================


def test_is_remote_path_local_paths_return_false():
    assert is_remote_path(Path("/local/path")) is False
    assert is_remote_path(Path(".")) is False
    assert is_remote_path(None) is False
    assert is_remote_path("/some/path") is False
    assert is_remote_path(object()) is False


def test_is_remote_path_file_protocol_returns_false():
    class MockFilePath:
        protocol = "file"

    class MockEmptyProtocol:
        protocol = ""

    assert is_remote_path(MockFilePath()) is False
    assert is_remote_path(MockEmptyProtocol()) is False


@pytest.mark.parametrize("protocol", ["s3", "gcs", "az", "http", "https"])
def test_is_remote_path_cloud_protocols_return_true(protocol):
    class MockCloudPath:
        pass

    MockCloudPath.protocol = protocol
    assert is_remote_path(MockCloudPath()) is True


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_is_remote_path_real_upath():
    assert is_remote_path(UPath("/tmp/test")) is False
    assert is_remote_path(UPath("memory://test/path")) is True


# =============================================================================
# Tests for relative_path()
# =============================================================================


def test_relative_path_basic():
    assert relative_path(Path("/a/b"), Path("/a/b/c/d.txt")) == Path("c/d.txt")
    assert relative_path("/a/b", "/a/b/c.txt") == Path("c.txt")


def test_relative_path_navigation():
    # Sibling directory
    assert relative_path(Path("/a/b/c"), Path("/a/b/d/e.txt")) == Path("../d/e.txt")
    # Parent directory
    assert relative_path(Path("/a/b/c/d"), Path("/a/b/e.txt")) == Path("../../e.txt")


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_relative_path_with_upath():
    src = UPath("/home/user/docs")
    target = UPath("/home/user/docs/images/img.png")
    assert relative_path(src, target) == Path("images/img.png")


# =============================================================================
# UPath integration tests (memory filesystem)
# =============================================================================


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_json_roundtrip(sample_doc, memory_base):
    path = memory_base / "doc.json"
    sample_doc.save_as_json(path)
    assert path.exists()

    loaded = DoclingDocument.load_from_json(path)
    assert sample_doc.export_to_dict() == loaded.export_to_dict()


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_yaml_roundtrip(sample_doc, memory_base):
    path = memory_base / "doc.yaml"
    sample_doc.save_as_yaml(path)
    assert path.exists()

    loaded = DoclingDocument.load_from_yaml(path)
    assert sample_doc.export_to_dict() == loaded.export_to_dict()


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_markdown(sample_doc, memory_base):
    path = memory_base / "doc.md"
    sample_doc.save_as_markdown(path)
    assert path.exists()
    assert len(path.read_text()) > 0


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_html(sample_doc, memory_base):
    path = memory_base / "doc.html"
    sample_doc.save_as_html(path)
    assert path.exists()
    assert "<html" in path.read_text().lower()


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_doctags(sample_doc, memory_base):
    path = memory_base / "doc.dt"
    sample_doc.save_as_doctags(path)
    assert path.exists()


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_referenced_mode(sample_doc, memory_base):
    json_path = memory_base / "doc.json"
    artifacts_dir = memory_base / "artifacts"

    sample_doc.save_as_json(
        json_path, artifacts_dir=artifacts_dir, image_mode=ImageRefMode.REFERENCED
    )
    assert json_path.exists()


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_referenced_images_use_string_uri(sample_doc, memory_base):
    image_dir = memory_base / "images"
    doc_with_refs = sample_doc._with_pictures_refs(image_dir=image_dir, page_no=None)

    for pic in doc_with_refs.pictures:
        if pic.image is not None and pic.image.uri is not None:
            # For remote storage, URI should not be a UPath (can't serialize)
            assert not is_remote_path(pic.image.uri)


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_nested_path(sample_doc, memory_base):
    path = memory_base / "a" / "b" / "c" / "doc.json"
    sample_doc.save_as_json(path)
    assert path.exists()


@pytest.mark.skipif(not HAS_UPATH, reason="universal-pathlib not installed")
def test_upath_empty_document(memory_base):
    doc = DoclingDocument(name="Empty")
    path = memory_base / "empty.json"

    doc.save_as_json(path)
    loaded = DoclingDocument.load_from_json(path)

    assert loaded.name == "Empty"
    assert len(loaded.texts) == 0
