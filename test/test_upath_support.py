"""Tests for UPath/fsspec support for cloud storage compatibility."""

import uuid
from pathlib import Path

import pytest
from PIL import Image as PILImage
from pydantic import AnyUrl
from upath import UPath

from docling_core.types.doc import DoclingDocument, ImageRefMode
from docling_core.types.doc.base import Size
from docling_core.types.doc.document import ImageRef
from docling_core.types.doc.utils import is_remote_path, relative_path


@pytest.fixture
def memory_base():
    """Provide a unique memory:// base path for each test."""
    return UPath(f"memory://test-{uuid.uuid4()}")


# =============================================================================
# Tests for is_remote_path()
# =============================================================================


def test_is_remote_path():
    """Test is_remote_path() function with various path types."""
    # Local paths should return False
    assert is_remote_path(Path("/local/path")) is False
    assert is_remote_path(Path(".")) is False
    assert is_remote_path(None) is False
    assert is_remote_path("/some/path") is False
    assert is_remote_path(object()) is False

    # File protocol should return False
    class MockFilePath:
        protocol = "file"

    class MockEmptyProtocol:
        protocol = ""

    assert is_remote_path(MockFilePath()) is False
    assert is_remote_path(MockEmptyProtocol()) is False

    # Cloud protocols should return True
    for protocol in ["s3", "gcs", "az", "http", "https"]:

        class MockCloudPath:
            pass

        MockCloudPath.protocol = protocol
        assert is_remote_path(MockCloudPath()) is True

    # Real UPath objects
    assert is_remote_path(UPath("/tmp/test")) is False
    assert is_remote_path(UPath("memory://test/path")) is True


# =============================================================================
# Tests for relative_path()
# =============================================================================


def test_relative_path():
    """Test relative_path() function with various scenarios."""
    # Basic relative path computation
    assert relative_path(Path("/a/b"), Path("/a/b/c/d.txt")) == Path("c/d.txt")
    assert relative_path("/a/b", "/a/b/c.txt") == Path("c.txt")

    # Navigation with parent directories
    # Sibling directory
    assert relative_path(Path("/a/b/c"), Path("/a/b/d/e.txt")) == Path("../d/e.txt")
    # Parent directory
    assert relative_path(Path("/a/b/c/d"), Path("/a/b/e.txt")) == Path("../../e.txt")

    # UPath with local file protocol
    src = UPath("/home/user/docs")
    target = UPath("/home/user/docs/images/img.png")
    assert relative_path(src, target) == Path("images/img.png")


# =============================================================================
# UPath integration tests (memory filesystem)
# =============================================================================


def test_upath_integration(sample_doc, memory_base):
    """Test UPath integration with various document operations."""
    # JSON roundtrip
    json_path = memory_base / "doc.json"
    sample_doc.save_as_json(json_path)
    assert json_path.exists()
    loaded = DoclingDocument.load_from_json(json_path)
    assert sample_doc.export_to_dict() == loaded.export_to_dict()

    # YAML roundtrip
    yaml_path = memory_base / "doc.yaml"
    sample_doc.save_as_yaml(yaml_path)
    assert yaml_path.exists()
    loaded = DoclingDocument.load_from_yaml(yaml_path)
    assert sample_doc.export_to_dict() == loaded.export_to_dict()

    # Markdown export
    md_path = memory_base / "doc.md"
    sample_doc.save_as_markdown(md_path)
    assert md_path.exists()
    assert len(md_path.read_text()) > 0

    # HTML export
    html_path = memory_base / "doc.html"
    sample_doc.save_as_html(html_path)
    assert html_path.exists()
    assert "<html" in html_path.read_text().lower()

    # Doctags export
    dt_path = memory_base / "doc.dt"
    sample_doc.save_as_doctags(dt_path)
    assert dt_path.exists()

    # Referenced mode
    json_ref_path = memory_base / "doc_ref.json"
    artifacts_dir = memory_base / "artifacts"
    sample_doc.save_as_json(json_ref_path, artifacts_dir=artifacts_dir, image_mode=ImageRefMode.REFERENCED)
    assert json_ref_path.exists()

    # Referenced images use string URI
    image_dir = memory_base / "images"
    doc_with_refs = sample_doc._with_pictures_refs(image_dir=image_dir, page_no=None)
    for pic in doc_with_refs.pictures:
        if pic.image is not None and pic.image.uri is not None:
            # For remote storage, URI should not be a UPath (can't serialize)
            assert not is_remote_path(pic.image.uri)

    # Nested path support
    nested_path = memory_base / "a" / "b" / "c" / "doc.json"
    sample_doc.save_as_json(nested_path)
    assert nested_path.exists()

    # Empty document
    empty_doc = DoclingDocument(name="Empty")
    empty_path = memory_base / "empty.json"
    empty_doc.save_as_json(empty_path)
    loaded_empty = DoclingDocument.load_from_json(empty_path)
    assert loaded_empty.name == "Empty"
    assert len(loaded_empty.texts) == 0


def test_upath_include_page_images(memory_base):
    """Test _with_pictures_refs() with include_page_images=True on a remote path.

    Ensures that page images are saved via BytesIO (UPath-compatible) and that
    page.image.uri is stored as an AnyUrl, not a Path, for remote storage.
    """
    page_img = PILImage.new("RGB", (64, 64), "blue")
    image_ref = ImageRef.from_pil(image=page_img, dpi=72)

    doc = DoclingDocument(name="PageImageDoc")
    doc.add_page(page_no=1, size=Size(width=64, height=64), image=image_ref)

    image_dir = memory_base / "page_images"

    result = doc._with_pictures_refs(
        image_dir=image_dir,
        page_no=None,
        include_page_images=True,
    )

    # The image directory must have been created and contain the page image file
    assert image_dir.exists()
    saved_files = list(image_dir.iterdir())
    assert len(saved_files) == 1, "Expected exactly one page image to be saved"
    assert saved_files[0].name.endswith(".png")

    # For a remote path the URI must be an AnyUrl, not a plain Path or UPath,
    # so that the document remains JSON-serialisable
    page = result.pages[1]
    assert page.image is not None
    assert page.image.uri is not None
    assert isinstance(page.image.uri, AnyUrl), f"Expected AnyUrl for remote page image URI, got {type(page.image.uri)}"
    assert not is_remote_path(page.image.uri)
