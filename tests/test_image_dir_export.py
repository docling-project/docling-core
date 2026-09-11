"""Tests for the image_dir / image_uri_prefix parameters of the export methods."""

from pathlib import Path

import pytest

from docling_core.types.doc import ImageRefMode


def _saved_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("*.png"))


def test_export_to_markdown_referenced_saves_images(sample_doc, tmp_path):
    """export_to_markdown with image_dir saves images and references them."""
    image_dir = tmp_path / "images"

    md = sample_doc.export_to_markdown(
        image_mode=ImageRefMode.REFERENCED,
        image_dir=image_dir,
        image_uri_prefix="images/",
    )

    saved = _saved_images(image_dir)
    assert saved, "expected at least one image to be saved"
    # The output references the portable prefixed URI, not an absolute path.
    for img in saved:
        assert f"images/{img.name}" in md
    assert str(image_dir.resolve()) not in md


def test_export_to_html_referenced_saves_images(sample_doc, tmp_path):
    """export_to_html with image_dir saves images and references them."""
    image_dir = tmp_path / "images"

    html = sample_doc.export_to_html(
        image_mode=ImageRefMode.REFERENCED,
        image_dir=image_dir,
        image_uri_prefix="images/",
    )

    saved = _saved_images(image_dir)
    assert saved, "expected at least one image to be saved"
    for img in saved:
        assert f"images/{img.name}" in html


def test_export_to_markdown_image_dir_without_referenced_raises(sample_doc, tmp_path):
    """Passing image_dir without REFERENCED mode raises ValueError."""
    with pytest.raises(ValueError, match=r"ImageRefMode\.REFERENCED"):
        sample_doc.export_to_markdown(image_dir=tmp_path / "images")


def test_export_to_html_image_dir_without_referenced_raises(sample_doc, tmp_path):
    """Passing image_dir without REFERENCED mode raises ValueError."""
    with pytest.raises(ValueError, match=r"ImageRefMode\.REFERENCED"):
        sample_doc.export_to_html(image_dir=tmp_path / "images")
