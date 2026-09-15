"""Tests for the ``docling-serialize`` CLI."""

import zipfile
from pathlib import Path

from typer.testing import CliRunner

from docling_core.cli.serialize import app
from docling_core.types.doc import DoclingDocument

DOC_LANG_ARCHIVE_FIXTURE = Path("tests/data/doc/doclang_archive/save/two_pages.json")


def test_serialize_json_to_dclx_packages_images(tmp_path: Path) -> None:
    """DCLX output packages page rasters and picture assets."""
    output = tmp_path / "nested" / "two_pages.dclx"

    result = CliRunner().invoke(
        app,
        [str(DOC_LANG_ARCHIVE_FIXTURE), "--to", "dclx", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert f"Wrote {output}" in result.output

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "document.xml" in names
        assert "pages/1.png" in names
        assert "pages/2.png" in names
        assert any(name.startswith("assets/") for name in names)

    loaded = DoclingDocument.load_from_doclang_archive(
        output,
        artifacts_dir=tmp_path / "loaded",
    )
    assert len(loaded.pages) == 2
    assert loaded.pictures
    assert loaded.pictures[0].image is not None
    assert loaded.pictures[0].image.pil_image is not None


def test_serialize_dclx_requires_output_path() -> None:
    """Binary DCLX output is not written to standard output."""
    result = CliRunner().invoke(
        app,
        [str(DOC_LANG_ARCHIVE_FIXTURE), "--to", "dclx"],
    )

    assert result.exit_code == 2
    assert "--output is required when --to dclx" in result.output
