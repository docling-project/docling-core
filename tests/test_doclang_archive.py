"""Tests for DocLang archive (``.dclx``) save and load."""

import json
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from docling_core.types.doc import DoclingDocument, ImageRef, ImageRefMode
from tests.test_data_gen_flag import GEN_TEST_DATA

DOC_LANG_ARCHIVE_DIR = Path("tests/data/doc/doclang_archive")
SAVE_FIXTURE_JSON = DOC_LANG_ARCHIVE_DIR / "save" / "two_pages.json"
LOAD_FIXTURE_DCLX = DOC_LANG_ARCHIVE_DIR / "load" / "two_pages.dclx"
LOAD_FIXTURE_GT_JSON = DOC_LANG_ARCHIVE_DIR / "load" / "two_pages.gt.json"


@pytest.fixture
def save_fixture_doc(tmp_path: Path) -> DoclingDocument:
    """Load a private copy of the save-side input document."""
    doc_json = tmp_path / "two_pages.json"
    shutil.copy2(SAVE_FIXTURE_JSON, doc_json)
    return DoclingDocument.load_from_json(doc_json)


def _strip_image_uris(data: Any) -> Any:
    """Drop image URIs for golden comparisons (embedded base64 is kept separately)."""
    if isinstance(data, dict):
        return {key: _strip_image_uris(value) for key, value in data.items() if key != "uri"}
    if isinstance(data, list):
        return [_strip_image_uris(item) for item in data]
    return data


def _load_json_without_image_uris(path: Path) -> Any:
    with open(path, encoding="utf-8") as fr:
        return _strip_image_uris(json.load(fr))


def _with_embedded_page_images(doc: DoclingDocument) -> DoclingDocument:
    """Embed page raster images so ``save_as_json`` golden output is path-independent."""
    result = deepcopy(doc)
    for page in result.pages.values():
        if page.image is not None and page.image.pil_image is not None:
            page.image = ImageRef.from_pil(page.image.pil_image, dpi=page.image.dpi)
    return result


def _save_loaded_doc_as_json_for_golden(doc: DoclingDocument, path: Path) -> None:
    _with_embedded_page_images(doc).save_as_json(path)


def _verify_loaded_doclang_archive(*, actual: DoclingDocument, exp_file: Path, tmp_path: Path) -> None:
    actual_json = tmp_path / "loaded.json"

    if GEN_TEST_DATA:
        _save_loaded_doc_as_json_for_golden(actual, exp_file)
        return

    _save_loaded_doc_as_json_for_golden(actual, actual_json)

    actual_data = _load_json_without_image_uris(actual_json)
    expected_data = _load_json_without_image_uris(exp_file)
    assert actual_data == expected_data, f"Loaded DocLang archive document differs from {exp_file}"

    with open(exp_file, encoding="utf-8") as fr:
        expected_raw = fr.read()
    with open(actual_json, encoding="utf-8") as fr:
        actual_raw = fr.read()
    if "data:image/png;base64" in expected_raw:
        assert "data:image/png;base64" in actual_raw


def test_save_as_doclang_archive(save_fixture_doc: DoclingDocument, tmp_path: Path) -> None:
    dclx = tmp_path / "two_pages.dclx"
    save_fixture_doc.save_as_doclang_archive(dclx)

    assert dclx.is_file()
    assert len(save_fixture_doc.pages) == 2

    with zipfile.ZipFile(dclx) as archive:
        names = archive.namelist()
        assert "document.xml" in names
        assert "pages/1.png" in names
        assert "pages/2.png" in names
        assert any(name.startswith("assets/") for name in names)

        xml = archive.read("document.xml").decode("utf-8")
        assert "base64" not in xml
        assert 'uri="assets/' in xml


def test_save_as_doclang_archive_with_namespace(save_fixture_doc: DoclingDocument, tmp_path: Path) -> None:
    from doclang import SchematronBackendNotFound, ValidationError

    from docling_core.transforms.serializer._doclang_utils import DOCLANG_NAMESPACE

    def document_xml(dclx: Path) -> str:
        with zipfile.ZipFile(dclx) as archive:
            return archive.read("document.xml").decode("utf-8")

    # by default, no namespace is declared (unchanged output)
    plain = tmp_path / "plain.dclx"
    save_fixture_doc.save_as_doclang_archive(plain)
    assert f'xmlns="{DOCLANG_NAMESPACE}"' not in document_xml(plain)

    namespaced = tmp_path / "namespaced.dclx"
    save_fixture_doc.save_as_doclang_archive(namespaced, include_namespace=True)
    assert f'xmlns="{DOCLANG_NAMESPACE}"' in document_xml(namespaced)

    # the namespaced archive loads like the plain one
    loaded = DoclingDocument.load_from_doclang_archive(namespaced, artifacts_dir=tmp_path / "namespaced_artifacts")
    loaded_plain = DoclingDocument.load_from_doclang_archive(plain, artifacts_dir=tmp_path / "plain_artifacts")
    assert loaded.export_to_markdown() == loaded_plain.export_to_markdown()

    # validation needs the namespace (and a Schematron backend)
    try:
        save_fixture_doc.save_as_doclang_archive(tmp_path / "validated.dclx", include_namespace=True, validate=True)
    except SchematronBackendNotFound:
        pytest.skip("no Schematron backend installed for doclang")
    with pytest.raises(ValidationError):
        save_fixture_doc.save_as_doclang_archive(tmp_path / "invalid.dclx", validate=True)


def test_save_as_doclang_archive_image_mode(save_fixture_doc: DoclingDocument, tmp_path: Path) -> None:
    # placeholder: no picture images stored or referenced, page images still stored
    dclx = tmp_path / "placeholder.dclx"
    save_fixture_doc.save_as_doclang_archive(dclx, image_mode=ImageRefMode.PLACEHOLDER)
    with zipfile.ZipFile(dclx) as archive:
        names = archive.namelist()
        assert "pages/1.png" in names
        assert "pages/2.png" in names
        assert not any(name.startswith("assets/") for name in names)
        assert "<src" not in archive.read("document.xml").decode("utf-8")

    with pytest.raises(ValueError):
        save_fixture_doc.save_as_doclang_archive(tmp_path / "embedded.dclx", image_mode=ImageRefMode.EMBEDDED)


def test_export_to_doclang_image_mode(save_fixture_doc: DoclingDocument, tmp_path: Path) -> None:
    assert "<src" not in save_fixture_doc.export_to_doclang()
    assert 'uri="data:image/png;base64' in save_fixture_doc.export_to_doclang(image_mode=ImageRefMode.EMBEDDED)

    saved = tmp_path / "embedded.dclg.xml"
    save_fixture_doc.save_as_doclang(saved, image_mode=ImageRefMode.EMBEDDED)
    assert 'uri="data:image/png;base64' in saved.read_text(encoding="utf-8")


def test_doclang_resolution(save_fixture_doc: DoclingDocument, tmp_path: Path) -> None:
    default = save_fixture_doc.export_to_doclang()
    custom = save_fixture_doc.export_to_doclang(xsize=2048, ysize=1024)
    assert '<default_resolution width="2048" height="1024"/>' in custom
    assert '<default_resolution width="2048" height="1024"/>' not in default
    assert default != custom

    out = tmp_path / "doc.dclg.xml"
    save_fixture_doc.save_as_doclang(out, xsize=2048, ysize=1024)
    assert '<default_resolution width="2048" height="1024"/>' in out.read_text(encoding="utf-8")

    archive = tmp_path / "doc.dclx"
    save_fixture_doc.save_as_doclang_archive(archive, xsize=2048, ysize=1024)
    with zipfile.ZipFile(archive) as zf:
        assert '<default_resolution width="2048" height="1024"/>' in zf.read("document.xml").decode("utf-8")


def test_load_from_doclang_archive(tmp_path: Path) -> None:
    loaded = DoclingDocument.load_from_doclang_archive(
        LOAD_FIXTURE_DCLX,
        artifacts_dir=tmp_path / "two_pages_artifacts",
    )

    _verify_loaded_doclang_archive(actual=loaded, exp_file=LOAD_FIXTURE_GT_JSON, tmp_path=tmp_path)

    assert len(loaded.pages) == 2
    assert loaded.pictures[0].image is not None
    assert loaded.pictures[0].image.pil_image is not None
    for page_no in (1, 2):
        page_image = loaded.pages[page_no].image
        assert page_image is not None
        assert page_image.pil_image is not None


def test_doclang_archive_roundtrip(save_fixture_doc: DoclingDocument, tmp_path: Path) -> None:
    dclx = tmp_path / "two_pages.dclx"
    save_fixture_doc.save_as_doclang_archive(dclx)

    loaded = DoclingDocument.load_from_doclang_archive(
        dclx,
        artifacts_dir=tmp_path / "two_pages_artifacts",
    )

    assert len(loaded.pages) == 2
    assert len(loaded.pictures) == len(save_fixture_doc.pictures)

    orig_picture = save_fixture_doc.pictures[0].get_image(doc=save_fixture_doc)
    loaded_picture = loaded.pictures[0].get_image(doc=loaded)
    assert orig_picture is not None
    assert loaded_picture is not None
    assert orig_picture.size == loaded_picture.size

    roundtrip_dclx = tmp_path / "two_pages_roundtrip.dclx"
    loaded.save_as_doclang_archive(roundtrip_dclx)

    with zipfile.ZipFile(roundtrip_dclx) as archive:
        assert "pages/1.png" in archive.namelist()
        assert "pages/2.png" in archive.namelist()

    reloaded = DoclingDocument.load_from_doclang_archive(
        roundtrip_dclx,
        artifacts_dir=tmp_path / "two_pages_roundtrip_artifacts",
    )
    assert len(reloaded.pages) == 2
    assert len(reloaded.pictures) == len(loaded.pictures)
    assert reloaded.pictures[0].image is not None
    assert reloaded.pictures[0].image.pil_image is not None


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_safe_extract_zip_archive_rejects_oversize_member(tmp_path: Path) -> None:
    from docling_core.types.doc.utils import safe_extract_zip_archive

    archive_path = tmp_path / "oversize_member.dclx"
    _write_zip(archive_path, {"document.xml": b"A" * 4096})

    with pytest.raises(ValueError, match="Archive member exceeds size limit"):
        safe_extract_zip_archive(
            archive_path,
            tmp_path / "out",
            max_member_size=128,
            max_total_size=10 * 1024 * 1024,
        )


def test_safe_extract_zip_archive_rejects_oversize_total(tmp_path: Path) -> None:
    from docling_core.types.doc.utils import safe_extract_zip_archive

    archive_path = tmp_path / "oversize_total.dclx"
    _write_zip(
        archive_path,
        {
            "a.txt": b"A" * 100,
            "b.txt": b"B" * 100,
        },
    )

    with pytest.raises(ValueError, match="total uncompressed size limit"):
        safe_extract_zip_archive(
            archive_path,
            tmp_path / "out",
            max_member_size=10 * 1024 * 1024,
            max_total_size=150,
        )


def test_copy_zip_member_bounded_enforces_actual_bytes() -> None:
    """Budgets must apply to bytes read, not only declared ZipInfo.file_size."""
    from io import BytesIO

    from docling_core.types.doc.utils import _copy_zip_member_bounded

    src = BytesIO(b"C" * 4096)
    dst = BytesIO()
    with pytest.raises(ValueError, match="Archive member exceeds size limit"):
        _copy_zip_member_bounded(
            src,
            dst,
            member="document.xml",
            max_member_size=128,
            max_total_size=10 * 1024 * 1024,
            remaining_total=10 * 1024 * 1024,
        )
    assert len(dst.getvalue()) == 128


def test_copy_zip_member_bounded_enforces_remaining_total() -> None:
    from io import BytesIO

    from docling_core.types.doc.utils import _copy_zip_member_bounded

    src = BytesIO(b"C" * 4096)
    dst = BytesIO()
    with pytest.raises(ValueError, match="total uncompressed size limit"):
        _copy_zip_member_bounded(
            src,
            dst,
            member="document.xml",
            max_member_size=10 * 1024 * 1024,
            max_total_size=200,
            remaining_total=100,
        )
    assert len(dst.getvalue()) == 100


def test_load_from_doclang_archive_forwards_size_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Archive member exceeds size limit"):
        DoclingDocument.load_from_doclang_archive(
            LOAD_FIXTURE_DCLX,
            artifacts_dir=tmp_path / "artifacts",
            max_member_size=128,
            max_total_size=10 * 1024 * 1024,
        )


def test_load_from_doclang_archive_rejects_oversize_document_xml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``document.xml`` is gated by the DocLang XML budget before parsing."""
    from docling_core.utils.settings import settings

    archive_path = tmp_path / "big_xml.dclx"
    # Keep under default zip member caps; still over a tightened XML parse budget.
    xml = b"<doclang>" + (b"x" * 4096) + b"</doclang>"
    _write_zip(archive_path, {"document.xml": xml})

    monkeypatch.setattr(settings, "max_doclang_xml_bytes", 1024)
    with pytest.raises(ValueError, match="exceeds size limit"):
        DoclingDocument.load_from_doclang_archive(
            archive_path,
            artifacts_dir=tmp_path / "artifacts",
        )


def test_load_from_doclang_archive_rejects_oversize_page_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Extracted ``pages/`` rasters must respect ``max_image_decoded_size`` before PIL open."""
    from docling_core.utils.settings import settings

    archive_path = tmp_path / "big_page.dclx"
    xml = b"""<doclang>
  <page_break/>
</doclang>"""
    # Minimal valid-looking PNG header plus padding past the test budget.
    oversize = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 256)
    _write_zip(
        archive_path,
        {
            "document.xml": xml,
            "pages/1.png": oversize,
        },
    )

    monkeypatch.setattr(settings, "max_image_decoded_size", 64)
    with pytest.raises(ValueError, match="exceeds size limit"):
        DoclingDocument.load_from_doclang_archive(
            archive_path,
            artifacts_dir=tmp_path / "artifacts",
        )


def test_load_from_doclang_archive_rejects_oversize_asset_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extracted ``assets/`` picture sources must respect ``max_image_decoded_size``."""
    from docling_core.utils.settings import settings

    archive_path = tmp_path / "big_asset.dclx"
    xml = b"""<doclang>
  <picture>
    <src uri="assets/big.png"/>
  </picture>
</doclang>"""
    oversize = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 256)
    _write_zip(
        archive_path,
        {
            "document.xml": xml,
            "assets/big.png": oversize,
        },
    )

    monkeypatch.setattr(settings, "max_image_decoded_size", 64)
    with pytest.raises(ValueError, match="exceeds size limit"):
        DoclingDocument.load_from_doclang_archive(
            archive_path,
            artifacts_dir=tmp_path / "artifacts",
        )
