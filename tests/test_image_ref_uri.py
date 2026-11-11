import json
from pathlib import Path

import pytest
from PIL import Image as PILImage

from docling_core.types.doc import DoclingDocument, ImageRef, ImageRefMode
from docling_core.types.doc.base import Size
from docling_core.utils.settings import settings


def _make_ref(uri: str) -> ImageRef:
    return ImageRef(mimetype="image/png", dpi=72, size=Size(width=10, height=10), uri=uri)


def test_image_ref_uri_payload_stays_posix():
    """Path-like URIs serialize with POSIX separators.

    Without a serializer, an input like ``'images/fig1.png'`` validated into
    ``WindowsPath('images\\fig1.png')`` on Windows and the JSON payload carried
    ``images\\fig1.png``, making documents non-interchangeable across
    platforms. The in-memory type stays the declared ``AnyUrl | Path``; the
    field serializer renders ``as_posix()`` at serialization time only.
    """
    ref = _make_ref("images/fig1.png")
    assert isinstance(ref.uri, Path)

    assert json.loads(ref.model_dump_json())["uri"] == "images/fig1.png"


def test_image_ref_uri_root_relative_payload_posix():
    ref = _make_ref("/abs/p.png")
    assert json.loads(ref.model_dump_json())["uri"] == "/abs/p.png"


def test_image_ref_uri_url_unchanged():
    ref = _make_ref("https://cdn.example.com/a.png")
    assert type(ref.uri).__name__ == "AnyUrl"
    assert json.loads(ref.model_dump_json())["uri"] == "https://cdn.example.com/a.png"


def test_image_ref_uri_json_roundtrip_posix():
    ref = _make_ref("images/fig1.png")
    loaded = ImageRef.model_validate_json(ref.model_dump_json())
    assert isinstance(loaded.uri, Path)
    assert json.loads(loaded.model_dump_json())["uri"] == "images/fig1.png"


def test_pil_image_loads_from_path_uri(tmp_path, monkeypatch):
    """Picture loading keeps working through a native Path URI."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    PILImage.new("RGB", (4, 4), color="red").save(img_dir / "fig1.png")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "allow_image_file_uri", True)
    ref = _make_ref("images/fig1.png")
    assert isinstance(ref.uri, Path)
    pil = ref.pil_image
    assert pil is not None
    assert pil.size == (4, 4)


def test_pil_image_path_uri_honors_file_uri_gate(tmp_path, monkeypatch):
    """A path URI is still a local read, so the opt-in gate applies."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    PILImage.new("RGB", (4, 4), color="red").save(img_dir / "fig1.png")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "allow_image_file_uri", False)
    ref = _make_ref("images/fig1.png")
    with pytest.raises(ValueError, match="not enabled"):
        ref.pil_image


def test_load_local_images_resolves_relative_uri(tmp_path, monkeypatch):
    """load_local_images resolves a relative path URI against base_dir."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    PILImage.new("RGB", (4, 4), color="red").save(img_dir / "fig1.png")

    monkeypatch.setattr(settings, "allow_image_file_uri", False)
    doc = DoclingDocument(name="images")
    doc.add_picture(image=_make_ref("images/fig1.png"))
    doc.load_local_images(base_dir=tmp_path)

    image = doc.pictures[0].image
    assert image is not None
    assert str(image.uri) == str(Path("images/fig1.png"))
    assert (pil := image.pil_image) is not None
    assert pil.size == (4, 4)


def test_save_as_json_emits_posix_image_uris(tmp_path):
    """save_as_json with REFERENCED images must write portable (POSIX) URIs
    even on Windows, where the reference computation produces OS paths."""
    doc = DoclingDocument(name="images")
    doc.add_picture(image=ImageRef.from_pil(PILImage.new("RGB", (8, 8), color="blue"), dpi=72))

    json_path = tmp_path / "doc.json"
    doc.save_as_json(
        json_path,
        artifacts_dir=tmp_path / "images",
        image_mode=ImageRefMode.REFERENCED,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    uri = payload["pictures"][0]["image"]["uri"]
    assert "\\" not in uri, f"backslash leaked into image URI: {uri!r}"
