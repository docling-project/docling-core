import json
from pathlib import PurePosixPath

from PIL import Image as PILImage

from docling_core.types.doc import DoclingDocument, ImageRef, ImageRefMode
from docling_core.types.doc.base import Size


def _make_ref(uri: str) -> ImageRef:
    return ImageRef(mimetype="image/png", dpi=72, size=Size(width=10, height=10), uri=uri)


def test_image_ref_uri_keeps_posix_separators():
    """Relative and root-relative URIs must not gain OS-native separators.

    With a concrete ``Path`` in the union, ``'images/fig1.png'`` was coerced
    to ``WindowsPath('images\\fig1.png')`` on Windows, so the JSON payload
    carried ``images\\\\fig1.png`` and documents were not interchangeable
    across platforms.
    """
    ref = _make_ref("images/fig1.png")
    assert str(ref.uri) == "images/fig1.png"
    assert json.loads(ref.model_dump_json())["uri"] == "images/fig1.png"


def test_image_ref_uri_root_relative_keeps_leading_slash():
    ref = _make_ref("/abs/p.png")
    assert str(ref.uri) == "/abs/p.png"


def test_image_ref_uri_url_branch_unchanged():
    ref = _make_ref("https://cdn.example.com/a.png")
    assert type(ref.uri).__name__ == "AnyUrl"
    assert str(ref.uri) == "https://cdn.example.com/a.png"


def test_image_ref_uri_json_roundtrip():
    ref = _make_ref("images/fig1.png")
    loaded = ImageRef.model_validate_json(ref.model_dump_json())
    assert str(loaded.uri) == "images/fig1.png"


def test_pil_image_loads_from_pure_posix_uri(tmp_path, monkeypatch):
    """Picture loading must keep working through the PurePosixPath URI."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    PILImage.new("RGB", (4, 4), color="red").save(img_dir / "fig1.png")

    monkeypatch.chdir(tmp_path)
    ref = _make_ref("images/fig1.png")
    assert isinstance(ref.uri, PurePosixPath)
    pil = ref.pil_image
    assert pil is not None
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
