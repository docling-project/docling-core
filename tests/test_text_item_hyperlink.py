import json
from pathlib import Path

from pydantic import AnyUrl

from docling_core.types.doc import DocItemLabel, DoclingDocument


def test_hyperlink_path_payload_stays_posix():
    """Path-like hyperlinks serialize with POSIX separators.

    Hyperlinks are URIs: the in-memory value keeps the declared native
    ``Path`` type, and at serialization time ``as_posix()`` renders it, so
    payloads never carry OS-native separators and documents stay
    interchangeable across platforms.
    """
    doc = DoclingDocument(name="hyperlink")
    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink=Path("/home.html"))

    hyperlink = doc.texts[0].hyperlink
    assert hyperlink is not None and not isinstance(hyperlink, str)

    payload = doc.export_to_dict()
    assert payload["texts"][0]["hyperlink"] == "/home.html"

    assert "(\\home.html)" not in doc.export_to_markdown()
    assert "(/home.html)" in doc.export_to_markdown()


def test_hyperlink_pathlike_input_payload_posix():
    """Native Path inputs serialize with POSIX separators on every platform."""
    doc = DoclingDocument(name="hyperlink")
    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink=Path("a") / "b.html")
    # The drive-letter form takes the same route on both platforms:
    # WindowsPath('E:/x/y.html') and PosixPath('E:/x/y.html') both render
    # 'E:/x/y.html' through as_posix().
    doc.add_text(label=DocItemLabel.TEXT, text="more", hyperlink=Path("E:/x/y.html"))

    payload = doc.export_to_dict()
    assert payload["texts"][0]["hyperlink"] == "a/b.html"
    assert payload["texts"][1]["hyperlink"] == "E:/x/y.html"


def test_hyperlink_url_unchanged():
    """Absolute URLs keep taking the AnyUrl branch of the union."""
    doc = DoclingDocument(name="hyperlink")
    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink=AnyUrl("https://example.com/a"))
    assert type(doc.texts[0].hyperlink).__name__ == "AnyUrl"
    assert str(doc.texts[0].hyperlink).rstrip("/") == "https://example.com/a"


def test_hyperlink_json_roundtrip_posix(tmp_path):
    doc = DoclingDocument(name="hyperlink")
    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink=Path("/home.html"))
    path = tmp_path / "doc.json"
    doc.save_as_json(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["texts"][0]["hyperlink"] == "/home.html"

    loaded = DoclingDocument.load_from_json(path)
    assert loaded.export_to_dict()["texts"][0]["hyperlink"] == "/home.html"
