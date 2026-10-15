from pathlib import Path, PureWindowsPath

from docling_core.types.doc import DocItemLabel, DoclingDocument


def test_hyperlink_keeps_posix_separators():
    """Root-relative and relative URLs must not gain OS-native separators.

    Hyperlinks are URLs. With a concrete ``Path`` in the union, a value like
    ``/home.html`` was coerced to ``WindowsPath('\\home.html')`` on Windows,
    so documents serialized on Windows carried broken ``[body](\\home.html)``
    links and JSON payloads incompatible with documents created on POSIX
    platforms.
    """
    doc = DoclingDocument(name="hyperlink")
    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink="/home.html")

    hyperlink = doc.texts[0].hyperlink
    assert str(hyperlink) == "/home.html"

    payload = doc.export_to_dict()
    assert payload["texts"][0]["hyperlink"] == "/home.html"

    assert "(\\home.html)" not in doc.export_to_markdown()
    assert "(/home.html)" in doc.export_to_markdown()


def test_hyperlink_normalizes_pathlike_input():
    """Path-like inputs are normalized to POSIX separators on every platform."""
    doc = DoclingDocument(name="hyperlink")
    # PureWindowsPath behaves identically on every platform, so the input
    # carries literal backslashes even on POSIX CI runners.
    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink=PureWindowsPath("a/b.html"))
    assert str(doc.texts[0].hyperlink) == "a/b.html"

    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink=PureWindowsPath("E:/x/y.html"))
    assert str(doc.texts[1].hyperlink) == "E:/x/y.html"


def test_hyperlink_url_unchanged():
    """Absolute URLs keep taking the AnyUrl branch of the union."""
    doc = DoclingDocument(name="hyperlink")
    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink="https://example.com/a")
    assert type(doc.texts[0].hyperlink).__name__ == "AnyUrl"
    assert str(doc.texts[0].hyperlink).rstrip("/") == "https://example.com/a"


def test_hyperlink_json_roundtrip_posix(tmp_path: Path):
    doc = DoclingDocument(name="hyperlink")
    doc.add_text(label=DocItemLabel.TEXT, text="body", hyperlink="/home.html")
    path = tmp_path / "doc.json"
    doc.save_as_json(path)

    loaded = DoclingDocument.load_from_json(path)
    assert str(loaded.texts[0].hyperlink) == "/home.html"
