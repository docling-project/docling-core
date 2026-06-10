"""Test the pydantic models in package utils."""

import json
from io import BytesIO
from pathlib import Path

from pydantic import Field
from requests import Response

from docling_core.types.doc import DocItemLabel, DoclingDocument, TableData
from docling_core.types.doc.document import GroupLabel, RichTableCell
from docling_core.utils.alias import AliasModel
from docling_core.utils.file import resolve_source_to_path, resolve_source_to_stream

from .test_data_gen_flag import GEN_TEST_DATA


def _closeable_response(status_code=200, content=b"", headers=None) -> Response:
    """Build a Response with a real, closeable ``raw`` like a streamed response.

    ``resolve_source_to_stream`` closes the response (via ``with``) to release the
    streamed connection, which a bare ``Response()`` (``raw is None``) cannot do.
    """
    r = Response()
    r.status_code = status_code
    r._content = content
    r._content_consumed = True
    r.raw = BytesIO(content)
    if headers:
        r.headers.update(headers)
    return r


def build_single_cell_rich_table_doc(text: str) -> DoclingDocument:
    """Build a doc with a single-cell table whose cell is a RichTableCell ref to a text group."""
    doc = DoclingDocument(name="single_cell_rich_table")
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1))
    wrapper = doc.add_group(parent=table, label=GroupLabel.UNSPECIFIED)
    doc.add_text(parent=wrapper, label=DocItemLabel.TEXT, text=text)
    doc.add_table_cell(
        table_item=table,
        cell=RichTableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            ref=wrapper.get_ref(),
            text="",
        ),
    )
    return doc


def assert_or_generate_ground_truth(
    result_text: str,
    exp_path: Path,
    error_msg: str = "Serialized text should match expected output",
    is_json: bool = False,
) -> None:
    """Helper function to either generate ground truth or assert against it.

    This function supports the GEN_TEST_DATA pattern for test maintenance:
    - When GEN_TEST_DATA=1, it writes the result to the ground truth file
    - Otherwise, it reads the ground truth file and asserts equality

    Args:
        result_text: The serialized result text to compare or save
        exp_path: Path to the expected/ground truth file
        error_msg: Error message to display if assertion fails
        is_json: If True, compare as JSON objects instead of raw strings
    """
    if GEN_TEST_DATA:
        with open(exp_path, "w", encoding="utf-8") as f:
            f.write(result_text)
    else:
        with open(exp_path, encoding="utf-8") as f:
            if is_json:
                expected = json.load(f)
                actual = json.loads(result_text)
                assert actual == expected, error_msg
            else:
                expected = f.read()
                assert result_text == expected, error_msg


def assert_or_generate_json_ground_truth(
    result_data: dict | list,
    exp_path: Path | str,
    error_msg: str = "JSON data should match expected output",
) -> None:
    """Helper function to either generate JSON ground truth or assert against it.

    This is a convenience wrapper for JSON data that handles serialization with
    proper formatting (indent=4, trailing newline).

    Args:
        result_data: The data structure (dict or list) to compare or save
        exp_path: Path to the expected/ground truth JSON file
        error_msg: Error message to display if assertion fails
    """
    exp_path = Path(exp_path) if isinstance(exp_path, str) else exp_path

    if GEN_TEST_DATA:
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4)
            f.write("\n")
    else:
        with open(exp_path, encoding="utf-8") as f:
            expected = json.load(f)
        assert result_data == expected, error_msg


def test_alias_model() -> None:
    """Test the functionality of AliasModel."""

    class AliasModelChild(AliasModel):
        foo: str = Field(alias="boo")

    data = {"foo": "lorem ipsum"}
    data_alias = {"boo": "lorem ipsum"}

    # data validated from dict, JSON, and constructor can use field names or aliases

    AliasModelChild.model_validate(data_alias)
    AliasModelChild.model_validate(data)

    AliasModelChild.model_validate_json(json.dumps(data_alias))
    AliasModelChild.model_validate_json(json.dumps(data))

    AliasModelChild(boo="lorem ipsum")  # type: ignore[call-arg]
    AliasModelChild(foo="lorem ipsum")

    # children classes will also inherite the populate_by_name

    class AliasModelGrandChild(AliasModelChild):
        var: int

    AliasModelGrandChild(boo="lorem ipsum", var=3)  # type: ignore[call-arg]
    AliasModelGrandChild(foo="lorem ipsum", var=3)

    # serialized data will always use aliases

    obj = AliasModelChild.model_validate(data_alias)
    assert obj.model_dump() == data_alias
    assert obj.model_dump() != data

    assert obj.model_dump_json() == json.dumps(data_alias, separators=(",", ":"))
    assert obj.model_dump_json() != json.dumps(data, separators=(",", ":"))


def test_resolve_source_to_path_url_wout_path(monkeypatch):
    expected_str = "foo"
    expected_bytes = bytes(expected_str, "utf-8")

    def get_dummy_response(*args, **kwargs):
        r = Response()
        r.status_code = 200
        r._content = expected_bytes
        return r

    monkeypatch.setattr("requests.get", get_dummy_response)
    monkeypatch.setattr(
        "requests.models.Response.iter_content",
        lambda *args, **kwargs: [expected_bytes],
    )
    path = resolve_source_to_path("https://pypi.org")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert text == expected_str


def test_resolve_source_to_stream_url_wout_path(monkeypatch):
    expected_str = "foo"
    expected_bytes = bytes(expected_str, "utf-8")

    def get_dummy_response(*args, **kwargs):
        r = Response()
        r.status_code = 200
        r._content = expected_bytes
        return r

    monkeypatch.setattr("requests.get", get_dummy_response)
    monkeypatch.setattr(
        "requests.models.Response.iter_content",
        lambda *args, **kwargs: [expected_bytes],
    )
    doc_stream = resolve_source_to_stream("https://pypi.org")
    assert doc_stream.name == "file"

    text = doc_stream.stream.read().decode("utf8")
    assert text == expected_str


def test_sanitize_filename_paths():
    """Test filename sanitization for path-like inputs."""
    from docling_core.utils.file import _sanitize_filename

    assert _sanitize_filename("../../etc/config.txt") == "config.txt"

    assert _sanitize_filename("/etc/config.txt") == "config.txt"

    assert _sanitize_filename("..\\..\\windows\\system32\\config") == "config"
    assert _sanitize_filename("C:\\Windows\\System32\\config") == "config"

    assert _sanitize_filename("../../../etc\\config.txt") == "config.txt"

    assert _sanitize_filename("document.pdf") == "document.pdf"
    assert _sanitize_filename("my-file_123.txt") == "my-file_123.txt"

    assert _sanitize_filename("") is None
    assert _sanitize_filename(".") is None
    assert _sanitize_filename("..") is None


def test_is_safe_url_rejects_private_networks():
    """Test URL filtering for non-public network ranges."""
    from docling_core.utils.file import _is_safe_url

    assert not _is_safe_url("http://10.0.0.1/file")
    assert not _is_safe_url("http://172.16.0.1/file")
    assert not _is_safe_url("http://192.168.1.1/file")

    assert not _is_safe_url("http://127.0.0.1/file")
    assert not _is_safe_url("http://localhost/file")

    assert not _is_safe_url("http://169.254.169.254/latest/meta-data/")

    assert not _is_safe_url("http://[::1]/file")
    assert not _is_safe_url("http://[fe80::1]/file")

    assert _is_safe_url("http://8.8.8.8/file")
    assert _is_safe_url("https://example.com/file")
    assert _is_safe_url("https://github.com/github/file")


def test_resolve_remote_filename_sanitizes_content_disposition(monkeypatch):
    """Test filename normalization from Content-Disposition."""
    from requests import Response

    from docling_core.utils.file import resolve_source_to_stream

    def get_response(*args, **kwargs):
        return _closeable_response(
            content=b"test content",
            headers={
                "Content-Disposition": 'attachment; filename="../../etc/config.txt"'
            },
        )

    monkeypatch.setattr("requests.Session.get", get_response)

    doc_stream = resolve_source_to_stream("https://example.com/file")
    assert doc_stream.name == "config.txt"


def test_resolve_source_rejects_non_public_urls(monkeypatch):
    """Test that non-public URLs are rejected."""
    import pytest

    from docling_core.utils.file import resolve_source_to_stream

    with pytest.raises(ValueError, match="URL is not allowed"):
        resolve_source_to_stream("http://127.0.0.1/file")

    with pytest.raises(ValueError, match="URL is not allowed"):
        resolve_source_to_stream("http://10.0.0.1/file")

    with pytest.raises(ValueError, match="URL is not allowed"):
        resolve_source_to_stream("http://192.168.1.1/file")

    with pytest.raises(ValueError, match="URL is not allowed"):
        resolve_source_to_stream("http://169.254.169.254/latest/meta-data/")


def test_resolve_source_rejects_unsupported_scheme():
    """Test that unsupported URL schemes are rejected before file fallback."""
    import pytest

    from docling_core.utils.file import resolve_source_to_stream

    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        resolve_source_to_stream("ftp://some-server/file.pdf")


def test_resolve_source_to_path_sanitizes_filename(monkeypatch, tmp_path):
    """Test that saved filenames stay within the target directory."""
    from requests import Response

    from docling_core.utils.file import resolve_source_to_path

    def get_response(*args, **kwargs):
        return _closeable_response(
            content=b"test content",
            headers={
                "Content-Disposition": 'attachment; filename="../../../../tmp/output.txt"'
            },
        )

    monkeypatch.setattr("requests.Session.get", get_response)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    result_path = resolve_source_to_path("https://example.com/file", workdir=cache_dir)

    assert result_path.parent == cache_dir
    assert result_path.name == "output.txt"
    assert result_path.exists()

    assert not (tmp_path.parent.parent.parent / "tmp" / "output.txt").exists()


def test_redirect_limit_enforced(monkeypatch):
    """Test that redirect limits are configured on the session."""
    from requests import Response, Session

    from docling_core.utils.file import _MAX_REDIRECTS

    session_created = []

    original_init = Session.__init__

    def track_session_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        session_created.append(self)

    monkeypatch.setattr(Session, "__init__", track_session_init)

    def mock_get(*args, **kwargs):
        r = Response()
        r.status_code = 200
        r._content = b"test"
        return r

    monkeypatch.setattr(Session, "get", mock_get)

    from docling_core.utils.file import resolve_source_to_stream

    try:
        resolve_source_to_stream("https://example.com/file")
    except Exception:
        pass

    assert len(session_created) > 0
    session = session_created[0]
    assert session.max_redirects == _MAX_REDIRECTS


def test_redirect_to_non_public_ip_rejected(monkeypatch):
    """Test that redirects to non-public addresses are rejected."""
    import pytest
    from requests import Response, Session

    from docling_core.utils.file import resolve_source_to_stream

    def mock_get_with_redirect(self, *args, **kwargs):
        r = Response()
        r.status_code = 302
        r.headers["location"] = "http://192.168.1.1/private-file"
        r.url = args[0] if args else kwargs.get("url", "http://example.com")

        if hasattr(self, "hooks") and "response" in self.hooks:
            for hook in self.hooks["response"]:
                hook(r)

        return r

    monkeypatch.setattr(Session, "get", mock_get_with_redirect)

    with pytest.raises(ValueError, match="Redirect target is not allowed"):
        resolve_source_to_stream("https://example.com/redirect")


def test_resolve_source_rejects_large_content_length(monkeypatch):
    import pytest
    from requests import Response, Session

    from docling_core.utils.file import resolve_source_to_stream

    def mock_get(self, *args, **kwargs):
        return _closeable_response(
            content=b"ignored", headers={"Content-Length": "10"}
        )

    monkeypatch.setattr(Session, "get", mock_get)

    with pytest.raises(ValueError, match="maximum allowed size"):
        resolve_source_to_stream("https://example.com/file", max_file_size=5)


def test_resolve_source_stops_when_stream_exceeds_limit(monkeypatch):
    import pytest
    from requests import Response, Session

    from docling_core.utils.file import resolve_source_to_stream

    def mock_get(self, *args, **kwargs):
        r = _closeable_response(headers={"Content-Length": "4"})

        def iter_content(chunk_size=1):
            yield b"abcd"
            yield b"ef"

        r.iter_content = iter_content
        return r

    monkeypatch.setattr(Session, "get", mock_get)

    with pytest.raises(ValueError, match="maximum allowed size"):
        resolve_source_to_stream("https://example.com/file", max_file_size=5)
