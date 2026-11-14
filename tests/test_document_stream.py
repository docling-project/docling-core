# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO, StringIO, UnsupportedOperation

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema
from typing_extensions import override

from docling_core.types.io import DocumentStream


class DuckStream:
    def __init__(self, data=b"hello"):
        self.inner = BytesIO(data)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return self.inner.read(size)

    def seek(self, offset, whence=0):
        return self.inner.seek(offset, whence)

    def tell(self):
        return self.inner.tell()

    def close(self):
        self.inner.close()


@pytest.mark.parametrize("kind", ["bytesio", "buffered", "raw", "duck"])
def test_binary_stream_identity_and_position(tmp_path, kind):
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")
    stream = {
        "bytesio": lambda: BytesIO(b"hello"),
        "buffered": lambda: path.open("rb"),
        "raw": lambda: path.open("rb", buffering=0),
        "duck": DuckStream,
    }[kind]()
    try:
        stream.seek(2)
        document = TypeAdapter(DocumentStream).validate_python({"name": "hello.txt", "stream": stream})
        assert document.stream is stream
        assert stream.tell() == 2
        assert document.model_dump(exclude={"stream"}) == {"name": "hello.txt"}
        if isinstance(stream, DuckStream):
            assert stream.read_sizes == [0]
        assert stream.read() == b"llo"
    finally:
        stream.close()


class NonSeekable(DuckStream):
    @override
    def seek(self, offset, whence=0):
        raise UnsupportedOperation("not seekable")


class NonReadable(DuckStream):
    @override
    def read(self, size=-1):
        raise UnsupportedOperation("not readable")


@pytest.mark.parametrize("stream", [StringIO("hello"), NonSeekable(), NonReadable(), object(), b"hello"])
def test_invalid_streams(stream):
    with pytest.raises(ValidationError):
        DocumentStream(name="invalid.txt", stream=stream)


def test_closed_stream():
    stream = BytesIO(b"hello")
    stream.close()
    with pytest.raises(ValidationError):
        DocumentStream(name="closed.txt", stream=stream)


def test_schema_remains_python_only():
    with pytest.raises(PydanticInvalidForJsonSchema):
        DocumentStream.model_json_schema()


def test_real_text_and_write_only_files(tmp_path):
    path = tmp_path / "stream.txt"
    path.write_text("hello", encoding="utf-8")
    with path.open("r", encoding="utf-8") as stream:
        with pytest.raises(ValidationError, match="bytes"):
            DocumentStream(name=path.name, stream=stream)
    with path.open("wb") as stream:
        with pytest.raises(ValidationError, match="readable"):
            DocumentStream(name=path.name, stream=stream)


class AbsoluteSeekOnly(DuckStream):
    @override
    def seek(self, offset, whence=0):
        if whence == 2:
            raise UnsupportedOperation("end-relative seek is not supported")
        return super().seek(offset, whence)


def test_end_relative_seek_is_required_without_consuming_content():
    stream = AbsoluteSeekOnly()
    try:
        stream.seek(2)
        with pytest.raises(ValidationError, match="seekable"):
            DocumentStream(name="absolute-only.txt", stream=stream)
        assert stream.tell() == 2
        assert stream.read_sizes == [0]
        assert not stream.inner.closed
    finally:
        stream.close()
