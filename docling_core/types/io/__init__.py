# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Models for io."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator


@runtime_checkable
class BinaryStream(Protocol):
    """Readable binary stream supporting absolute and end-relative seeks.

    Consumers may move the cursor and close the stream when they unload it.
    No inheritance from an ``io`` class is required.
    """

    def read(self, size: int = -1, /) -> bytes: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def tell(self) -> int: ...

    def close(self) -> None: ...


class DocumentStream(BaseModel):
    """A filename and a readable, seekable binary stream.

    Validation preserves the stream object and cursor, and reads no content.
    Backend support for streams other than ``BytesIO`` depends on the consumer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    stream: BinaryStream

    @field_validator("stream")
    @classmethod
    def validate_binary_stream(cls, stream: BinaryStream) -> BinaryStream:
        try:
            position = stream.tell()
            try:
                content = stream.read(0)
                stream.seek(0, 2)
            finally:
                stream.seek(position)
        except (OSError, ValueError, TypeError) as error:
            raise ValueError("stream must be readable and seekable") from error
        if not isinstance(content, bytes):
            raise ValueError("stream.read() must return bytes, not text")
        return stream
