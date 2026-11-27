"""Define publication metadata and chapter indexing for Markdown serialization."""

import json
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseMetadataSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.items.text import SectionHeaderItem, TitleItem

_OFFSET_WIDTH = 10
"""Width the chapter offsets are padded to.

The frontmatter is rendered twice: once to measure how far it shifts the body,
and once with the shifted offsets. Padding every offset to a fixed width keeps
the second rendering the same size as the first, so the offsets it advertises
remain correct.
"""


class PubMetadata(BaseModel):
    """Publication metadata to include in Markdown front matter."""

    title: str | None = Field(default=None, description="The title of the book.")
    authors: list[str] = Field(default_factory=list, description="The creators of the book.")
    published: str | None = Field(default=None, description="The publication date, as spelled in the source.")
    language: str | None = Field(default=None, description="The language of the book.")
    source_file: str | None = Field(default=None, description="The file name the book was converted from.")


class ChapterPosition(BaseModel):
    """Where a chapter heading starts within the serialized book."""

    title: str = Field(description="The heading text of the chapter.")
    line: int = Field(description="The 1-based line the heading is on.")
    byte: int = Field(description="The absolute UTF-8 byte offset of the heading.")


class PubMetadataSerializer(BaseModel, BaseMetadataSerializer):
    """Render publication fields and a chapter index as YAML front matter.

    Use with ``MarkdownDocSerializer(metadata_serializer=PubMetadataSerializer(...))``.
    Chapter positions address the complete output: lines are one-based and bytes
    are UTF-8 offsets. Save with ``Path.write_bytes(result.text.encode("utf-8"))``
    to preserve offsets on platforms that translate newlines in text mode.
    """

    metadata: PubMetadata | None = None

    @override
    def serialize(
        self,
        *,
        doc: DoclingDocument,
        doc_serializer: BaseDocSerializer,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize publication metadata using the finalized Markdown parts."""
        chapters = self._collect_chapters(parts)
        preliminary = self._render_frontmatter(chapters)
        byte_shift = len(preliminary.encode("utf-8"))
        line_shift = preliminary.count("\n")
        shifted = [
            chapter.model_copy(update={"byte": chapter.byte + byte_shift, "line": chapter.line + line_shift})
            for chapter in chapters
        ]
        frontmatter = self._render_frontmatter(shifted)
        if len(frontmatter.encode("utf-8")) != byte_shift:
            raise ValueError("Publication frontmatter changed size after applying the chapter offsets")

        return create_ser_result(text=frontmatter)

    def _collect_chapters(self, parts: list[SerializationResult]) -> list[ChapterPosition]:
        """Locate the chapter headings among the top-level parts of the body.

        A chapter is a part that opens on a top-level heading, i.e. a
        ``TitleItem`` (rendered as ``#``) or a level-1 ``SectionHeaderItem``
        (rendered as ``##``). Both heading styles can mark chapter boundaries in a publication.
        """
        chapters: list[ChapterPosition] = []
        byte_offset = 0
        line_number = 1
        has_content = False

        for part in parts:
            part_text = part.text
            if has_content:
                # The parts are joined by a blank line.
                byte_offset += len(b"\n\n")
                line_number += 2

            first_item = part.spans[0].item if part.spans else None
            if isinstance(first_item, TitleItem) or (
                isinstance(first_item, SectionHeaderItem) and first_item.level == 1
            ):
                chapters.append(ChapterPosition(title=first_item.text, line=line_number, byte=byte_offset))

            byte_offset += len(part_text.encode("utf-8"))
            line_number += part_text.count("\n")
            has_content = True

        return chapters

    def _render_frontmatter(self, chapters: list[ChapterPosition]) -> str:
        """Render the YAML frontmatter block, trailing blank line included."""
        metadata = self.metadata
        lines = ["---"]
        if metadata is not None:
            fields: tuple[tuple[str, str | list[str] | None], ...] = (
                ("title", metadata.title),
                ("authors", metadata.authors or None),
                ("published", metadata.published),
                ("language", metadata.language),
                ("source_file", metadata.source_file),
            )
            for key, value in fields:
                if value is not None:
                    lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")

        lines.append("chapters:" if chapters else "chapters: []")
        for chapter in chapters:
            lines.extend(
                [
                    f"  - title: {json.dumps(chapter.title, ensure_ascii=False)}",
                    f"    line: {chapter.line:>{_OFFSET_WIDTH}}",
                    f"    byte: {chapter.byte:>{_OFFSET_WIDTH}}",
                ]
            )

        lines.append("---")
        return "\n".join(lines) + "\n\n"
