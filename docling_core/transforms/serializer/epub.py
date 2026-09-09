"""Define classes for EPUB book Markdown serialization."""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import override

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc.items.text import SectionHeaderItem, TitleItem

_OFFSET_WIDTH = 10
"""Width the chapter offsets are padded to.

The frontmatter is rendered twice: once to measure how far it shifts the body,
and once with the shifted offsets. Padding every offset to a fixed width keeps
the second rendering the same size as the first, so the offsets it advertises
remain correct.
"""


class EpubMetadata(BaseModel):
    """Book metadata, typically read from an EPUB package document."""

    title: Optional[str] = Field(default=None, description="The title of the book.")
    authors: list[str] = Field(default_factory=list, description="The creators of the book.")
    published: Optional[str] = Field(default=None, description="The publication date, as spelled in the source.")
    language: Optional[str] = Field(default=None, description="The language of the book.")
    source_file: Optional[str] = Field(default=None, description="The file name the book was converted from.")


class EpubParams(MarkdownParams):
    """EPUB-specific serialization parameters."""

    metadata: Optional[EpubMetadata] = Field(
        default=None,
        description="Book metadata to render as frontmatter. Unset fields are left out.",
        exclude=True,
    )


class ChapterPosition(BaseModel):
    """Where a chapter heading starts within the serialized book."""

    title: str = Field(description="The heading text of the chapter.")
    line: int = Field(description="The 1-based line the heading is on.")
    byte: int = Field(description="The absolute UTF-8 byte offset of the heading.")


class EpubDocSerializer(MarkdownDocSerializer):
    """EPUB-specific document serializer."""

    params: EpubParams = EpubParams()

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a document out of its parts, prefixed by the book frontmatter."""
        body_result = super().serialize_doc(parts=parts, **kwargs)

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
            raise ValueError("EPUB frontmatter changed size after applying the chapter offsets")

        return create_ser_result(text=f"{frontmatter}{body_result.text}", span_source=parts)

    def _collect_chapters(self, parts: list[SerializationResult]) -> list[ChapterPosition]:
        """Locate the chapter headings among the top-level parts of the body.

        A chapter is a part that opens on a top-level heading, i.e. a
        ``TitleItem`` (rendered as ``#``) or a level-1 ``SectionHeaderItem``
        (rendered as ``##``). Both are matched because an EPUB spine document
        may title its chapters with either ``<h1>`` or ``<h2>``.
        """
        chapters: list[ChapterPosition] = []
        byte_offset = 0
        line_number = 1
        has_content = False

        for part in parts:
            if not part.text:
                continue
            part_text = self._finalize_part_text(part.text)
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

    def _finalize_part_text(self, text: str) -> str:
        """Apply the body-level substitutions before measuring a serialized part."""
        if not self.requires_page_break():
            return text

        page_separator = self.params.page_break_placeholder or ""
        for full_match, _, _ in self._get_page_breaks(text=text):
            text = text.replace(full_match, page_separator)
        return text

    def _render_frontmatter(self, chapters: list[ChapterPosition]) -> str:
        """Render the YAML frontmatter block, trailing blank line included."""
        metadata = self.params.metadata
        lines = ["---"]
        if metadata is not None:
            fields: tuple[tuple[str, Optional[str] | list[str]], ...] = (
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
