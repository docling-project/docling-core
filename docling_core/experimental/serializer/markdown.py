#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for Markdown serialization."""
import html
import re
import textwrap
from pathlib import Path
from typing import Optional, Union

from pydantic import AnyUrl, BaseModel, PositiveInt
from tabulate import tabulate
from typing_extensions import override

from docling_core.experimental.serializer.base import (
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.experimental.serializer.common import DocSerializer
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DoclingDocument,
    Formatting,
    FormItem,
    FormulaItem,
    ImageRef,
    InlineGroup,
    KeyValueItem,
    NodeItem,
    OrderedList,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
    UnorderedList,
)


class MarkdownTextSerializer(BaseModel, BaseTextSerializer):
    """Markdown-specific text item serializer."""

    wrap_width: Optional[PositiveInt] = None

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        escape_html = True
        escape_underscores = True
        if isinstance(item, TitleItem):
            res = f"# {item.text}"
        elif isinstance(item, SectionHeaderItem):
            res = f"{(item.level + 1) * '#'} {item.text}"
        elif isinstance(item, CodeItem):
            res = f"`{item.text}`" if is_inline_scope else f"```\n{item.text}\n```"
            escape_html = False
            escape_underscores = False
        elif isinstance(item, FormulaItem):
            if item.text:
                res = f"${item.text}$" if is_inline_scope else f"$${item.text}$$"
            elif item.orig:
                res = "<!-- formula-not-decoded -->"
            else:
                res = ""
            escape_html = False
            escape_underscores = False
        elif self.wrap_width:
            res = textwrap.fill(item.text, width=self.wrap_width)
        else:
            res = item.text
        res = doc_serializer.post_process(
            text=res,
            escape_html=escape_html,
            escape_underscores=escape_underscores,
            formatting=item.formatting,
            hyperlink=item.hyperlink,
        )
        return SerializationResult(text=res)


class MarkdownTableSerializer(BaseTableSerializer):
    """Markdown-specific table item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        text_parts: list[str] = []

        if caption_txt := doc_serializer.serialize_captions(
            item=item,
        ).text:
            text_parts.append(caption_txt)

        if item.self_ref not in doc_serializer.get_excluded_refs():
            rows = [
                [
                    # make sure that md tables are not broken
                    # due to newline chars in the text
                    col.text.replace("\n", " ")
                    for col in row
                ]
                for row in item.data.grid
            ]
            if len(rows) > 1 and len(rows[0]) > 0:
                try:
                    table_text = tabulate(rows[1:], headers=rows[0], tablefmt="github")
                except ValueError:
                    table_text = tabulate(
                        rows[1:],
                        headers=rows[0],
                        tablefmt="github",
                        disable_numparse=True,
                    )
            else:
                table_text = ""
            if table_text:
                text_parts.append(table_text)

        text_res = "\n\n".join(text_parts)

        return SerializationResult(text=text_res)


class MarkdownPictureSerializer(BasePictureSerializer):
    """Markdown-specific picture item serializer."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        image_mode: Optional[ImageRefMode] = None,
        image_placeholder: Optional[str] = None,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_image_mode = (
            image_mode if image_mode is not None else ImageRefMode.PLACEHOLDER
        )
        my_image_placeholder = (
            image_placeholder if image_placeholder is not None else "<!-- image -->"
        )

        texts: list[str] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            separator="\n",
        )
        if cap_res.text:
            texts.append(cap_res.text)

        if item.self_ref not in doc_serializer.get_excluded_refs():
            img_res = self._serialize_image_part(
                item=item,
                doc=doc,
                image_mode=my_image_mode,
                image_placeholder=my_image_placeholder,
            )
            if img_res.text:
                texts.append(img_res.text)

        text_res = "\n\n".join(texts)

        return SerializationResult(text=text_res)

    def _serialize_image_part(
        self,
        item: PictureItem,
        doc: DoclingDocument,
        image_mode: ImageRefMode,
        image_placeholder: str,
        **kwargs,
    ) -> SerializationResult:
        error_response = (
            "<!-- 🖼️❌ Image not available. "
            "Please use `PdfPipelineOptions(generate_picture_images=True)`"
            " -->"
        )
        if image_mode == ImageRefMode.PLACEHOLDER:
            text_res = image_placeholder
        elif image_mode == ImageRefMode.EMBEDDED:
            # short-cut: we already have the image in base64
            if (
                isinstance(item.image, ImageRef)
                and isinstance(item.image.uri, AnyUrl)
                and item.image.uri.scheme == "data"
            ):
                text = f"![Image]({item.image.uri})"
                text_res = text
            else:
                # get the item.image._pil or crop it out of the page-image
                img = item.get_image(doc=doc)

                if img is not None:
                    imgb64 = item._image_to_base64(img)
                    text = f"![Image](data:image/png;base64,{imgb64})"

                    text_res = text
                else:
                    text_res = error_response
        elif image_mode == ImageRefMode.REFERENCED:
            if not isinstance(item.image, ImageRef) or (
                isinstance(item.image.uri, AnyUrl) and item.image.uri.scheme == "data"
            ):
                text_res = image_placeholder
            else:
                text_res = f"![Image]({str(item.image.uri)})"
        else:
            text_res = image_placeholder

        return SerializationResult(text=text_res)


class MarkdownKeyValueSerializer(BaseKeyValueSerializer):
    """Markdown-specific key-value item serializer."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # TODO add actual implementation
        text_res = (
            "<!-- missing-key-value-item -->"
            if item.self_ref not in doc_serializer.get_excluded_refs()
            else ""
        )
        return SerializationResult(text=text_res)


class MarkdownFormSerializer(BaseFormSerializer):
    """Markdown-specific form item serializer."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # TODO add actual implementation
        text_res = (
            "<!-- missing-form-item -->"
            if item.self_ref not in doc_serializer.get_excluded_refs()
            else ""
        )
        return SerializationResult(text=text_res)


class MarkdownListSerializer(BaseModel, BaseListSerializer):
    """Markdown-specific list serializer."""

    indent: int = 4

    @override
    def serialize(
        self,
        *,
        item: Union[UnorderedList, OrderedList],
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited or set()
        parts = doc_serializer.get_parts(
            node=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
        )
        sep = "\n"
        my_parts: list[SerializationResult] = []
        for p in parts:
            if p.text and p.text[0] == " " and my_parts:
                my_parts[-1].text = sep.join([my_parts[-1].text, p.text])  # update last
            else:
                my_parts.append(p)

        indent_str = list_level * self.indent * " "
        is_ol = isinstance(item, OrderedList)
        text_res = sep.join(
            [
                # avoid additional marker on already evaled sublists
                (
                    c.text
                    if c.text and c.text[0] == " "
                    else f"{indent_str}{f'{i + 1}.' if is_ol else '-'} {c.text}"
                )
                for i, c in enumerate(my_parts)
            ]
        )
        return SerializationResult(text=text_res)


class MarkdownInlineSerializer(BaseInlineSerializer):
    """Markdown-specific inline group serializer."""

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited or set()
        parts = doc_serializer.get_parts(
            node=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
        )
        text_res = " ".join([p.text for p in parts if p.text])
        return SerializationResult(text=text_res)


class MarkdownFallbackSerializer(BaseFallbackSerializer):
    """Markdown-specific fallback serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        if isinstance(item, DocItem):
            text_res = "<!-- missing-text -->"
        else:
            text_res = ""  # TODO go with explicit None return type?
        return SerializationResult(text=text_res)


class MarkdownDocSerializer(DocSerializer):
    """Markdown-specific document serializer."""

    text_serializer: BaseTextSerializer = MarkdownTextSerializer()
    table_serializer: BaseTableSerializer = MarkdownTableSerializer()
    picture_serializer: BasePictureSerializer = MarkdownPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = MarkdownKeyValueSerializer()
    form_serializer: BaseFormSerializer = MarkdownFormSerializer()
    fallback_serializer: BaseFallbackSerializer = MarkdownFallbackSerializer()

    list_serializer: BaseListSerializer = MarkdownListSerializer()
    inline_serializer: BaseInlineSerializer = MarkdownInlineSerializer()

    @override
    def serialize_bold(self, text: str, **kwargs):
        """Apply Markdown-specific bold serialization."""
        return f"**{text}**"

    @override
    def serialize_italic(self, text: str, **kwargs):
        """Apply Markdown-specific italic serialization."""
        return f"*{text}*"

    @override
    def serialize_strikethrough(self, text: str, **kwargs):
        """Apply Markdown-specific strikethrough serialization."""
        return f"~~{text}~~"

    @override
    def serialize_hyperlink(self, text: str, hyperlink: Union[AnyUrl, Path], **kwargs):
        """Apply Markdown-specific hyperlink serialization."""
        return f"[{text}]({str(hyperlink)})"

    @classmethod
    def _escape_underscores(cls, text: str):
        """Escape underscores but leave them intact in the URL.."""
        # Firstly, identify all the URL patterns.
        url_pattern = r"!\[.*?\]\((.*?)\)"

        parts = []
        last_end = 0

        for match in re.finditer(url_pattern, text):
            # Text to add before the URL (needs to be escaped)
            before_url = text[last_end : match.start()]
            parts.append(re.sub(r"(?<!\\)_", r"\_", before_url))

            # Add the full URL part (do not escape)
            parts.append(match.group(0))
            last_end = match.end()

        # Add the final part of the text (which needs to be escaped)
        if last_end < len(text):
            parts.append(re.sub(r"(?<!\\)_", r"\_", text[last_end:]))

        return "".join(parts)
        # return text.replace("_", r"\_")

    def post_process(
        self,
        text: str,
        *,
        escape_html: bool = True,
        escape_underscores: bool = True,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        **kwargs,
    ) -> str:
        """Apply some text post-processing steps."""
        res = text
        if escape_underscores and self.escape_underscores:
            res = self._escape_underscores(text)
        if escape_html:
            res = html.escape(res, quote=False)
        res = super().post_process(
            text=res,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return res

    @override
    def serialize(self, **kwargs) -> SerializationResult:
        """Run the serialization."""
        parts = self.get_parts()

        if not self.add_page_markers:
            text_res = "\n\n".join([p.text for p in parts if p.text])
            return SerializationResult(text=text_res)

        # Get all pages in the document
        all_pages: set[int] = set()
        page_items: dict[int, list[DocItem]] = {}

        # Collect all items with page information
        for ix, (item, _) in enumerate(
            self.doc.iterate_items(
                with_groups=True,
                traverse_pictures=True,
            )
        ):
            if ix < self.start or ix >= self.stop:
                continue

            if isinstance(item, DocItem) and item.prov and len(item.prov) > 0:
                page_no = item.prov[0].page_no

                # Skip pages that are not in the requested pages set
                if self.pages is not None and page_no not in self.pages:
                    continue

                all_pages.add(page_no)

                if page_no not in page_items:
                    page_items[page_no] = []

                page_items[page_no].append(item)

        # If no page information is available, return without page markers
        if not all_pages:
            text_res = "\n\n".join([p.text for p in parts if p.text])
            return SerializationResult(text=text_res)

        # Build the result with page markers
        result = []

        # Sort pages
        sorted_pages = sorted(all_pages)

        # Process each page
        for page_no in sorted_pages:
            # Add page marker
            result.append(f"##PAGE {page_no}##")

            # Filter parts for this page
            page_parts = []

            # Get all items on this page
            items_on_page = page_items.get(page_no, [])

            # Process all parts
            for part in parts:
                if not part.text:
                    continue

                # Check if this part belongs to the current page
                # This is a simplified approach - in a real implementation,
                # you would need a more robust way to associate parts with pages
                for item in items_on_page:
                    # Simple heuristic: if the item's text is in the part's text
                    if hasattr(item, "text") and item.text in part.text:
                        page_parts.append(part.text)
                        break

            # Add all parts for this page
            if page_parts:
                result.extend(page_parts)

        # If we couldn't associate parts with pages properly,
        # fall back to the original approach
        if not any(result[1:]):
            # Check if we have any content after the first page marker
            # Fall back to original approach without page markers
            text_res = "\n\n".join([p.text for p in parts if p.text])
            return SerializationResult(text=text_res)

        text_res = "\n\n".join(result)
        return SerializationResult(text=text_res)
