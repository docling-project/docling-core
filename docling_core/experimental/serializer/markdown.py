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
from docling_core.experimental.serializer.common import CommonParams, DocSerializer
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CodeItem,
    ContentLayer,
    DocItem,
    DoclingDocument,
    FloatingItem,
    Formatting,
    FormItem,
    FormulaItem,
    ImageRef,
    InlineGroup,
    KeyValueItem,
    NodeItem,
    OrderedList,
    PictureItem,
    PictureTabularChartData,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
    UnorderedList,
)


class MarkdownParams(CommonParams):
    """Markdown-specific serialization parameters."""

    layers: set[ContentLayer] = {ContentLayer.BODY}
    image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
    image_placeholder: str = "<!-- image -->"
    enable_chart_tables: bool = True
    indent: int = 4
    wrap_width: Optional[PositiveInt] = None
    page_break_placeholder: Optional[str] = None  # e.g. "<!-- page break -->"
    escape_underscores: bool = True


class MarkdownTextSerializer(BaseModel, BaseTextSerializer):
    """Markdown-specific text item serializer."""

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
        params = MarkdownParams(**kwargs)
        parts: list[str] = []
        escape_html = True
        escape_underscores = True
        if isinstance(item, TitleItem):
            text = f"# {item.text}"
        elif isinstance(item, SectionHeaderItem):
            text = f"{(item.level + 1) * '#'} {item.text}"
        elif isinstance(item, CodeItem):
            text = f"`{item.text}`" if is_inline_scope else f"```\n{item.text}\n```"
            escape_html = False
            escape_underscores = False
        elif isinstance(item, FormulaItem):
            if item.text:
                text = f"${item.text}$" if is_inline_scope else f"$${item.text}$$"
            elif item.orig:
                text = "<!-- formula-not-decoded -->"
            else:
                text = ""
            escape_html = False
            escape_underscores = False
        elif params.wrap_width:
            text = textwrap.fill(item.text, width=params.wrap_width)
        else:
            text = item.text
        parts.append(text)

        if isinstance(item, FloatingItem):
            cap_text = doc_serializer.serialize_captions(item=item, **kwargs).text
            if cap_text:
                parts.append(cap_text)

        text_res = (" " if is_inline_scope else "\n\n").join(parts)
        text_res = doc_serializer.post_process(
            text=text_res,
            escape_html=escape_html,
            escape_underscores=escape_underscores,
            formatting=item.formatting,
            hyperlink=item.hyperlink,
        )
        return SerializationResult(text=text_res)


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

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            text_parts.append(cap_res.text)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
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
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = MarkdownParams(**kwargs)

        texts: list[str] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            texts.append(cap_res.text)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            img_res = self._serialize_image_part(
                item=item,
                doc=doc,
                image_mode=params.image_mode,
                image_placeholder=params.image_placeholder,
            )
            if img_res.text:
                texts.append(img_res.text)

        if params.enable_chart_tables:
            # Check if picture has attached PictureTabularChartData
            tabular_chart_annotations = [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureTabularChartData)
            ]
            if len(tabular_chart_annotations) > 0:
                temp_doc = DoclingDocument(name="temp")
                temp_table = temp_doc.add_table(
                    data=tabular_chart_annotations[0].chart_data
                )
                md_table_content = temp_table.export_to_markdown(temp_doc)
                if len(md_table_content) > 0:
                    texts.append(md_table_content)
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
        params = MarkdownParams(**kwargs)
        my_visited = visited or set()
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )
        sep = "\n"
        my_parts: list[SerializationResult] = []
        for p in parts:
            if p.text and p.text[0] == " " and my_parts:
                my_parts[-1].text = sep.join([my_parts[-1].text, p.text])  # update last
            else:
                my_parts.append(p)

        indent_str = list_level * params.indent * " "
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
            item=item,
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

    params: MarkdownParams = MarkdownParams()

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
        params = self.params.merge_with_patch(patch=kwargs)
        if escape_underscores and params.escape_underscores:
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
    def serialize_page(self, parts: list[SerializationResult]) -> SerializationResult:
        """Serialize a page out of its parts."""
        text_res = "\n\n".join([p.text for p in parts])
        return SerializationResult(text=text_res)

    @override
    def serialize_doc(self, pages: list[SerializationResult]) -> SerializationResult:
        """Serialize a document out of its pages."""
        if self.params.page_break_placeholder is not None:
            sep = f"\n\n{self.params.page_break_placeholder}\n\n"
            text_res = sep.join([p.text for p in pages if p.text])
            return SerializationResult(text=text_res)
        else:
            return self.serialize_page(parts=pages)
