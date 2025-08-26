#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for Markdown serialization."""
import html
import re
import textwrap
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import AnyUrl, BaseModel, PositiveInt
from tabulate import tabulate
from typing_extensions import override

from docling_core.transforms.serializer.base import (
    BaseAnnotationSerializer,
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
from docling_core.transforms.serializer.common import (
    CommonParams,
    DocSerializer,
    _get_annotation_text,
    create_ser_result,
)
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CodeItem,
    ContentLayer,
    DescriptionAnnotation,
    DocItem,
    DoclingDocument,
    FloatingItem,
    Formatting,
    FormItem,
    FormulaItem,
    ImageRef,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    ListItem,
    NodeItem,
    PictureClassificationData,
    PictureItem,
    PictureMoleculeData,
    PictureTabularChartData,
    RichTableCell,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)


def _get_annotation_ser_result(
    ann_kind: str, ann_text: str, mark_annotation: bool, doc_item: DocItem
):
    return create_ser_result(
        text=(
            (
                f'<!--<annotation kind="{ann_kind}">-->'
                f"{ann_text}"
                f"<!--<annotation/>-->"
            )
            if mark_annotation
            else ann_text
        ),
        span_source=doc_item,
    )


class OrigListItemMarkerMode(str, Enum):
    """Display mode for original list item marker."""

    NEVER = "never"
    ALWAYS = "always"
    AUTO = "auto"


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
    escape_html: bool = True
    include_annotations: bool = True
    mark_annotations: bool = False
    orig_list_item_marker_mode: OrigListItemMarkerMode = OrigListItemMarkerMode.AUTO
    ensure_valid_list_item_marker: bool = True


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
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited if visited is not None else set()
        params = MarkdownParams(**kwargs)
        res_parts: list[SerializationResult] = []
        text = item.text
        escape_html = True
        escape_underscores = True
        processing_pending = True
        if isinstance(item, (ListItem, TitleItem, SectionHeaderItem)):
            # case where processing/formatting should be applied first (in inner scope)
            processing_pending = False
            if (
                text == ""
                and len(item.children) == 1
                and isinstance(
                    (child_group := item.children[0].resolve(doc)), InlineGroup
                )
            ):
                # case of inline within heading / list item
                ser_res = doc_serializer.serialize(item=child_group)
                text = ser_res.text
                for span in ser_res.spans:
                    my_visited.add(span.item.self_ref)
            else:
                text = doc_serializer.post_process(
                    text=text,
                    escape_html=escape_html,
                    escape_underscores=escape_underscores,
                    formatting=item.formatting,
                    hyperlink=item.hyperlink,
                )

            if isinstance(item, ListItem):
                pieces: list[str] = []
                case_auto = (
                    params.orig_list_item_marker_mode == OrigListItemMarkerMode.AUTO
                    and bool(re.search(r"[a-zA-Z0-9]", item.marker))
                )
                case_already_valid = (
                    params.ensure_valid_list_item_marker
                    and params.orig_list_item_marker_mode
                    != OrigListItemMarkerMode.NEVER
                    and (
                        item.marker in ["-", "*", "+"]
                        or re.fullmatch(r"\d+\.", item.marker)
                    )
                )

                # wrap with outer marker (if applicable)
                if params.ensure_valid_list_item_marker and not case_already_valid:
                    assert item.parent and isinstance(
                        (list_group := item.parent.resolve(doc)), ListGroup
                    )
                    if list_group.first_item_is_enumerated(doc) and (
                        params.orig_list_item_marker_mode != OrigListItemMarkerMode.AUTO
                        or not item.marker
                    ):
                        pos = -1
                        for i, child in enumerate(list_group.children):
                            if child.resolve(doc) == item:
                                pos = i
                                break
                        md_marker = f"{pos + 1}."
                    else:
                        md_marker = "-"
                    pieces.append(md_marker)

                # include original marker (if applicable)
                if item.marker and (
                    params.orig_list_item_marker_mode == OrigListItemMarkerMode.ALWAYS
                    or case_auto
                    or case_already_valid
                ):
                    pieces.append(item.marker)

                pieces.append(text)
                text_part = " ".join(pieces)
            else:
                num_hashes = 1 if isinstance(item, TitleItem) else item.level + 1
                text_part = f"{num_hashes * '#'} {text}"
        elif isinstance(item, CodeItem):
            text_part = f"`{text}`" if is_inline_scope else f"```\n{text}\n```"
            escape_html = False
            escape_underscores = False
        elif isinstance(item, FormulaItem):
            if text:
                text_part = f"${text}$" if is_inline_scope else f"$${text}$$"
            elif item.orig:
                text_part = "<!-- formula-not-decoded -->"
            else:
                text_part = ""
            escape_html = False
            escape_underscores = False
        elif params.wrap_width:
            # although wrapping is not guaranteed if post-processing makes changes
            text_part = textwrap.fill(text, width=params.wrap_width)
        else:
            text_part = text

        if text_part:
            text_res = create_ser_result(text=text_part, span_source=item)
            res_parts.append(text_res)

        if isinstance(item, FloatingItem):
            cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
            if cap_res.text:
                res_parts.append(cap_res)

        text = (" " if is_inline_scope else "\n\n").join([r.text for r in res_parts])
        if processing_pending:
            text = doc_serializer.post_process(
                text=text,
                escape_html=escape_html,
                escape_underscores=escape_underscores,
                formatting=item.formatting,
                hyperlink=item.hyperlink,
            )
        return create_ser_result(text=text, span_source=res_parts)


class MarkdownAnnotationSerializer(BaseModel, BaseAnnotationSerializer):
    """Markdown-specific annotation serializer."""

    def serialize(
        self,
        *,
        item: DocItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's annotations."""
        params = MarkdownParams(**kwargs)

        res_parts: list[SerializationResult] = []
        for ann in item.get_annotations():
            if isinstance(
                ann,
                (
                    PictureClassificationData,
                    DescriptionAnnotation,
                    PictureMoleculeData,
                ),
            ):
                if ann_text := _get_annotation_text(ann):
                    ann_res = create_ser_result(
                        text=(
                            (
                                f'<!--<annotation kind="{ann.kind}">-->'
                                f"{ann_text}"
                                f"<!--<annotation/>-->"
                            )
                            if params.mark_annotations
                            else ann_text
                        ),
                        span_source=item,
                    )
                    res_parts.append(ann_res)
        return create_ser_result(
            text="\n\n".join([r.text for r in res_parts if r.text]),
            span_source=item,
        )


class MarkdownTableSerializer(BaseTableSerializer):
    """Markdown-specific table item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = MarkdownParams(**kwargs)
        res_parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):

            if params.include_annotations:

                ann_res = doc_serializer.serialize_annotations(
                    item=item,
                    **kwargs,
                )
                if ann_res.text:
                    res_parts.append(ann_res)

            rows = [
                [
                    # make sure that md tables are not broken
                    # due to newline chars in the text
                    (
                        doc_serializer.serialize(
                            item=col.ref.resolve(doc=doc), **kwargs
                        ).text
                        if isinstance(col, RichTableCell)
                        else col.text
                    ).replace("\n", " ")
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
                res_parts.append(create_ser_result(text=table_text, span_source=item))

        text_res = "\n\n".join([r.text for r in res_parts])

        return create_ser_result(text=text_res, span_source=res_parts)


class MarkdownPictureSerializer(BasePictureSerializer):
    """Markdown-specific picture item serializer."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = MarkdownParams(**kwargs)

        res_parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            res_parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if params.include_annotations:
                ann_res = doc_serializer.serialize_annotations(
                    item=item,
                    **kwargs,
                )
                if ann_res.text:
                    res_parts.append(ann_res)

            img_res = self._serialize_image_part(
                item=item,
                doc=doc,
                image_mode=params.image_mode,
                image_placeholder=params.image_placeholder,
            )
            if img_res.text:
                res_parts.append(img_res)

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
                    res_parts.append(
                        create_ser_result(text=md_table_content, span_source=item)
                    )
        text_res = "\n\n".join([r.text for r in res_parts if r.text])

        return create_ser_result(text=text_res, span_source=res_parts)

    def _serialize_image_part(
        self,
        item: PictureItem,
        doc: DoclingDocument,
        image_mode: ImageRefMode,
        image_placeholder: str,
        **kwargs: Any,
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

        return create_ser_result(text=text_res, span_source=item)


class MarkdownKeyValueSerializer(BaseKeyValueSerializer):
    """Markdown-specific key-value item serializer."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # TODO add actual implementation
        if item.self_ref not in doc_serializer.get_excluded_refs():
            return create_ser_result(
                text="<!-- missing-key-value-item -->",
                span_source=item,
            )
        else:
            return create_ser_result()


class MarkdownFormSerializer(BaseFormSerializer):
    """Markdown-specific form item serializer."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # TODO add actual implementation
        if item.self_ref not in doc_serializer.get_excluded_refs():
            return create_ser_result(
                text="<!-- missing-form-item -->",
                span_source=item,
            )
        else:
            return create_ser_result()


class MarkdownListSerializer(BaseModel, BaseListSerializer):
    """Markdown-specific list serializer."""

    @override
    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = MarkdownParams(**kwargs)
        my_visited = visited if visited is not None else set()
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
            if (
                my_parts
                and p.text
                and p.spans
                and p.spans[0].item.parent
                and isinstance(p.spans[0].item.parent.resolve(doc), InlineGroup)
            ):
                my_parts[-1].text = f"{my_parts[-1].text}{p.text}"  # append to last
                my_parts[-1].spans.extend(p.spans)
            else:
                my_parts.append(p)

        indent_str = list_level * params.indent * " "
        text_res = sep.join(
            [
                # avoid additional marker on already evaled sublists
                (c.text if c.text and c.text[0] == " " else f"{indent_str}{c.text}")
                for c in my_parts
            ]
        )
        return create_ser_result(text=text_res, span_source=my_parts)


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
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited if visited is not None else set()
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
            **kwargs,
        )
        text_res = " ".join([p.text for p in parts if p.text])
        return create_ser_result(text=text_res, span_source=parts)


class MarkdownFallbackSerializer(BaseFallbackSerializer):
    """Markdown-specific fallback serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        if isinstance(item, DocItem):
            return create_ser_result(
                text="<!-- missing-text -->",
                span_source=item,
            )
        else:
            return create_ser_result()


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

    annotation_serializer: BaseAnnotationSerializer = MarkdownAnnotationSerializer()

    params: MarkdownParams = MarkdownParams()

    @override
    def serialize_bold(self, text: str, **kwargs: Any):
        """Apply Markdown-specific bold serialization."""
        return f"**{text}**"

    @override
    def serialize_italic(self, text: str, **kwargs: Any):
        """Apply Markdown-specific italic serialization."""
        return f"*{text}*"

    @override
    def serialize_strikethrough(self, text: str, **kwargs: Any):
        """Apply Markdown-specific strikethrough serialization."""
        return f"~~{text}~~"

    @override
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ):
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
        **kwargs: Any,
    ) -> str:
        """Apply some text post-processing steps."""
        res = text
        params = self.params.merge_with_patch(patch=kwargs)
        if escape_underscores and params.escape_underscores:
            res = self._escape_underscores(text)
        if escape_html and params.escape_html:
            res = html.escape(res, quote=False)
        res = super().post_process(
            text=res,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return res

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize a document out of its parts."""
        text_res = "\n\n".join([p.text for p in parts if p.text])
        if self.requires_page_break():
            page_sep = self.params.page_break_placeholder or ""
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        return create_ser_result(text=text_res, span_source=parts)

    @override
    def requires_page_break(self) -> bool:
        """Whether to add page breaks."""
        return self.params.page_break_placeholder is not None
