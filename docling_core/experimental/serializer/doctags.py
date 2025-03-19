#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define classes for Doctags serialization."""
import html
from pathlib import Path
from typing import Optional, Union

from pydantic import AnyUrl, BaseModel
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
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DoclingDocument,
    Formatting,
    FormItem,
    InlineGroup,
    KeyValueItem,
    ListItem,
    NodeItem,
    OrderedList,
    PictureClassificationData,
    PictureItem,
    PictureMoleculeData,
    SectionHeaderItem,
    TableItem,
    TextItem,
    UnorderedList,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.tokens import DocumentToken


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


class DocTagsParams(CommonParams):
    """DocTags-specific serialization parameters."""

    new_line: str = ""
    xsize: int = 500
    ysize: int = 500
    add_location: bool = True
    add_content: bool = True  # TODO remove as always true?
    add_cell_location: bool = True
    add_cell_text: bool = True
    add_caption: bool = True

    add_page_index: bool = True  # TODO where are these used
    add_table_cell_location: bool = False  # TODO where are these used
    add_table_cell_text: bool = True  # TODO where are these used


class DocTagsTextSerializer(BaseModel, BaseTextSerializer):
    """DocTags-specific text item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = DocTagsParams(**kwargs)
        wrap_tag: Optional[str] = item.label
        parts: list[str] = []

        if params.add_location:
            location = item.get_location_tokens(
                doc=doc,
                new_line=params.new_line,
                xsize=params.xsize,
                ysize=params.ysize,
            )
            if location:
                parts.append(location)

        if params.add_content:
            text_part = item.text
            escape_html = False  # TODO review
            text_part = doc_serializer.post_process(
                text=text_part,
                escape_html=escape_html,
                formatting=item.formatting,
                hyperlink=item.hyperlink,
            )

            if isinstance(item, CodeItem):
                text_part = f"<_{item.code_language}_>{text_part}"
            else:
                text_part = text_part.strip()
                if isinstance(item, SectionHeaderItem):
                    wrap_tag = f"{item.label}_level_{item.level}"
                elif isinstance(item, ListItem):
                    wrap_tag = None

            if text_part:
                parts.append(text_part)

        res = "".join(parts)
        if wrap_tag is not None:
            res = _wrap(text=res, wrap_tag=wrap_tag)
        return SerializationResult(text=res)


class DocTagsTableSerializer(BaseTableSerializer):
    """DocTags-specific table item serializer."""

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
        params = DocTagsParams(**kwargs)

        body = ""

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if params.add_location:
                body += item.get_location_tokens(
                    doc=doc,
                    new_line=params.new_line,
                    xsize=params.xsize,
                    ysize=params.ysize,
                )

            body += item.export_to_otsl(
                doc,
                params.add_cell_location,
                params.add_cell_text,
                params.xsize,
                params.ysize,
            )

        if params.add_caption and len(item.captions):
            text = doc_serializer.serialize_captions(item, separator="", **kwargs).text

            if len(text):
                body += f"<{DocItemLabel.CAPTION.value}>"
                for caption in item.captions:
                    if caption.cref not in doc_serializer.get_excluded_refs(**kwargs):
                        body += caption.resolve(doc).get_location_tokens(
                            doc=doc,
                            new_line=params.new_line,
                            xsize=params.xsize,
                            ysize=params.ysize,
                        )
                body += f"{text.strip()}"
                body += f"</{DocItemLabel.CAPTION.value}>"
                body += f"{params.new_line}"

        if body:
            body = _wrap(text=body, wrap_tag=DocumentToken.OTSL.value)

        return SerializationResult(text=body)


class DocTagsPictureSerializer(BasePictureSerializer):
    """DocTags-specific picture item serializer."""

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
        params = DocTagsParams(**kwargs)

        parts: list[str] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = ""
            if params.add_location:
                body += item.get_location_tokens(
                    doc=doc,
                    new_line=params.new_line,
                    xsize=params.xsize,
                    ysize=params.ysize,
                )

            classifications = [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureClassificationData)
            ]
            if len(classifications) > 0:
                # ! TODO: currently this code assumes class_name is of type 'str'
                # ! TODO: when it will change to an ENUM --> adapt code
                predicted_class = classifications[0].predicted_classes[0].class_name
                body += DocumentToken.get_picture_classification_token(predicted_class)

            smiles_annotations = [
                ann for ann in item.annotations if isinstance(ann, PictureMoleculeData)
            ]
            if len(smiles_annotations) > 0:
                body += _wrap(
                    text=smiles_annotations[0].smi, wrap_tag=DocumentToken.SMILES.value
                )
            parts.append(body)

        if params.add_caption and len(item.captions):
            text = item.caption_text(doc)

            if len(text):
                body = ""
                for caption in item.captions:
                    if caption.cref not in doc_serializer.get_excluded_refs(**kwargs):
                        body += caption.resolve(doc).get_location_tokens(
                            doc=doc,
                            new_line=params.new_line,
                            xsize=params.xsize,
                            ysize=params.ysize,
                        )
                        body += f"{text.strip()}"
                if body:
                    body = _wrap(text=body, wrap_tag=DocItemLabel.CAPTION.value)
                    parts.append(body)

        text = "".join(parts)
        if text:
            text = _wrap(text=text, wrap_tag=item.label.value)
        return SerializationResult(text=text)


class DocTagsKeyValueSerializer(BaseKeyValueSerializer):
    """DocTags-specific key-value item serializer."""

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
        text_res = ""
        return SerializationResult(text=text_res)


class DocTagsFormSerializer(BaseFormSerializer):
    """DocTags-specific form item serializer."""

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
        text_res = ""
        return SerializationResult(text=text_res)


class DocTagsListSerializer(BaseModel, BaseListSerializer):
    """DocTags-specific list serializer."""

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
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )
        if parts:
            text_res = "\n".join(
                [_wrap(text=p.text, wrap_tag=DocItemLabel.LIST_ITEM) for p in parts]
            )
            text_res = f"{text_res}\n"  # NOTE: explicit
            wrap_tag = (
                DocumentToken.ORDERED_LIST.value
                if isinstance(item, OrderedList)
                else DocumentToken.UNORDERED_LIST.value
            )
            text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        else:
            text_res = ""
        return SerializationResult(text=text_res)


class DocTagsInlineSerializer(BaseInlineSerializer):
    """DocTags-specific inline group serializer."""

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
            **kwargs,
        )
        wrap_tag = DocumentToken.INLINE.value
        text_res = "\n".join([p.text for p in parts if p.text])
        text_res = f"{text_res}\n"  # NOTE: explicit
        text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        return SerializationResult(text=text_res)


class DocTagsFallbackSerializer(BaseFallbackSerializer):
    """DocTags-specific fallback serializer."""

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


class DocTagsDocSerializer(DocSerializer):
    """DocTags-specific document serializer."""

    text_serializer: BaseTextSerializer = DocTagsTextSerializer()
    table_serializer: BaseTableSerializer = DocTagsTableSerializer()
    picture_serializer: BasePictureSerializer = DocTagsPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = DocTagsKeyValueSerializer()
    form_serializer: BaseFormSerializer = DocTagsFormSerializer()
    fallback_serializer: BaseFallbackSerializer = DocTagsFallbackSerializer()

    list_serializer: BaseListSerializer = DocTagsListSerializer()
    inline_serializer: BaseInlineSerializer = DocTagsInlineSerializer()

    params: DocTagsParams = DocTagsParams()

    def post_process(
        self,
        text: str,
        *,
        escape_html: bool = True,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        **kwargs,
    ) -> str:
        """Apply some text post-processing steps."""
        res = text
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
        text_res = "\n".join([p.text for p in parts])
        return SerializationResult(text=text_res)

    @override
    def serialize_doc(self, pages: list[SerializationResult]) -> SerializationResult:
        """Serialize a document out of its pages."""
        page_sep = f"\n<{DocumentToken.PAGE_BREAK.value}>\n"
        content = page_sep.join([p.text for p in pages if p.text])
        wrap_tag = DocumentToken.DOCUMENT.value
        text_res = f"<{wrap_tag}>{content}\n</{wrap_tag}>"
        return SerializationResult(text=text_res)
