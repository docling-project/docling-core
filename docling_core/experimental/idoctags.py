"""Define classes for DocTags serialization."""

from typing import Any, Final

from typing_extensions import override

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.doctags import (
    DocTagsDocSerializer,
    DocTagsParams,
    _get_delim,
)
from docling_core.types.doc.tokens import DocumentToken

DOCTAGS_VERSION: Final = "1.0.0"


class IDocTagsParams(DocTagsParams):
    """DocTags-specific serialization parameters."""


class IDocTagsDocSerializer(DocTagsDocSerializer):
    """DocTags document serializer."""

    @override
    def serialize_doc(
        self,
        *,
        parts: list[SerializationResult],
        **kwargs: Any,
    ) -> SerializationResult:
        """DocTags-specific document serializer."""
        delim = _get_delim(params=self.params)
        text_res = delim.join([p.text for p in parts if p.text])

        if self.params.add_page_break:
            page_sep = f"<{DocumentToken.PAGE_BREAK.value}>"
            for full_match, _, _ in self._get_page_breaks(text=text_res):
                text_res = text_res.replace(full_match, page_sep)

        wrap_tag = DocumentToken.DOCUMENT.value
        text_res = f"<{wrap_tag}><version>{DOCTAGS_VERSION}</version>{text_res}{delim}</{wrap_tag}>"
        return create_ser_result(text=text_res, span_source=parts)
