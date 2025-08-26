#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT
#

"""Define base classes for serialization."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import AnyUrl, BaseModel

from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    FloatingItem,
    FormItem,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    NodeItem,
    PictureItem,
    TableItem,
    TextItem,
)


class Span(BaseModel):
    """Class encapsulating fine-granular document span information."""

    item: DocItem
    # prov_idx: Optional[PositiveInt] = None  # None to be interpreted as whole DocItem


class SerializationResult(BaseModel):
    """SerializationResult."""

    text: str = ""
    spans: list[Span] = []
    # group: Optional[GroupItem] = None  # set when result reflects specific group item

    def get_unique_doc_items(self) -> list[DocItem]:
        """Get the doc items corresponding to this result."""
        seen_doc_item_refs: set[str] = set()
        doc_items: list[DocItem] = []
        for span in self.spans:
            if span.item.self_ref not in seen_doc_item_refs:
                seen_doc_item_refs.add(span.item.self_ref)
                doc_items.append(span.item)
        return doc_items


class BaseTextSerializer(ABC):
    """Base class for text item serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        ...


class BaseTableSerializer(ABC):
    """Base class for table item serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        ...


class BasePictureSerializer(ABC):
    """Base class for picture item serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        ...


class BaseKeyValueSerializer(ABC):
    """Base class for key value item serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        ...


class BaseFormSerializer(ABC):
    """Base class for form item serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        ...


class BaseListSerializer(ABC):
    """Base class for list serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: ListGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        ...


class BaseInlineSerializer(ABC):
    """Base class for inline serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        ...


class BaseFallbackSerializer(ABC):
    """Base fallback class for item serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed item."""
        ...


class BaseDocSerializer(ABC):
    """Base class for document serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: Optional[NodeItem] = None,
        **kwargs: Any,
    ) -> SerializationResult:
        """Run the serialization."""
        ...

    @abstractmethod
    def serialize_bold(self, text: str, **kwargs: Any) -> str:
        """Hook for bold formatting serialization."""
        ...

    @abstractmethod
    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        """Hook for italic formatting serialization."""
        ...

    @abstractmethod
    def serialize_underline(self, text: str, **kwargs: Any) -> str:
        """Hook for underline formatting serialization."""
        ...

    @abstractmethod
    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        """Hook for strikethrough formatting serialization."""
        ...

    @abstractmethod
    def serialize_subscript(self, text: str, **kwargs: Any) -> str:
        """Hook for subscript formatting serialization."""
        ...

    @abstractmethod
    def serialize_superscript(self, text: str, **kwargs: Any) -> str:
        """Hook for superscript formatting serialization."""
        ...

    @abstractmethod
    def serialize_hyperlink(
        self,
        text: str,
        hyperlink: Union[AnyUrl, Path],
        **kwargs: Any,
    ) -> str:
        """Hook for hyperlink serialization."""
        ...

    @abstractmethod
    def get_parts(
        self,
        item: Optional[NodeItem] = None,
        **kwargs: Any,
    ) -> list[SerializationResult]:
        """Get the components to be combined for serializing this node."""
        ...

    @abstractmethod
    def post_process(
        self,
        text: str,
        **kwargs: Any,
    ) -> str:
        """Apply some text post-processing steps."""
        ...

    @abstractmethod
    def serialize_captions(
        self,
        item: FloatingItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's captions."""
        ...

    @abstractmethod
    def serialize_annotations(
        self,
        item: DocItem,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize the item's annotations."""
        ...

    @abstractmethod
    def get_excluded_refs(self, **kwargs: Any) -> set[str]:
        """Get references to excluded items."""
        ...

    @abstractmethod
    def requires_page_break(self) -> bool:
        """Whether to add page breaks."""
        ...


class BaseSerializerProvider(ABC):
    """Base class for document serializer providers."""

    @abstractmethod
    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        """Get a the associated serializer."""
        ...


class BaseAnnotationSerializer(ABC):
    """Base class for annotation serializers."""

    @abstractmethod
    def serialize(
        self,
        *,
        item: DocItem,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serializes the passed annotation."""
        ...
