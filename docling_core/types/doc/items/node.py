"""Base document-tree node items: NodeItem, DocItem, FloatingItem."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Optional

from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from docling_core.types.base import _JSON_POINTER_REGEX
from docling_core.types.doc.common.annotations import BaseAnnotation
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.common.meta import BaseMeta, FloatingMeta
from docling_core.types.doc.common.origin import SourceType
from docling_core.types.doc.common.reference import FineRef, ImageRef, ProvenanceItem, RefItem
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.tokens import DocumentToken

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class NodeItem(BaseModel):
    """NodeItem."""

    self_ref: str = Field(pattern=_JSON_POINTER_REGEX)
    parent: Optional[RefItem] = None
    children: list[RefItem] = []

    content_layer: ContentLayer = ContentLayer.BODY

    model_config = ConfigDict(extra="forbid")

    meta: Optional[BaseMeta] = None

    def get_ref(self) -> RefItem:
        """get_ref."""
        return RefItem(cref=self.self_ref)

    def _get_parent_ref(self, doc: "DoclingDocument", stack: list[int]) -> Optional[RefItem]:
        """get_parent_ref."""
        if len(stack) == 0:
            return self.parent
        elif len(stack) > 0 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._get_parent_ref(doc=doc, stack=stack[1:])

        return None

    def _delete_child(self, doc: "DoclingDocument", stack: list[int]) -> bool:
        """Delete child node in tree."""
        if len(stack) == 1 and stack[0] < len(self.children):
            del self.children[stack[0]]
            return True
        elif len(stack) > 1 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._delete_child(doc=doc, stack=stack[1:])

        return False

    def _update_child(self, doc: "DoclingDocument", stack: list[int], new_ref: RefItem) -> bool:
        """Update child node in tree."""
        if len(stack) == 1 and stack[0] < len(self.children):
            # ensure the parent is correct
            new_item = new_ref.resolve(doc=doc)
            new_item.parent = self.get_ref()

            self.children[stack[0]] = new_ref
            return True
        elif len(stack) > 1 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._update_child(doc=doc, stack=stack[1:], new_ref=new_ref)

        return False

    def _add_child(self, doc: "DoclingDocument", stack: list[int], new_ref: RefItem) -> bool:
        """Append child to node identified by stack."""
        if len(stack) == 0:
            # ensure the parent is correct
            new_item = new_ref.resolve(doc=doc)
            new_item.parent = self.get_ref()

            self.children.append(new_ref)
            return True
        elif len(stack) > 0 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._add_child(doc=doc, stack=stack[1:], new_ref=new_ref)

        return False

    def _add_sibling(
        self,
        doc: "DoclingDocument",
        stack: list[int],
        new_ref: RefItem,
        after: bool = True,
    ) -> bool:
        """Add sibling node in tree."""
        if len(stack) == 1 and stack[0] <= len(self.children) and (not after):
            # ensure the parent is correct
            new_item = new_ref.resolve(doc=doc)
            new_item.parent = self.get_ref()

            self.children.insert(stack[0], new_ref)
            return True
        elif len(stack) == 1 and stack[0] < len(self.children) and (after):
            # ensure the parent is correct
            new_item = new_ref.resolve(doc=doc)
            new_item.parent = self.get_ref()

            self.children.insert(stack[0] + 1, new_ref)
            return True
        elif len(stack) > 1 and stack[0] < len(self.children):
            item = self.children[stack[0]].resolve(doc)
            return item._add_sibling(doc=doc, stack=stack[1:], new_ref=new_ref, after=after)

        return False


class DocItem(NodeItem):
    """Base type for any element that carries content, can be a leaf node."""

    label: DocItemLabel
    prov: list[ProvenanceItem] = []
    source: Annotated[
        list[SourceType],
        Field(
            description="The provenance of this document item. Currently, it is only used for media track provenance."
        ),
    ] = []
    comments: list[FineRef] = []  # References to comment items annotating this content

    @model_serializer(mode="wrap")
    def _custom_pydantic_serialize(self, handler: SerializerFunctionWrapHandler) -> dict:
        dumped = handler(self)

        # suppress serializing comment and source lists when empty:
        for field in {"comments", "source"}:
            if dumped.get(field) == []:
                del dumped[field]

        return dumped

    def get_location_tokens(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        self_closing: bool = False,
    ) -> str:
        """Get the location string for the BaseCell."""
        if not len(self.prov):
            return ""

        location = ""
        for prov in self.prov:
            page_w, page_h = doc.pages[prov.page_no].size.as_tuple()

            loc_str = DocumentToken.get_location(
                bbox=prov.bbox.to_top_left_origin(page_h).as_tuple(),
                page_w=page_w,
                page_h=page_h,
                xsize=xsize,
                ysize=ysize,
                self_closing=self_closing,
            )
            location += loc_str

        return location

    def get_image(self, doc: "DoclingDocument", prov_index: int = 0) -> Optional[PILImage.Image]:
        """Returns the image of this DocItem.

        The function returns None if this DocItem has no valid provenance or
        if a valid image of the page containing this DocItem is not available
        in doc.
        """
        if not self.prov or prov_index >= len(self.prov):
            return None
        prov = self.prov[prov_index]
        if not isinstance(prov, ProvenanceItem):
            return None

        page = doc.pages.get(prov.page_no)
        if page is None or page.size is None or page.image is None:
            return None

        page_image = page.image.pil_image
        if not page_image:
            return None
        crop_bbox = (
            self.prov[prov_index]
            .bbox.to_top_left_origin(page_height=page.size.height)
            .scale_to_size(old_size=page.size, new_size=page.image.size)
            # .scaled(scale=page_image.height / page.size.height)
        )
        return page_image.crop(crop_bbox.as_tuple())

    def get_annotations(self) -> Sequence[BaseAnnotation]:
        """Get the annotations of this DocItem."""
        return []


class FloatingItem(DocItem):
    """FloatingItem."""

    meta: Optional[FloatingMeta] = None

    captions: list[RefItem] = []
    references: list[RefItem] = []
    footnotes: list[RefItem] = []
    image: Optional[ImageRef] = None

    def caption_text(self, doc: "DoclingDocument") -> str:
        """Computes the caption as a single text."""
        text = ""
        for cap in self.captions:
            text += cap.resolve(doc).text
        return text

    def get_image(self, doc: "DoclingDocument", prov_index: int = 0) -> Optional[PILImage.Image]:
        """Returns the image corresponding to this FloatingItem.

        This function returns the PIL image from self.image if one is available.
        Otherwise, it uses DocItem.get_image to get an image of this FloatingItem.

        In particular, when self.image is None, the function returns None if this
        FloatingItem has no valid provenance or the doc does not contain a valid image
        for the required page.
        """
        if self.image is not None:
            return self.image.pil_image
        return super().get_image(doc=doc, prov_index=prov_index)
