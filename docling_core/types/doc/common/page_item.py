"""Per-page item model referenced by ``DoclingDocument.pages``."""

from typing import Optional

from pydantic import BaseModel

from docling_core.types.doc.base import Size
from docling_core.types.doc.common.reference import ImageRef


class PageItem(BaseModel):
    """PageItem."""

    # A page carries separate root items for furniture and body,
    # only referencing items on the page
    size: Size
    image: Optional[ImageRef] = None
    page_no: int
