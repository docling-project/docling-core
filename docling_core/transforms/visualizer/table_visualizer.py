"""Define classes for layout visualization."""

import logging
from copy import deepcopy
from typing import Optional

from PIL import ImageDraw
from PIL.Image import Image
from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.visualizer.base import BaseVisualizer
from docling_core.types.doc.document import ContentLayer, DoclingDocument, TableItem

_log = logging.getLogger(__name__)


class TableVisualizer(BaseVisualizer):
    """Table visualizer."""

    class Params(BaseModel):
        """Table visualization parameters."""

        # show_Label: bool = False
        show_cells: bool = True
        # show_rows: bool = False
        # show_cols: bool = False

    base_visualizer: Optional[BaseVisualizer] = None
    params: Params = Params()

    def _draw_table_cells(
        self,
        table: TableItem,
        page_image: Image,
        page_height: float,
        scale_x: float,
        scale_y: float,
    ):
        """Draw individual table cells."""
        draw = ImageDraw.Draw(page_image, "RGBA")

        for cell in table.data.table_cells:
            if cell.bbox is not None:

                tl_bbox = cell.bbox.to_top_left_origin(page_height=page_height)

                cell_color = (256, 0, 0, 32)  # Transparent black for cells

                cx0, cy0, cx1, cy1 = tl_bbox.as_tuple()
                cx0 *= scale_x
                cx1 *= scale_x
                cy0 *= scale_y
                cy1 *= scale_y

                draw.rectangle(
                    [(cx0, cy0), (cx1, cy1)],
                    outline=(256, 0, 0, 128),
                    fill=cell_color,
                )

    def _draw_doc_tables(
        self,
        doc: DoclingDocument,
        images: Optional[dict[Optional[int], Image]] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
    ):
        """Draw the document tables."""
        my_images: dict[Optional[int], Image] = {}

        if images is not None:
            my_images = images

        if included_content_layers is None:
            included_content_layers = {c for c in ContentLayer}

        # Initialise `my_images` beforehand: sometimes, you have the
        # page-images but no DocItems!
        for page_nr, page in doc.pages.items():
            page_image = doc.pages[page_nr].image
            if page_image is None or (pil_img := page_image.pil_image) is None:
                raise RuntimeError("Cannot visualize document without images")
            elif page_nr not in my_images:
                image = deepcopy(pil_img)
                my_images[page_nr] = image

        for idx, (elem, _) in enumerate(
            doc.iterate_items(included_content_layers=included_content_layers)
        ):
            if not isinstance(elem, TableItem):
                continue
            if len(elem.prov) == 0:
                continue  # Skip elements without provenances

            if len(elem.prov) == 1:

                page_nr = elem.prov[0].page_no

                if page_nr in my_images:
                    image = my_images[page_nr]

                    if self.params.show_cells:
                        self._draw_table_cells(
                            table=elem,
                            page_height=doc.pages[page_nr].size.height,
                            page_image=image,
                            scale_x=image.width / doc.pages[page_nr].size.width,
                            scale_y=image.height / doc.pages[page_nr].size.height,
                        )

                else:
                    raise RuntimeError(f"Cannot visualize page-image for {page_nr}")

            else:
                _log.error("Can not yet visualise tables with multiple provenances")

        return my_images

    @override
    def get_visualization(
        self,
        *,
        doc: DoclingDocument,
        **kwargs,
    ) -> dict[Optional[int], Image]:
        """Get visualization of the document as images by page."""
        base_images = (
            self.base_visualizer.get_visualization(doc=doc, **kwargs)
            if self.base_visualizer
            else None
        )
        return self._draw_doc_tables(
            doc=doc,
            images=base_images,
        )
