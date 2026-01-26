"""Define classes for layout visualization."""

import logging
from typing import Optional

from PIL import Image as PILImage
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
        show_rows: bool = False
        show_cols: bool = False
        minimal_row_bboxes: bool = True
        minimal_col_bboxes: bool = True

        cell_color: tuple[int, int, int, int] = (255, 0, 0, 32)
        cell_outline: tuple[int, int, int, int] = (255, 0, 0, 128)

        row_color: tuple[int, int, int, int] = (255, 0, 0, 32)
        row_outline: tuple[int, int, int, int] = (255, 0, 0, 128)

        row_header_color: tuple[int, int, int, int] = (0, 255, 0, 32)
        row_header_outline: tuple[int, int, int, int] = (0, 255, 0, 128)

        row_section_color: tuple[int, int, int, int] = (255, 165, 0, 32)
        row_section_outline: tuple[int, int, int, int] = (255, 165, 0, 128)

        col_color: tuple[int, int, int, int] = (0, 255, 0, 32)
        col_outline: tuple[int, int, int, int] = (0, 255, 0, 128)

        col_header_color: tuple[int, int, int, int] = (0, 0, 255, 32)
        col_header_outline: tuple[int, int, int, int] = (0, 0, 255, 128)

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
        # Create transparent overlay for proper alpha compositing
        overlay = PILImage.new("RGBA", page_image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        main_draw = ImageDraw.Draw(page_image)

        for cell in table.data.table_cells:
            if cell.bbox is not None:
                tl_bbox = cell.bbox.to_top_left_origin(page_height=page_height)

                cell_color = self.params.cell_color
                cell_outline = self.params.cell_outline
                if cell.column_header:
                    cell_color = self.params.col_header_color
                    cell_outline = self.params.col_header_outline
                if cell.row_header:
                    cell_color = self.params.row_header_color
                    cell_outline = self.params.row_header_outline
                if cell.row_section:
                    cell_color = self.params.row_section_color
                    cell_outline = self.params.row_section_outline

                cx0, cy0, cx1, cy1 = tl_bbox.as_tuple()
                cx0 *= scale_x
                cx1 *= scale_x
                cy0 *= scale_y
                cy1 *= scale_y

                # Draw fill on overlay (for transparency)
                overlay_draw.rectangle([(cx0, cy0), (cx1, cy1)], fill=cell_color, outline=None)

                # Draw outline on main image
                main_draw.rectangle([(cx0, cy0), (cx1, cy1)], outline=cell_outline, fill=None)

        # Alpha composite the overlay onto the page image
        composited = PILImage.alpha_composite(page_image.convert("RGBA"), overlay)
        page_image.paste(composited.convert(page_image.mode))

    def _draw_table_rows(
        self,
        table: TableItem,
        page_image: Image,
        page_height: float,
        scale_x: float,
        scale_y: float,
    ):
        """Draw individual table rows."""
        # Create transparent overlay for proper alpha compositing
        overlay = PILImage.new("RGBA", page_image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        main_draw = ImageDraw.Draw(page_image)

        rows = table.data.get_row_bounding_boxes(minimal=self.params.minimal_row_bboxes)

        for rid, bbox in rows.items():
            tl_bbox = bbox.to_top_left_origin(page_height=page_height)

            cx0, cy0, cx1, cy1 = tl_bbox.as_tuple()
            cx0 *= scale_x
            cx1 *= scale_x
            cy0 *= scale_y
            cy1 *= scale_y

            # Draw fill on overlay (for transparency)
            overlay_draw.rectangle([(cx0, cy0), (cx1, cy1)], fill=self.params.row_color, outline=None)

            # Draw outline on main image
            main_draw.rectangle([(cx0, cy0), (cx1, cy1)], outline=self.params.row_outline, fill=None)

        # Alpha composite the overlay onto the page image
        composited = PILImage.alpha_composite(page_image.convert("RGBA"), overlay)
        page_image.paste(composited.convert(page_image.mode))

    def _draw_table_cols(
        self,
        table: TableItem,
        page_image: Image,
        page_height: float,
        scale_x: float,
        scale_y: float,
    ):
        """Draw individual table columns."""
        # Create transparent overlay for proper alpha compositing
        overlay = PILImage.new("RGBA", page_image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        main_draw = ImageDraw.Draw(page_image)

        cols = table.data.get_column_bounding_boxes(minimal=self.params.minimal_col_bboxes)

        for cid, bbox in cols.items():
            tl_bbox = bbox.to_top_left_origin(page_height=page_height)

            cx0, cy0, cx1, cy1 = tl_bbox.as_tuple()
            cx0 *= scale_x
            cx1 *= scale_x
            cy0 *= scale_y
            cy1 *= scale_y

            # Draw fill on overlay (for transparency)
            overlay_draw.rectangle([(cx0, cy0), (cx1, cy1)], fill=self.params.col_color, outline=None)

            # Draw outline on main image
            main_draw.rectangle([(cx0, cy0), (cx1, cy1)], outline=self.params.col_outline, fill=None)

        # Alpha composite the overlay onto the page image
        composited = PILImage.alpha_composite(page_image.convert("RGBA"), overlay)
        page_image.paste(composited.convert(page_image.mode))

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
            included_content_layers = set(ContentLayer)

        # Initialise `my_images` beforehand: sometimes, you have the
        # page-images but no DocItems!
        for page_nr, page in doc.pages.items():
            page_image = doc.pages[page_nr].image
            if page_image is None or (pil_img := page_image.pil_image) is None:
                raise RuntimeError("Cannot visualize document without images")
            elif page_nr not in my_images:
                image = pil_img.copy()
                my_images[page_nr] = image

        for idx, (elem, _) in enumerate(doc.iterate_items(included_content_layers=included_content_layers)):
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

                    if self.params.show_rows:
                        self._draw_table_rows(
                            table=elem,
                            page_height=doc.pages[page_nr].size.height,
                            page_image=image,
                            scale_x=image.width / doc.pages[page_nr].size.width,
                            scale_y=image.height / doc.pages[page_nr].size.height,
                        )

                    if self.params.show_cols:
                        self._draw_table_cols(
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
        base_images = self.base_visualizer.get_visualization(doc=doc, **kwargs) if self.base_visualizer else None
        return self._draw_doc_tables(
            doc=doc,
            images=base_images,
        )
