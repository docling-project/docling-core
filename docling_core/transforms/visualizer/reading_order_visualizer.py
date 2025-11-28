"""Define classes for reading order visualization."""

from copy import deepcopy
from typing import Optional, Union

from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont
from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.visualizer.base import BaseVisualizer
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    DoclingDocument,
    PictureItem,
)


class _NumberDrawingData(BaseModel):
    xy: tuple[float, float]
    text: str


class ReadingOrderVisualizer(BaseVisualizer):
    """Reading order visualizer."""

    class Params(BaseModel):
        """Layout visualization parameters."""

        show_label: bool = True
        show_branch_numbering: bool = False
        content_layers: set[ContentLayer] = {
            cl for cl in ContentLayer if cl != ContentLayer.BACKGROUND
        }

    base_visualizer: Optional[BaseVisualizer] = None
    params: Params = Params()

    def _get_picture_context(
        self, elem: DocItem, doc: DoclingDocument
    ) -> Optional[str]:
        """Get the picture self_ref if element is nested inside a PictureItem, None otherwise."""
        current = elem
        while current.parent is not None:
            parent = current.parent.resolve(doc)
            if isinstance(parent, PictureItem):
                return parent.self_ref
            if not isinstance(parent, DocItem):
                break
            current = parent
        return None

    def _draw_arrow(
        self,
        draw: ImageDraw.ImageDraw,
        arrow_coords: tuple[float, float, float, float],
        line_width: int = 2,
        color: str = "red",
    ):
        """Draw an arrow inside the given draw object."""
        x0, y0, x1, y1 = arrow_coords

        # Arrow parameters
        start_point = (x0, y0)  # Starting point of the arrow
        end_point = (x1, y1)  # Ending point of the arrow
        arrowhead_length = 20  # Length of the arrowhead
        arrowhead_width = 10  # Width of the arrowhead

        # Draw the arrow shaft (line)
        draw.line([start_point, end_point], fill=color, width=line_width)

        # Calculate the arrowhead points
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        distance = (dx**2 + dy**2) ** 0.5 + 0.01  # Length of the arrow shaft

        # Normalized direction vector for the arrow shaft
        ux, uy = dx / distance, dy / distance

        # Base of the arrowhead
        base_x = end_point[0] - ux * arrowhead_length
        base_y = end_point[1] - uy * arrowhead_length

        # Left and right points of the arrowhead
        left_x = base_x - uy * arrowhead_width
        left_y = base_y + ux * arrowhead_width
        right_x = base_x + uy * arrowhead_width
        right_y = base_y - ux * arrowhead_width

        # Draw the arrowhead (triangle)
        draw.polygon(
            [end_point, (left_x, left_y), (right_x, right_y)],
            fill=color,
        )
        return draw

    def _draw_doc_reading_order(
        self,
        doc: DoclingDocument,
        images: Optional[dict[Optional[int], Image]] = None,
    ):
        """Draw the reading order."""
        font: Union[ImageFont.ImageFont, FreeTypeFont]
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            # Fallback to default font if arial is not available
            font = ImageFont.load_default()

        # Separate reading order paths for outside vs inside pictures
        # Key: (page_no, picture_ref_or_None) -> (x0, y0, element_index)
        # picture_ref is None for elements outside any picture, otherwise the picture's self_ref
        reading_order_state: dict[
            tuple[int, Optional[str]], tuple[float, float, int]
        ] = {}
        number_data_to_draw: dict[int, list[_NumberDrawingData]] = {}
        # Only int keys are used (from prov.page_no), even if input images has Optional[int] keys
        my_images: dict[int, Image] = {
            k: v for k, v in (images or {}).items() if k is not None
        }
        prev_page: Optional[int] = None
        element_index = 0

        for elem, _ in doc.iterate_items(
            included_content_layers=self.params.content_layers,
            traverse_pictures=True,
        ):
            if not isinstance(elem, DocItem):
                continue

            picture_ref = self._get_picture_context(elem, doc)
            # Include all elements in reading order:
            # - Top-level PictureItems are part of the outer reading order (picture_ref is None)
            # - Nested PictureItems are part of their parent picture's reading order (picture_ref is not None)
            # - Other elements follow the same pattern

            if len(elem.prov) == 0:
                continue  # Skip elements without provenances

            for prov in elem.prov:
                page_no = prov.page_no
                image = my_images.get(page_no)

                if page_no not in number_data_to_draw:
                    number_data_to_draw[page_no] = []

                if image is None or prev_page is None or page_no != prev_page:
                    # new page begins - reset all reading order paths
                    prev_page = page_no
                    reading_order_state.clear()

                    if image is None:
                        page_image = doc.pages[page_no].image
                        if (
                            page_image is None
                            or (pil_img := page_image.pil_image) is None
                        ):
                            raise RuntimeError(
                                "Cannot visualize document without images"
                            )
                        else:
                            image = deepcopy(pil_img)
                            my_images[page_no] = image
                draw = ImageDraw.Draw(image, "RGBA")

                tlo_bbox = prov.bbox.to_top_left_origin(
                    page_height=doc.pages[prov.page_no].size.height
                )
                ro_bbox = tlo_bbox.normalized(doc.pages[prov.page_no].size)
                ro_bbox.l = round(ro_bbox.l * image.width)  # noqa: E741
                ro_bbox.r = round(ro_bbox.r * image.width)
                ro_bbox.t = round(ro_bbox.t * image.height)
                ro_bbox.b = round(ro_bbox.b * image.height)

                if ro_bbox.b > ro_bbox.t:
                    ro_bbox.b, ro_bbox.t = ro_bbox.t, ro_bbox.b

                path_key = (page_no, picture_ref)
                state = reading_order_state.get(path_key)

                x1 = (ro_bbox.l + ro_bbox.r) / 2.0
                y1 = (ro_bbox.b + ro_bbox.t) / 2.0

                if state is None:
                    # Start of a new reading order path (outside or inside picture)
                    reading_order_state[path_key] = (x1, y1, element_index)
                    number_data_to_draw[page_no].append(
                        _NumberDrawingData(
                            xy=(x1, y1),
                            text=f"{element_index}",
                        )
                    )
                    element_index += 1
                else:
                    # Continue existing reading order path
                    x0, y0, _ = state
                    # Use different color for picture-internal paths
                    arrow_color = "blue" if picture_ref is not None else "red"
                    draw = self._draw_arrow(
                        draw=draw,
                        arrow_coords=(x0, y0, x1, y1),
                        line_width=2,
                        color=arrow_color,
                    )
                    reading_order_state[path_key] = (x1, y1, state[2])

        if self.params.show_branch_numbering:
            # post-drawing the numbers to ensure they are rendered on top-layer
            for page in number_data_to_draw:
                if (image := my_images.get(page)) is None:
                    continue
                draw = ImageDraw.Draw(image, "RGBA")

                for num_item in number_data_to_draw[page]:

                    text_bbox = draw.textbbox(num_item.xy, num_item.text, font)
                    text_bg_padding = 5
                    draw.ellipse(
                        [
                            (
                                text_bbox[0] - text_bg_padding,
                                text_bbox[1] - text_bg_padding,
                            ),
                            (
                                text_bbox[2] + text_bg_padding,
                                text_bbox[3] + text_bg_padding,
                            ),
                        ],
                        fill="orange",
                    )
                    draw.text(
                        num_item.xy,
                        text=num_item.text,
                        fill="black",
                        font=font,
                    )

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
        return self._draw_doc_reading_order(
            doc=doc,
            images=base_images,
        )
