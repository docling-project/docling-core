"""Define classes for reading order visualization."""

from copy import deepcopy
from typing import Optional, Union

from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont
from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.visualizer.base import BaseVisualizer
from docling_core.types.doc.document import ContentLayer, DocItem, DoclingDocument


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
        angle = (dx**2 + dy**2) ** 0.5 + 0.01  # Length of the arrow shaft

        # Normalized direction vector for the arrow shaft
        ux, uy = dx / angle, dy / angle

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
        x0, y0 = None, None
        number_data_to_draw: dict[Optional[int], list[_NumberDrawingData]] = {}
        my_images: dict[Optional[int], Image] = images or {}
        prev_page = None
        i = 0
        for elem, _ in doc.iterate_items(
            included_content_layers=self.params.content_layers,
        ):
            if not isinstance(elem, DocItem):
                continue
            if len(elem.prov) == 0:
                continue  # Skip elements without provenances

            for prov in elem.prov:
                page_no = prov.page_no
                image = my_images.get(page_no)

                if page_no not in number_data_to_draw:
                    number_data_to_draw[page_no] = []

                if image is None or prev_page is None or page_no != prev_page:
                    # new page begins
                    prev_page = page_no
                    x0 = y0 = None

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

                if x0 is None and y0 is None:
                    # is_root= True
                    x0 = (ro_bbox.l + ro_bbox.r) / 2.0
                    y0 = (ro_bbox.b + ro_bbox.t) / 2.0

                    number_data_to_draw[page_no].append(
                        _NumberDrawingData(
                            xy=(x0, y0),
                            text=f"{i}",
                        )
                    )
                    i += 1

                else:
                    # is_root = False
                    assert x0 is not None
                    assert y0 is not None

                    x1 = (ro_bbox.l + ro_bbox.r) / 2.0
                    y1 = (ro_bbox.b + ro_bbox.t) / 2.0

                    draw = self._draw_arrow(
                        draw=draw,
                        arrow_coords=(x0, y0, x1, y1),
                        line_width=2,
                        color="red",
                    )

                    x0, y0 = x1, y1

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
