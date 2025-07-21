"""Define classes for layout visualization."""

from copy import deepcopy
from typing import Literal, Optional, Union

from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont
from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.visualizer.base import BaseVisualizer
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.document import ContentLayer, DocItem, DoclingDocument
from docling_core.types.doc.page import BoundingRectangle, TextCell


class _TLBoundingRectangle(BoundingRectangle):
    coord_origin: Literal[CoordOrigin.TOPLEFT] = CoordOrigin.TOPLEFT


class _TLTextCell(TextCell):
    rect: _TLBoundingRectangle


class _TLCluster(BaseModel):
    id: int
    label: DocItemLabel
    brec: _TLBoundingRectangle
    confidence: float = 1.0
    cells: list[_TLTextCell] = []
    children: list["_TLCluster"] = []  # Add child cluster support


class LayoutVisualizer(BaseVisualizer):
    """Layout visualizer."""

    class Params(BaseModel):
        """Layout visualization parameters."""

        show_label: bool = True
        content_layers: set[ContentLayer] = {cl for cl in ContentLayer}

    base_visualizer: Optional[BaseVisualizer] = None
    params: Params = Params()

    def _draw_clusters(
        self, image: Image, clusters: list[_TLCluster], scale_x: float, scale_y: float
    ) -> None:
        """Draw clusters on an image."""
        draw = ImageDraw.Draw(image, "RGBA")
        # Create a smaller font for the labels
        font: Union[ImageFont.ImageFont, FreeTypeFont]
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            # Fallback to default font if arial is not available
            font = ImageFont.load_default()
        for c_tl in clusters:
            all_clusters = [c_tl, *c_tl.children]
            for c in all_clusters:
                # Draw cells first (underneath)
                cell_color = (0, 0, 0, 40)  # Transparent black for cells
                for tc in c.cells:
                    cx0, cy0, cx1, cy1 = tc.rect.to_bounding_box().as_tuple()
                    cx0 *= scale_x
                    cx1 *= scale_x
                    cy0 *= scale_y
                    cy1 *= scale_y

                    draw.rectangle(
                        [(cx0, cy0), (cx1, cy1)],
                        outline=None,
                        fill=cell_color,
                    )
                # Draw cluster rectangle
                x0, y0, x1, y1 = c.brec.to_bounding_box().as_tuple()
                x0 *= scale_x
                x1 *= scale_x
                y0 *= scale_y
                y1 *= scale_y

                cluster_fill_color = (*list(DocItemLabel.get_color(c.label)), 70)
                cluster_outline_color = (
                    *list(DocItemLabel.get_color(c.label)),
                    255,
                )
                draw.rectangle(
                    [(x0, y0), (x1, y1)],
                    outline=cluster_outline_color,
                    fill=cluster_fill_color,
                )

                if self.params.show_label:
                    # Add label name and confidence
                    label_text = f"{c.label.name} ({c.confidence:.2f})"
                    # Create semi-transparent background for text
                    text_bbox = draw.textbbox((x0, y0), label_text, font=font)
                    text_bg_padding = 2
                    draw.rectangle(
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
                        fill=(255, 255, 255, 180),  # Semi-transparent white
                    )
                    # Draw text
                    draw.text(
                        (x0, y0),
                        label_text,
                        fill=(0, 0, 0, 255),  # Solid black
                        font=font,
                    )

    def _draw_doc_layout(
        self,
        doc: DoclingDocument,
        images: Optional[dict[Optional[int], Image]] = None,
        included_content_layers: Optional[set[ContentLayer]] = None,
    ):
        """Draw the document clusters and optionaly the reading order."""
        clusters = []
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

        prev_image = None
        prev_page_nr = None
        for idx, (elem, _) in enumerate(
            doc.iterate_items(
                included_content_layers=included_content_layers, traverse_pictures=True
            )
        ):
            if not isinstance(elem, DocItem):
                continue
            if len(elem.prov) == 0:
                continue  # Skip elements without provenances

            for prov in elem.prov:
                page_nr = prov.page_no

                if page_nr in my_images:
                    image = my_images[page_nr]
                else:
                    raise RuntimeError(f"Cannot visualize page-image for {page_nr}")

                if prev_page_nr is None or page_nr != prev_page_nr:  # changing page
                    # dump previous drawing
                    if prev_page_nr is not None and prev_image and clusters:
                        self._draw_clusters(
                            image=prev_image,
                            clusters=clusters,
                            scale_x=prev_image.width
                            / doc.pages[prev_page_nr].size.width,
                            scale_y=prev_image.height
                            / doc.pages[prev_page_nr].size.height,
                        )
                        clusters = []

                tlo_bbox = prov.bbox.to_top_left_origin(
                    page_height=doc.pages[prov.page_no].size.height
                )
                cluster = _TLCluster(
                    id=idx,
                    label=elem.label,
                    brec=_TLBoundingRectangle.from_bounding_box(bbox=tlo_bbox),
                    cells=[],
                )
                clusters.append(cluster)

                prev_page_nr = page_nr
                prev_image = image

        # complete last drawing
        if prev_page_nr is not None and prev_image and clusters:
            self._draw_clusters(
                image=prev_image,
                clusters=clusters,
                scale_x=prev_image.width / doc.pages[prev_page_nr].size.width,
                scale_y=prev_image.height / doc.pages[prev_page_nr].size.height,
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
        return self._draw_doc_layout(
            doc=doc,
            images=base_images,
        )
