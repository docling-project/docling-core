"""Key‑value visualizer overlaying key/value cells and their links on page images.

This module complements :py:class:`layout_visualizer.LayoutVisualizer` by drawing
*key* and *value* cells plus the directed links between them.  It can be stacked
on top of any other :py:class:`BaseVisualizer` – e.g. first draw the general
layout, then add the key‑value layer.
"""

from copy import deepcopy
from typing import Optional, Union

from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont
from pydantic import BaseModel
from typing_extensions import override

from docling_core.transforms.visualizer.base import BaseVisualizer
from docling_core.types.doc.document import ContentLayer, DoclingDocument
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel

# ---------------------------------------------------------------------------
# Helper functions / constants
# ---------------------------------------------------------------------------

# Semi‑transparent RGBA colours for key / value cells and their connecting link
_KEY_FILL = (0, 170, 0, 70)  # greenish
_VALUE_FILL = (0, 0, 200, 70)  # bluish
_LINK_COLOUR = (255, 0, 0, 255)  # red line (solid)

_LABEL_TXT_COLOUR = (0, 0, 0, 255)
_LABEL_BG_COLOUR = (255, 255, 255, 180)  # semi‑transparent white


class KeyValueVisualizer(BaseVisualizer):
    """Draw key/value graphs stored in :py:attr:`DoclingDocument.key_value_items`."""

    class Params(BaseModel):
        """Parameters for KeyValueVisualizer controlling label and cell id display, and content layers to visualize."""

        show_label: bool = True  # draw cell text close to bbox
        show_cell_id: bool = False  # annotate each rectangle with its cell_id
        content_layers: set[ContentLayer] = {cl for cl in ContentLayer}

    base_visualizer: Optional[BaseVisualizer] = None
    params: Params = Params()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _cell_fill(self, label: GraphCellLabel) -> tuple[int, int, int, int]:
        """Return RGBA fill colour depending on *label*."""
        return _KEY_FILL if label == GraphCellLabel.KEY else _VALUE_FILL

    def _draw_key_value_layer(
        self,
        *,
        image: Image,
        doc: DoclingDocument,
        page_no: int,
        scale_x: float,
        scale_y: float,
    ) -> None:
        """Draw every key‑value graph that has cells on *page_no* onto *image*."""
        draw = ImageDraw.Draw(image, "RGBA")
        # Choose a small truetype font if available, otherwise default bitmap font
        font: Union[ImageFont.ImageFont, FreeTypeFont]
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            font = ImageFont.load_default()

        for kv_item in doc.key_value_items:
            cell_dict = {cell.cell_id: cell for cell in kv_item.graph.cells}

            # ------------------------------------------------------------------
            # First draw cells (rectangles + optional labels)
            # ------------------------------------------------------------------
            for cell in cell_dict.values():
                if cell.prov is None or cell.prov.page_no != page_no:
                    continue  # skip cells not on this page or without bbox

                tl_bbox = cell.prov.bbox.to_top_left_origin(
                    page_height=doc.pages[page_no].size.height
                )
                x0, y0, x1, y1 = tl_bbox.as_tuple()
                x0 *= scale_x
                x1 *= scale_x
                y0 *= scale_y
                y1 *= scale_y
                fill_rgba = self._cell_fill(cell.label)

                draw.rectangle(
                    [(x0, y0), (x1, y1)],
                    outline=fill_rgba[:-1] + (255,),
                    fill=fill_rgba,
                )

                if self.params.show_label:
                    txt_parts = []
                    if self.params.show_cell_id:
                        txt_parts.append(str(cell.cell_id))
                    txt_parts.append(cell.text)
                    label_text = " | ".join(txt_parts)

                    tbx = draw.textbbox((x0, y0), label_text, font=font)
                    pad = 2
                    draw.rectangle(
                        [(tbx[0] - pad, tbx[1] - pad), (tbx[2] + pad, tbx[3] + pad)],
                        fill=_LABEL_BG_COLOUR,
                    )
                    draw.text((x0, y0), label_text, font=font, fill=_LABEL_TXT_COLOUR)

            # ------------------------------------------------------------------
            # Then draw links (after rectangles so they appear on top)
            # ------------------------------------------------------------------
            for link in kv_item.graph.links:
                if link.label != GraphLinkLabel.TO_VALUE:
                    # Future‑proof: ignore other link types silently
                    continue

                src_cell = cell_dict.get(link.source_cell_id)
                tgt_cell = cell_dict.get(link.target_cell_id)
                if src_cell is None or tgt_cell is None:
                    continue
                if (
                    src_cell.prov is None
                    or tgt_cell.prov is None
                    or src_cell.prov.page_no != page_no
                    or tgt_cell.prov.page_no != page_no
                ):
                    continue  # only draw if both ends are on this page

                def _centre(bbox):
                    tl = bbox.to_top_left_origin(
                        page_height=doc.pages[page_no].size.height
                    )
                    l, t, r, b = tl.as_tuple()
                    return ((l + r) / 2 * scale_x, (t + b) / 2 * scale_y)

                src_xy = _centre(src_cell.prov.bbox)
                tgt_xy = _centre(tgt_cell.prov.bbox)

                draw.line([src_xy, tgt_xy], fill=_LINK_COLOUR, width=2)

                # draw a small arrow‑head by rendering a short orthogonal line
                # segment; exact geometry is not critical for visual inspection
                arrow_len = 6
                dx = tgt_xy[0] - src_xy[0]
                dy = tgt_xy[1] - src_xy[1]
                length = (dx**2 + dy**2) ** 0.5 or 1.0
                ux, uy = dx / length, dy / length
                # perpendicular vector
                px, py = -uy, ux
                # two points forming the arrow head triangle base
                head_base_left = (
                    tgt_xy[0] - ux * arrow_len - px * arrow_len / 2,
                    tgt_xy[1] - uy * arrow_len - py * arrow_len / 2,
                )
                head_base_right = (
                    tgt_xy[0] - ux * arrow_len + px * arrow_len / 2,
                    tgt_xy[1] - uy * arrow_len + py * arrow_len / 2,
                )
                draw.polygon(
                    [tgt_xy, head_base_left, head_base_right], fill=_LINK_COLOUR
                )

    # ---------------------------------------------------------------------
    # Public API – BaseVisualizer implementation
    # ---------------------------------------------------------------------

    @override
    def get_visualization(
        self,
        *,
        doc: DoclingDocument,
        included_content_layers: Optional[set[ContentLayer]] = None,
        **kwargs,
    ) -> dict[Optional[int], Image]:
        """Return page‑wise images with key/value overlay (incl. base layer)."""
        base_images = (
            self.base_visualizer.get_visualization(
                doc=doc, included_content_layers=included_content_layers, **kwargs
            )
            if self.base_visualizer
            else None
        )

        if included_content_layers is None:
            included_content_layers = {cl for cl in ContentLayer}

        images: dict[Optional[int], Image] = {}

        # Ensure we have page images to draw on
        for page_nr, page in doc.pages.items():
            base_img = (base_images or {}).get(page_nr)
            if base_img is None:
                if page.image is None or (pil_img := page.image.pil_image) is None:
                    raise RuntimeError("Cannot visualize document without page images")
                base_img = deepcopy(pil_img)
            images[page_nr] = base_img

        # Overlay key‑value content
        for page_nr, img in images.items():  # type: ignore
            assert isinstance(page_nr, int)
            scale_x = img.width / doc.pages[page_nr].size.width
            scale_y = img.height / doc.pages[page_nr].size.height
            self._draw_key_value_layer(
                image=img,
                doc=doc,
                page_no=page_nr,
                scale_x=scale_x,
                scale_y=scale_y,
            )

        return images
