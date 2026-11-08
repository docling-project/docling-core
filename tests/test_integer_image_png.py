"""Integer image export must retain the PNG pixel values used before Pillow 13."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from docling_core.types.doc import DoclingDocument, ImageRef, ImageRefMode


def test_integer_image_embedded_and_referenced_png(tmp_path: Path) -> None:
    image = Image.new("I", (6, 1))
    image.putdata([-1, 0, 255, 256, 65535, 65536])
    reference = ImageRef.from_pil(image, dpi=72)
    embedded = BytesIO(base64.b64decode(str(reference.uri).split(",", 1)[1]))
    doc = DoclingDocument(name="Integer picture")
    doc.add_picture(image=reference)
    doc.save_as_markdown(tmp_path / "document.md", image_mode=ImageRefMode.REFERENCED)
    [referenced] = tmp_path.rglob("*.png")

    for source in (embedded, referenced):
        with Image.open(source) as decoded:
            assert [decoded.getpixel((x, 0)) for x in range(6)] == [0, 0, 255, 256, 65535, 65535]
