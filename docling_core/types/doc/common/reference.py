"""Reference, image-reference and provenance models."""

import base64
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional, Union
from urllib.parse import unquote

import numpy as np
from PIL import Image as PILImage
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Self

from docling_core.types.base import _JSON_POINTER_REGEX
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.common.scalars import CharSpan
from docling_core.types.doc.utils import _ensure_within_size_limit
from docling_core.utils.settings import settings

try:
    import cv2

    CV2_INSTALLED = True
except ImportError:
    CV2_INSTALLED = False

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class RefItem(BaseModel):
    """RefItem."""

    cref: str = Field(alias="$ref", pattern=_JSON_POINTER_REGEX)

    # This method makes RefItem compatible with DocItem
    def get_ref(self):
        """get_ref."""
        return self

    model_config = ConfigDict(
        populate_by_name=True,
    )

    def _split_ref_to_path(self):
        """Get the path of the reference."""
        return self.cref.split("/")

    def resolve(self, doc: "DoclingDocument"):
        """Resolve the path in the document."""
        path_components = self.cref.split("/")
        if (num_comps := len(path_components)) == 3:
            _, path, index_str = path_components
            index = int(index_str)
            obj = doc.__getattribute__(path)[index]
        elif num_comps == 2:
            _, path = path_components
            obj = doc.__getattribute__(path)
        else:
            raise RuntimeError(f"Unsupported number of path components: {num_comps}")
        return obj

    def _update_with_lookup(
        self,
        lookup: dict[str, dict[int, int]],
    ) -> None:
        path = self._split_ref_to_path()
        if len(path) == 3 and (item_label := path[1]) in lookup:
            item_index = int(path[2])
            # Count how many items have been deleted in front of you
            delta = sum(val if item_index >= key else 0 for key, val in lookup[item_label].items())
            new_index = item_index + delta
            self.cref = f"#/{item_label}/{new_index}"


class FineRef(RefItem):
    """Fine-granular reference item that can capture span range info."""

    range: Optional[tuple[int, int]] = None  # start_inclusive, end_exclusive


class ImageRef(BaseModel):
    """ImageRef."""

    mimetype: str
    dpi: int
    size: Size
    uri: Union[AnyUrl, Path] = Field(union_mode="left_to_right")
    _pil: Optional[PILImage.Image] = None

    @property
    def pil_image(self) -> Optional[PILImage.Image]:
        """Return the PIL Image."""
        if self._pil is not None:
            return self._pil

        if isinstance(self.uri, AnyUrl):
            if self.uri.scheme == "file":
                if not settings.allow_image_file_uri:
                    raise ValueError("file:// URI scheme is not enabled.")
                file_path = Path(unquote(str(self.uri.path)))
                _ensure_within_size_limit(
                    file_path,
                    max_size=settings.max_image_decoded_size,
                    label="Image file",
                )
                self._pil = PILImage.open(file_path)
            elif self.uri.scheme == "data":
                encoded_img = str(self.uri).split(",")[1]
                decoded_img = base64.b64decode(encoded_img)

                if len(decoded_img) > settings.max_image_decoded_size:
                    raise ValueError(f"Decoded image exceeds size limit of {settings.max_image_decoded_size} bytes.")

                self._pil = PILImage.open(BytesIO(decoded_img))
            # else: Handle http request or other protocols...
        elif isinstance(self.uri, Path):
            _ensure_within_size_limit(
                self.uri,
                max_size=settings.max_image_decoded_size,
                label="Image file",
            )
            self._pil = PILImage.open(self.uri)

        return self._pil

    @field_validator("mimetype")
    @classmethod
    def validate_mimetype(cls, v):
        """validate_mimetype."""
        # Check if the provided MIME type is valid using mimetypes module
        if v not in mimetypes.types_map.values():
            raise ValueError(f"'{v}' is not a valid MIME type")
        return v

    @staticmethod
    def _to_img_str_cv2(image: PILImage.Image) -> str:
        arr = np.ascontiguousarray(np.asarray(image))

        if image.mode == "RGB":
            encoded = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif image.mode == "RGBA":
            encoded = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        elif image.mode == "L":
            encoded = arr
        else:
            return ImageRef._to_img_str_pil(image)

        ok, buffered = cv2.imencode(".png", encoded)
        if not ok:
            return ImageRef._to_img_str_pil(image)

        return base64.b64encode(buffered.tobytes()).decode("utf-8")

    @staticmethod
    def _to_img_str_pil(image: PILImage.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    @classmethod
    def from_pil(cls, image: PILImage.Image, dpi: int) -> Self:
        """Construct ImageRef from a PIL Image."""
        if CV2_INSTALLED:
            img_str = cls._to_img_str_cv2(image)
        else:
            img_str = cls._to_img_str_pil(image)
        img_uri = f"data:image/png;base64,{img_str}"
        return cls(
            mimetype="image/png",
            dpi=dpi,
            size=Size(width=image.width, height=image.height),
            uri=img_uri,
            _pil=image,
        )


class ProvenanceItem(BaseModel):
    """Provenance information for elements extracted from a textual document.

    A `ProvenanceItem` object acts as a lightweight pointer back into the original
    document for an extracted element. It applies to documents with an explicit
    or implicit layout, such as PDF, HTML, docx, or pptx.
    """

    page_no: Annotated[int, Field(description="Page number")]
    bbox: Annotated[BoundingBox, Field(description="Bounding box")]
    charspan: CharSpan


class PageItem(BaseModel):
    """PageItem."""

    # A page carries separate root items for furniture and body,
    # only referencing items on the page
    size: Size
    image: Optional[ImageRef] = None
    page_no: int
