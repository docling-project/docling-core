"""Picture item and its annotation-data union."""

import base64
import hashlib
import logging
import typing
import warnings
from collections.abc import Sequence
from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Optional, Union

from PIL import Image as PILImage
from pydantic import Field, model_validator
from typing_extensions import Self, deprecated

from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.common.annotations import BaseAnnotation, DescriptionAnnotation, MiscAnnotation
from docling_core.types.doc.common.meta import (
    DescriptionMetaField,
    MetaUtils,
    MoleculeMetaField,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    TabularChartMetaField,
)
from docling_core.types.doc.items.node import FloatingItem
from docling_core.types.doc.items.picture.charts import (
    PictureBarChartData,
    PictureLineChartData,
    PicturePieChartData,
    PictureScatterChartData,
    PictureStackedBarChartData,
    PictureTabularChartData,
)
from docling_core.types.doc.items.picture.classification import PictureClassificationData
from docling_core.types.doc.items.picture.molecule import PictureMoleculeData
from docling_core.types.doc.labels import DocItemLabel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

_logger = logging.getLogger(__name__)


PictureDataType = Annotated[
    Union[
        DescriptionAnnotation,
        MiscAnnotation,
        PictureClassificationData,
        PictureMoleculeData,
        PictureTabularChartData,
        PictureLineChartData,
        PictureBarChartData,
        PictureStackedBarChartData,
        PicturePieChartData,
        PictureScatterChartData,
    ],
    Field(discriminator="kind"),
]


class PictureItem(FloatingItem):
    """PictureItem."""

    label: typing.Literal[DocItemLabel.PICTURE, DocItemLabel.CHART] = DocItemLabel.PICTURE

    meta: Optional[PictureMeta] = None
    annotations: Annotated[
        list[PictureDataType],
        deprecated("Field `annotations` is deprecated; use `meta` instead."),
    ] = []

    @model_validator(mode="after")
    def _migrate_annotations_to_meta(self) -> Self:
        """Migrate the `annotations` field to `meta`."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)

            if self.annotations:
                _logger.info(
                    "Migrating deprecated `annotations` to `meta`; this will be removed in the future. "
                    "Note that only the first available instance of each annotation type will be migrated."
                )
                for ann in self.annotations:
                    # migrate annotations to meta

                    # ensure meta field is present
                    if self.meta is None:
                        self.meta = PictureMeta()

                    if isinstance(ann, PictureClassificationData) and self.meta.classification is None:
                        self.meta.classification = PictureClassificationMetaField(
                            predictions=[
                                PictureClassificationPrediction(
                                    class_name=pred.class_name,
                                    confidence=pred.confidence,
                                    created_by=ann.provenance,
                                )
                                for pred in ann.predicted_classes
                            ],
                        )
                    elif isinstance(ann, DescriptionAnnotation) and self.meta.description is None:
                        self.meta.description = DescriptionMetaField(
                            text=ann.text,
                            created_by=ann.provenance,
                        )
                    elif isinstance(ann, PictureMoleculeData) and self.meta.molecule is None:
                        self.meta.molecule = MoleculeMetaField(
                            smi=ann.smi,
                            confidence=ann.confidence,
                            created_by=ann.provenance,
                            **{
                                MetaUtils._create_migrated_meta_field_name(name="segmentation"): ann.segmentation,
                                MetaUtils._create_migrated_meta_field_name(name="class_name"): ann.class_name,
                            },
                        )
                    elif isinstance(ann, PictureTabularChartData) and self.meta.tabular_chart is None:
                        self.meta.tabular_chart = TabularChartMetaField(
                            title=ann.title,
                            chart_data=ann.chart_data,
                        )
                    elif not isinstance(
                        ann,
                        PictureClassificationData
                        | DescriptionAnnotation
                        | PictureMoleculeData
                        | PictureTabularChartData,
                    ) and not hasattr(
                        self.meta,
                        MetaUtils.create_meta_field_name(
                            namespace=MetaUtils._META_FIELD_LEGACY_NAMESPACE,
                            name=ann.kind,
                        ),
                    ):
                        self.meta.set_custom_field(
                            namespace=MetaUtils._META_FIELD_LEGACY_NAMESPACE,
                            name=ann.kind,
                            value=(ann.content if isinstance(ann, MiscAnnotation) else ann.model_dump(mode="json")),
                        )

            return self

    # Convert the image to Base64
    def _image_to_base64(self, pil_image, format="PNG"):
        """Base64 representation of the image."""
        buffered = BytesIO()
        pil_image.save(buffered, format=format)  # Save the image to the byte stream
        img_bytes = buffered.getvalue()  # Get the byte data
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")  # Encode to Base64 and decode to string
        return img_base64

    @staticmethod
    def _image_to_hexhash(img: Optional[PILImage.Image]) -> Optional[str]:
        """Hexash from the image."""
        if img is not None:
            # Convert the image to raw bytes
            image_bytes = img.tobytes()

            # Create a hash object (e.g., SHA-256)
            hasher = hashlib.sha256(usedforsecurity=False)

            # Feed the image bytes into the hash object
            hasher.update(image_bytes)

            # Get the hexadecimal representation of the hash
            return hasher.hexdigest()

        return None

    def export_to_markdown(
        self,
        doc: "DoclingDocument",
        add_caption: bool = True,  # deprecated
        image_mode: ImageRefMode = ImageRefMode.EMBEDDED,
        image_placeholder: str = "<!-- image -->",
    ) -> str:
        """Export picture to Markdown format."""
        from docling_core.transforms.serializer.markdown import (
            MarkdownDocSerializer,
            MarkdownParams,
        )

        if not add_caption:
            _logger.warning(
                "Argument `add_caption` is deprecated and will be ignored.",
            )

        serializer = MarkdownDocSerializer(
            doc=doc,
            params=MarkdownParams(
                image_mode=image_mode,
                image_placeholder=image_placeholder,
            ),
        )
        text = serializer.serialize(item=self).text
        return text

    def export_to_html(
        self,
        doc: "DoclingDocument",
        add_caption: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
    ) -> str:
        """Export picture to HTML format."""
        from docling_core.transforms.serializer.html import (
            HTMLDocSerializer,
            HTMLParams,
        )

        serializer = HTMLDocSerializer(
            doc=doc,
            params=HTMLParams(
                image_mode=image_mode,
            ),
        )
        text = serializer.serialize(item=self).text
        return text

    @deprecated("Use export_to_doctags() instead.")
    def export_to_document_tokens(self, *args, **kwargs):
        r"""Export to DocTags format."""
        return self.export_to_doctags(*args, **kwargs)

    def export_to_doctags(
        self,
        doc: "DoclingDocument",
        new_line: str = "",  # deprecated
        xsize: int = 500,
        ysize: int = 500,
        add_location: bool = True,
        add_caption: bool = True,
        add_content: bool = True,  # not used at the moment
    ):
        r"""Export picture to document tokens format.

        :param doc: "DoclingDocument":
        :param new_line: str (Default value = "")  Deprecated
        :param xsize: int:  (Default value = 500)
        :param ysize: int:  (Default value = 500)
        :param add_location: bool:  (Default value = True)
        :param add_caption: bool:  (Default value = True)
        :param add_content: bool:  (Default value = True)
        :param # not used at the moment

        """
        from docling_core.transforms.serializer.doctags import (
            DocTagsDocSerializer,
            DocTagsParams,
        )

        serializer = DocTagsDocSerializer(
            doc=doc,
            params=DocTagsParams(
                xsize=xsize,
                ysize=ysize,
                add_location=add_location,
                add_content=add_content,
                add_caption=add_caption,
            ),
        )
        text = serializer.serialize(item=self).text
        return text

    def get_annotations(self) -> Sequence[BaseAnnotation]:
        """Get the annotations of this PictureItem."""
        return self.annotations


BasePictureData = BaseAnnotation
PictureDescriptionData = DescriptionAnnotation
PictureMiscData = MiscAnnotation
