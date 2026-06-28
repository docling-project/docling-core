"""Picture classification annotation models."""

from typing import Literal

from pydantic import BaseModel, FieldSerializationInfo, field_serializer

from docling_core.types.doc.base import PydanticSerCtxKey, round_pydantic_float
from docling_core.types.doc.common.annotations import BaseAnnotation


class PictureClassificationClass(BaseModel):
    """PictureClassificationData."""

    class_name: str
    confidence: float

    @field_serializer("confidence")
    def _serialize(self, value: float, info: FieldSerializationInfo) -> float:
        return round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)


class PictureClassificationData(BaseAnnotation):
    """PictureClassificationData."""

    kind: Literal["classification"] = "classification"
    provenance: str
    predicted_classes: list[PictureClassificationClass]
