"""Picture molecule annotation model."""

from typing import Literal

from pydantic import FieldSerializationInfo, field_serializer

from docling_core.types.doc.base import PydanticSerCtxKey, round_pydantic_float
from docling_core.types.doc.common.annotations import BaseAnnotation


class PictureMoleculeData(BaseAnnotation):
    """PictureMoleculeData."""

    kind: Literal["molecule_data"] = "molecule_data"
    smi: str
    confidence: float
    class_name: str
    segmentation: list[tuple[float, float]]
    provenance: str

    @field_serializer("confidence")
    def _serialize(self, value: float, info: FieldSerializationInfo) -> float:
        return round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)
