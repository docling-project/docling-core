"""Picture-specific metadata models."""

from typing import Optional

from pydantic import Field

from docling_core.types.doc.common.meta import (
    BasePrediction,
    CodeMetaField,
    FloatingMeta,
    _ExtraAllowingModel,
)
from docling_core.types.doc.items.table.table_data import TableData


class PictureClassificationPrediction(BasePrediction):
    """Picture classification instance."""

    class_name: str


class PictureClassificationMetaField(_ExtraAllowingModel):
    """Picture classification metadata field."""

    predictions: list[PictureClassificationPrediction] = Field(default_factory=list, min_length=1)

    def get_main_prediction(self) -> PictureClassificationPrediction:
        """Get prediction with highest confidence (if confidence not available, first is used by convention)."""
        max_conf_pos: Optional[int] = None
        max_conf: Optional[float] = None
        for i, pred in enumerate(self.predictions):
            if pred.confidence is not None and (max_conf is None or pred.confidence > max_conf):
                max_conf_pos = i
                max_conf = pred.confidence
        return self.predictions[max_conf_pos if max_conf_pos is not None else 0]


class MoleculeMetaField(BasePrediction):
    """Molecule metadata field."""

    smi: str = Field(description="The SMILES representation of the molecule.")


class TabularChartMetaField(BasePrediction):
    """Tabular chart metadata field."""

    title: Optional[str] = None
    chart_data: TableData


class PictureMeta(FloatingMeta):
    """Metadata model for pictures."""

    classification: Optional[PictureClassificationMetaField] = None
    molecule: Optional[MoleculeMetaField] = None
    tabular_chart: Optional[TabularChartMetaField] = None
    code: Optional[CodeMetaField] = None
