"""Picture-specific metadata models."""

from typing import Optional

from pydantic import BaseModel, Field

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


class ChartAxis(BaseModel):
    """One axis of a chart picture (ChartToDocling DocLang extension).

    Mirrors the canonical ChartDoclang axis: which spatial direction it measures
    (``role`` = x / y / y2 / y3 / r / theta / z), its ``label`` and ``unit``, the
    ``scale`` (linear / log / categorical / time), and the tick ``categories`` of
    a categorical axis.
    """

    role: str
    label: Optional[str] = None
    unit: Optional[str] = None
    scale: Optional[str] = None
    categories: Optional[list[str]] = None


class ChartSeries(BaseModel):
    """One legend entry of a chart picture: a name bound to a visual encoding.

    Carries no data values (those live in ``tabular_chart``). ``color`` is a
    coarse human colour NAME (the exact hex stays in the upstream data, since a
    name is what is perceivable from the rendered image).
    """

    name: Optional[str] = None
    color: Optional[str] = None
    marker: Optional[str] = None
    line_style: Optional[str] = None
    mark_type: Optional[str] = None
    axis_ref: Optional[str] = None


class ChartColorLegend(BaseModel):
    """A continuous colour legend: colour encodes a scalar variable rather than
    naming discrete series (heatmap colorbar, colour-by-value scatter/bubble,
    continuously-coloured treemap). ``encodes`` names the variable; ``levels`` are
    the discrete swatch labels the legend literally shows, else ``value_range`` is
    the gradient's [lo, hi] with ``range_colors`` the [lo, hi] endpoint colours.
    """

    encodes: Optional[str] = None
    levels: Optional[list[str]] = None
    value_range: Optional[list[str]] = None
    range_colors: Optional[list[str]] = None


class PictureMeta(FloatingMeta):
    """Metadata model for pictures."""

    classification: Optional[PictureClassificationMetaField] = None
    molecule: Optional[MoleculeMetaField] = None
    tabular_chart: Optional[TabularChartMetaField] = None
    code: Optional[CodeMetaField] = None
    chart_axes: Optional[list[ChartAxis]] = None
    chart_series: Optional[list[ChartSeries]] = None
    chart_color_legend: Optional[ChartColorLegend] = None
