#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Models for annotations and predictions in CCS."""
from typing import Any

from pydantic import BaseModel

from docling_core.types.legacy_doc.base import BoundingBox

AnnotationReport = Any  # TODO


class Cell(BaseModel):
    """Cell."""

    id: int
    rawcell_id: int
    label: str


class Cluster(BaseModel):
    """Cluster."""

    model: str
    type: str
    bbox: BoundingBox
    cell_ids: list[int]
    merged: bool
    id: int


class Table(BaseModel):
    """Table."""

    cell_id: int
    label: str
    rows: list[int]
    cols: list[int]


class Info(BaseModel):
    """Info."""

    display_name: str
    model_name: str
    model_class: str
    model_version: str
    model_id: str


class Source(BaseModel):
    """Source."""

    type: str
    timestamp: float
    info: Info


class AnnotPredItem(BaseModel):
    """Annotation or prediction item."""

    cells: list[Cell]
    clusters: list[Cluster]
    tables: list[Table]
    source: Source


class Annotation(BaseModel):
    """Annotations."""

    annotations: list[AnnotPredItem]
    predictions: list[AnnotPredItem]
    reports: list[AnnotationReport]
