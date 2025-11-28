import importlib.metadata
import json
import logging
import os
from io import BytesIO
from typing import Dict, Optional, Set, Tuple

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption
from docling.datamodel.base_models import ConversionStatus
from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import (
    EvaluationModality,
    PredictionFormats,
    PredictionProviderType,
)
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import (
    does_intersection_area_exceed_threshold,
    from_pil_to_base64uri,
)

# from docling_core.types import DoclingDocument
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.page import (
    BoundingRectangle,
    PageGeometry,
    SegmentedPage,
    TextCell,
)
from docling_core.types.io import DocumentStream

# from docling_core.types.doc.labels import DocItemLabel

_log = logging.getLogger(__name__)


class AzureDocIntelligencePredictionProvider(BasePredictionProvider):
    """Provider that calls the Microsoft Azure Document Intelligence API for predicting the tables in document."""

    prediction_provider_type: PredictionProviderType = PredictionProviderType.AZURE

    prediction_modalities = [EvaluationModality.TABLE_STRUCTURE]

    def __init__(
        self,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(
            do_visualization=do_visualization,
            ignore_missing_predictions=ignore_missing_predictions,
            true_labels=true_labels,
            pred_labels=pred_labels,
        )

        # TODO - Need a temp directory to save Azure outputs
        # Validate the required library
        try:
            from azure.core.credentials import AzureKeyCredential  # type: ignore
        except ImportError:
            raise ImportError(
                "azure-ai-documentintelligence library is not installed.."
            )

        # Validate the required endpoints to call the API
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not endpoint or not key:
            raise ValueError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY must be set in environment variables."
            )

        self.doc_intelligence_client = DocumentIntelligenceClient(
            endpoint, AzureKeyCredential(key)
        )

    def extract_bbox_from_polygon(self, polygon):
        """Helper function to extract bbox coordinates from polygon data."""
        if not polygon or not isinstance(polygon, list):
            return {"l": 0, "t": 0, "r": 0, "b": 0}

        # Handle flat array format: [x1, y1, x2, y2, x3, y3, x4, y4]
        if len(polygon) >= 8 and all(isinstance(p, (int, float)) for p in polygon):
            return {"l": polygon[0], "t": polygon[1], "r": polygon[4], "b": polygon[5]}
        # Handle array of point objects: [{x, y}, {x, y}, ...]
        elif len(polygon) >= 4 and all(
            isinstance(p, dict) and "x" in p and "y" in p for p in polygon
        ):
            return {
                "l": polygon[0]["x"],
                "t": polygon[0]["y"],
                "r": polygon[2]["x"],
                "b": polygon[2]["y"],
            }
        else:
            return {"l": 0, "t": 0, "r": 0, "b": 0}

    def convert_azure_output_to_docling(
        self, analyze_result, record: DatasetRecord
    ) -> Tuple[DoclingDocument, Dict[int, SegmentedPage]]:
        """Converts Azure Document Intelligence output to DoclingDocument format."""
        doc = DoclingDocument(name=record.doc_id)
        segmented_pages: Dict[int, SegmentedPage] = {}

        for page in analyze_result.get("pages", []):
            page_no = page.get("pageNumber", 1)

            page_width = page.get("width")
            page_height = page.get("height")

            im = record.ground_truth_page_images[page_no - 1]

            # Add page with image
            image_ref = ImageRef(
                mimetype=f"image/png",
                dpi=72,
                size=Size(width=float(im.width), height=float(im.height)),
                uri=from_pil_to_base64uri(im),
            )
            page_item = PageItem(
                page_no=page_no,
                size=Size(width=page_width, height=page_height),
                image=image_ref,
            )
            doc.pages[page_no] = page_item

            if page_no not in segmented_pages.keys():
                seg_page = SegmentedPage(
                    dimension=PageGeometry(
                        angle=0,
                        rect=BoundingRectangle.from_bounding_box(
                            BoundingBox(
                                l=0,
                                t=0,
                                r=page_item.size.width,
                                b=page_item.size.height,
                            )
                        ),
                    )
                )
                segmented_pages[page_no] = seg_page

            for word in page.get("words", []):
                polygon = word.get("polygon", None)
                text_content = word.get("content", None)

                if text_content is not None and polygon is not None:
                    bbox = self.extract_bbox_from_polygon(polygon)
                    bbox_obj = BoundingBox(
                        l=bbox["l"],
                        t=bbox["t"],
                        r=bbox["r"],
                        b=bbox["b"],
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    segmented_pages[page_no].word_cells.append(
                        TextCell(
                            rect=BoundingRectangle.from_bounding_box(bbox_obj),
                            text=text_content,
                            orig=text_content,
                            # Keeping from_ocr flag False since Azure output doesn't indicate whether the given word is programmatic or OCR
                            from_ocr=False,
                        )
                    )

        # Iterate over tables in the response and add to DoclingDocument
        self._add_tables(analyze_result, doc)

        # Iterate over figures and add them as pictures in DoclingDocument
        self._add_figures(analyze_result, doc)

        # Iterate over paragraphs in the response and add populate fields like section headings, header-footer based on "role" field
        self._handle_paragraphs_based_on_roles(analyze_result, doc)

        return doc, segmented_pages

    def _add_figures(self, analyze_result, doc):
        for figure in analyze_result.get("figures", []):
            bounding_regions = figure["boundingRegions"][0]
            page_no = bounding_regions["pageNumber"]
            polygon = bounding_regions.get("polygon", [])
            bbox = self.extract_bbox_from_polygon(polygon)

            bbox_obj = BoundingBox(
                l=bbox["l"],
                t=bbox["t"],
                r=bbox["r"],
                b=bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            prov = ProvenanceItem(page_no=page_no, bbox=bbox_obj, charspan=(0, 0))
            doc.add_picture(prov=prov)

            # Figures can also have footnotes in azure output
            footnotes = figure.get("footnotes", [])
            for footnote in footnotes:
                bounding_regions = footnote["boundingRegions"][0]
                page_no = bounding_regions["pageNumber"]
                polygon = bounding_regions.get("polygon", [])
                bbox = self.extract_bbox_from_polygon(polygon)

                text_content = footnote.get("content", "")

                bbox_obj = BoundingBox(
                    l=bbox["l"],
                    t=bbox["t"],
                    r=bbox["r"],
                    b=bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                prov = ProvenanceItem(
                    page_no=page_no, bbox=bbox_obj, charspan=(0, len(text_content))
                )
                doc.add_text(label=DocItemLabel.FOOTNOTE, text=text_content, prov=prov)

    def _handle_paragraphs_based_on_roles(self, analyze_result, doc):
        for paragraph in analyze_result.get("paragraphs", []):
            bounding_regions = paragraph["boundingRegions"][0]
            page_no = bounding_regions["pageNumber"]
            polygon = bounding_regions.get("polygon", [])
            bbox = self.extract_bbox_from_polygon(polygon)

            text_content = paragraph.get("content", "")

            bbox_obj = BoundingBox(
                l=bbox["l"],
                t=bbox["t"],
                r=bbox["r"],
                b=bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            # Filter any text that overlaps with tables so the content is not duplicated
            if any(
                does_intersection_area_exceed_threshold(
                    bbox_obj, table.prov[0].bbox, 0.6
                )
                for table in doc.tables
            ):
                continue

            # Filter any text that overlaps with figures so the content is not duplicated
            if any(
                does_intersection_area_exceed_threshold(
                    bbox_obj, picture.prov[0].bbox, 0.6
                )
                for picture in doc.pictures
            ):
                continue

            prov = ProvenanceItem(
                page_no=page_no, bbox=bbox_obj, charspan=(0, len(text_content))
            )

            role = paragraph.get("role", None)
            if role:
                if role == "sectionHeading":
                    doc.add_heading(text=text_content, prov=prov)
                elif role == "title":
                    doc.add_title(text=text_content, prov=prov)
                elif role == "footnote":
                    doc.add_text(
                        label=DocItemLabel.FOOTNOTE, text=text_content, prov=prov
                    )
                elif role == "pageHeader":
                    doc.add_text(
                        label=DocItemLabel.PAGE_HEADER, text=text_content, prov=prov
                    )
                elif role == "pageFooter":
                    doc.add_text(
                        label=DocItemLabel.PAGE_FOOTER, text=text_content, prov=prov
                    )
                elif role == "pageNumber":
                    doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)
            else:
                doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

    def _add_tables(self, analyze_result, doc):
        for table in analyze_result.get("tables", []):
            bounding_region = table["boundingRegions"][0]
            page_no = bounding_region.get("pageNumber", 1)
            row_count = table.get("rowCount", 0)
            col_count = table.get("columnCount", 0)

            table_polygon = bounding_region.get("polygon", [])
            table_bbox = self.extract_bbox_from_polygon(table_polygon)

            table_bbox_obj = BoundingBox(
                l=table_bbox["l"],
                t=table_bbox["t"],
                r=table_bbox["r"],
                b=table_bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            table_prov = ProvenanceItem(
                page_no=page_no, bbox=table_bbox_obj, charspan=(0, 0)
            )

            table_cells = []

            for cell in table.get("cells", []):
                cell_text = cell.get("content", "").strip()
                row_index = cell.get("rowIndex", 0)
                col_index = cell.get("columnIndex", 0)
                row_span = cell.get("rowSpan", 1)
                col_span = cell.get("colSpan", 1)
                kind = cell.get("kind", None)
                is_row_header = True if kind == "rowHeader" else False
                is_col_header = True if kind == "columnHeader" else False

                cell_polygon = cell.get("boundingRegions", [{}])[0].get("polygon", [])
                cell_bbox = self.extract_bbox_from_polygon(cell_polygon)

                table_cell = TableCell(
                    bbox=BoundingBox(
                        l=cell_bbox["l"],
                        t=cell_bbox["t"],
                        r=cell_bbox["r"],
                        b=cell_bbox["b"],
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + row_span,
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + col_span,
                    text=cell_text,
                    column_header=is_col_header,
                    row_header=is_row_header,
                    row_section=False,
                )

                table_cells.append(table_cell)

            table_data = TableData(
                table_cells=table_cells, num_rows=row_count, num_cols=col_count
            )

            doc.add_table(prov=table_prov, data=table_data, caption=None)

    @property
    def prediction_format(self) -> PredictionFormats:
        """Get the prediction format."""
        return PredictionFormats.JSON

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Generate a prediction from Azure Doc Intelligence.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction

        Raises:
            RuntimeError: If ground truth doc is not available or if mime type is unsupported
        """

        status = ConversionStatus.SUCCESS
        assert record.original is not None

        try:
            _log.info(f"Processing file '{record.doc_id}'..")
            if record.mime_type == "application/pdf":
                if not isinstance(record.original, DocumentStream):
                    raise RuntimeError(
                        "Original document must be a DocumentStream for PDF files"
                    )
                # Call the Azure API by passing in the image for prediction
                poller = self.doc_intelligence_client.begin_analyze_document(
                    "prebuilt-layout",
                    record.original.stream,
                    features=[],
                    output=[AnalyzeOutputOption.FIGURES],
                )
                result = poller.result().as_dict()
                result_json = json.dumps(result)
                _log.info(
                    f"Successfully processed [{record.doc_id}] using Azure API..!!"
                )

            elif record.mime_type in ["image/png", "image/jpeg", "image/jpg"]:
                # Call the Azure API by passing in the image for prediction
                buf = BytesIO()

                # TODO do this in a loop for all page images in the doc, not just the first.
                save_format = "PNG" if record.mime_type == "image/png" else "JPEG"
                record.ground_truth_page_images[0].save(buf, format=save_format)

                poller = self.doc_intelligence_client.begin_analyze_document(
                    "prebuilt-layout",
                    BytesIO(buf.getvalue()),
                    features=[],
                    output=[AnalyzeOutputOption.FIGURES],
                )
                result = poller.result().as_dict()
                result_json = json.dumps(result, default=str)
                _log.info(
                    f"Successfully processed [{record.doc_id}] using Azure API..!!"
                )
            else:
                raise RuntimeError(
                    f"Unsupported mime type: {record.mime_type}. AzureDocIntelligencePredictionProvider supports 'application/pdf', 'image/png', 'image/jpeg', and 'image/jpg'"
                )
            # Convert the prediction to doclingDocument
            pred_doc, pred_segmented_pages = self.convert_azure_output_to_docling(
                json.loads(result_json), record
            )

        except Exception as e:
            _log.error(
                f"Error in TableFormer prediction for '{record.doc_id}': {str(e)}"
            )
            status = ConversionStatus.FAILURE
            if not self.ignore_missing_predictions:
                raise
            pred_doc = record.ground_truth_doc.model_copy(
                deep=True
            )  # Use copy of ground truth as fallback

        pred_record = self.create_dataset_record_with_prediction(
            record, pred_doc, result_json
        )
        pred_record.predicted_segmented_pages = pred_segmented_pages
        pred_record.status = status
        return pred_record

    def info(self) -> Dict:
        return {
            "asset": PredictionProviderType.AZURE,
            "version": importlib.metadata.version("azure-ai-documentintelligence"),
        }
