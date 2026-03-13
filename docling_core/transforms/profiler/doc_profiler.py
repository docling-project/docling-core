"""Document profiler for extracting statistics from DoclingDocument objects."""

import statistics
from collections.abc import Iterable
from typing import Annotated

import numpy as np
from annotated_types import Len
from pydantic import BaseModel, Field, computed_field
from typing_extensions import TypeAliasType

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

DecilesT = TypeAliasType("DecilesT", Annotated[list[float], Len(max_length=9, min_length=9)])
"""Type alias for deciles: list of 9 floats representing 1st through 9th deciles (10th, 20th, ..., 90th percentiles)."""


class Histogram(BaseModel):
    """Histogram representation with bins and frequencies."""

    bins: Annotated[list[float], Field(description="Histogram bin edges")] = []
    frequencies: Annotated[list[int], Field(description="Frequency count for each bin")] = []
    bin_width: Annotated[float, Field(description="Width of each bin")] = 0.0


class DocumentStats(BaseModel):
    """Statistics for a single DoclingDocument."""

    name: Annotated[str, Field(description="Document name")]
    num_pages: Annotated[int, Field(description="Number of pages in the document")] = 0
    num_tables: Annotated[int, Field(description="Number of tables in the document")] = 0
    num_pictures: Annotated[int, Field(description="Number of pictures in the document")] = 0
    num_texts: Annotated[int, Field(description="Number of text items in the document")] = 0
    num_key_value_items: Annotated[int, Field(description="Number of key-value items in the document")] = 0
    num_form_items: Annotated[int, Field(description="Number of form items in the document")] = 0

    # Label-specific counts
    num_section_headers: Annotated[int, Field(description="Number of section headers")] = 0
    num_list_items: Annotated[int, Field(description="Number of list items")] = 0
    num_code_items: Annotated[int, Field(description="Number of code items")] = 0
    num_formulas: Annotated[int, Field(description="Number of formula items")] = 0

    # Document characteristics
    origin_mimetype: Annotated[str | None, Field(description="Origin MIME type if available")] = None
    num_pictures_for_ocr: Annotated[
        int,
        Field(description="Number of pictures that would trigger OCR based on area coverage threshold"),
    ] = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_items(self) -> int:
        """Total number of items in the document."""
        return self.num_texts + self.num_tables + self.num_pictures + self.num_key_value_items + self.num_form_items

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_items_per_page(self) -> float:
        """Average number of items per page."""
        if self.num_pages == 0:
            return 0.0
        return self.total_items / self.num_pages


class CollectionStats(BaseModel):
    """Statistics for a collection of DoclingDocument objects."""

    num_documents: Annotated[int, Field(description="Total number of documents in the collection")] = 0

    # Page statistics
    total_pages: Annotated[int, Field(description="Total number of pages across all documents")] = 0
    min_pages: Annotated[int, Field(description="Minimum number of pages in a document")] = 0
    max_pages: Annotated[int, Field(description="Maximum number of pages in a document")] = 0
    deciles_pages: Annotated[DecilesT, Field(description="Deciles of pages per document")] = [0.0] * 9
    histogram_pages: Annotated[Histogram, Field(description="Histogram of pages per document")] = Histogram()
    mean_pages: Annotated[float, Field(description="Mean number of pages per document")] = 0.0
    std_pages: Annotated[float, Field(description="Standard deviation of pages per document")] = 0.0

    # Table statistics
    total_tables: Annotated[int, Field(description="Total number of tables across all documents")] = 0
    min_tables: Annotated[int, Field(description="Minimum number of tables in a document")] = 0
    max_tables: Annotated[int, Field(description="Maximum number of tables in a document")] = 0
    deciles_tables: Annotated[DecilesT, Field(description="Deciles of tables per document")] = [0.0] * 9
    histogram_tables: Annotated[Histogram, Field(description="Histogram of tables per document")] = Histogram()
    mean_tables: Annotated[float, Field(description="Mean number of tables per document")] = 0.0
    std_tables: Annotated[float, Field(description="Standard deviation of tables per document")] = 0.0

    # Picture statistics
    total_pictures: Annotated[int, Field(description="Total number of pictures across all documents")] = 0
    min_pictures: Annotated[int, Field(description="Minimum number of pictures in a document")] = 0
    max_pictures: Annotated[int, Field(description="Maximum number of pictures in a document")] = 0
    deciles_pictures: Annotated[DecilesT, Field(description="Deciles of pictures per document")] = [0.0] * 9
    histogram_pictures: Annotated[Histogram, Field(description="Histogram of pictures per document")] = Histogram()
    mean_pictures: Annotated[float, Field(description="Mean number of pictures per document")] = 0.0
    std_pictures: Annotated[float, Field(description="Standard deviation of pictures per document")] = 0.0

    # Text statistics
    total_texts: Annotated[int, Field(description="Total number of text items across all documents")] = 0
    min_texts: Annotated[int, Field(description="Minimum number of text items in a document")] = 0
    max_texts: Annotated[int, Field(description="Maximum number of text items in a document")] = 0
    deciles_texts: Annotated[DecilesT, Field(description="Deciles of text items per document")] = [0.0] * 9
    histogram_texts: Annotated[Histogram, Field(description="Histogram of text items per document")] = Histogram()
    mean_texts: Annotated[float, Field(description="Mean number of text items per document")] = 0.0
    std_texts: Annotated[float, Field(description="Standard deviation of text items per document")] = 0.0

    # Additional item statistics
    total_key_value_items: Annotated[int, Field(description="Total number of key-value items")] = 0
    total_form_items: Annotated[int, Field(description="Total number of form items")] = 0
    total_section_headers: Annotated[int, Field(description="Total number of section headers")] = 0
    total_list_items: Annotated[int, Field(description="Total number of list items")] = 0
    total_code_items: Annotated[int, Field(description="Total number of code items")] = 0
    total_formulas: Annotated[int, Field(description="Total number of formula items")] = 0

    # Document characteristics
    # Pictures for OCR statistics
    total_pictures_for_ocr: Annotated[
        int, Field(description="Total number of pictures requiring OCR across all documents")
    ] = 0
    min_pictures_for_ocr: Annotated[
        int, Field(description="Minimum number of pictures requiring OCR in a document")
    ] = 0
    max_pictures_for_ocr: Annotated[
        int, Field(description="Maximum number of pictures requiring OCR in a document")
    ] = 0
    deciles_pictures_for_ocr: Annotated[
        DecilesT, Field(description="Deciles of pictures requiring OCR per document")
    ] = [0.0] * 9
    histogram_pictures_for_ocr: Annotated[
        Histogram, Field(description="Histogram of pictures requiring OCR per document")
    ] = Histogram()
    mean_pictures_for_ocr: Annotated[float, Field(description="Mean number of pictures requiring OCR per document")] = (
        0.0
    )
    std_pictures_for_ocr: Annotated[
        float, Field(description="Standard deviation of pictures requiring OCR per document")
    ] = 0.0

    # MIME type distribution
    mimetype_distribution: Annotated[
        dict[str, int], Field(description="Distribution of MIME types in the collection")
    ] = {}

    # Per-document statistics (optional, for detailed analysis)
    document_stats: Annotated[list[DocumentStats], Field(description="Individual statistics for each document")] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_items(self) -> int:
        """Total number of items across all documents."""
        return (
            self.total_texts
            + self.total_tables
            + self.total_pictures
            + self.total_key_value_items
            + self.total_form_items
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_items_per_document(self) -> float:
        """Average number of items per document."""
        if self.num_documents == 0:
            return 0.0
        return self.total_items / self.num_documents

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_items_per_page(self) -> float:
        """Average number of items per page across all documents."""
        if self.total_pages == 0:
            return 0.0
        return self.total_items / self.total_pages


class DocumentProfiler:
    """Profiler for extracting statistics from DoclingDocument objects."""

    @staticmethod
    def _calculate_deciles(data: list[int]) -> list[float]:
        """Calculate deciles (1st through 9th) for a list of values.

        Args:
            data: List of integer values

        Returns:
            List of 9 floats representing [d1, d2, d3, d4, d5, d6, d7, d8, d9]
            (10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th percentiles)
        """
        if not data:
            return [0.0] * 9

        decile_values = np.percentile(data, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        return [float(val) for val in decile_values]

    @staticmethod
    def _calculate_histogram(data: list[int], num_bins: int = 10) -> Histogram:
        """Calculate histogram for a list of values.

        Args:
            data: List of integer values
            num_bins: Number of bins for the histogram (default: 10)

        Returns:
            Histogram object with bins and frequencies
        """
        if not data:
            return Histogram()

        # Use numpy to calculate histogram
        frequencies, bin_edges = np.histogram(data, bins=num_bins)

        # Calculate bin width
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0.0

        return Histogram(
            bins=[float(edge) for edge in bin_edges],
            frequencies=[int(freq) for freq in frequencies],
            bin_width=bin_width,
        )

    @staticmethod
    def profile_document(doc: DoclingDocument, bitmap_coverage_threshold: float = 0.05) -> DocumentStats:
        """Extract statistics from a single DoclingDocument.

        Args:
            doc: The DoclingDocument to profile
            bitmap_coverage_threshold: Threshold for picture area coverage (0-1) to trigger OCR.
                Pictures with area coverage above this threshold are counted as requiring OCR.
                Default is 0.05 (5% of page area).

        Returns:
            DocumentStatistics containing the extracted metrics
        """
        # Count items by label
        label_counts = {
            DocItemLabel.SECTION_HEADER: 0,
            DocItemLabel.LIST_ITEM: 0,
            DocItemLabel.CODE: 0,
            DocItemLabel.FORMULA: 0,
        }

        for text_item in doc.texts:
            if text_item.label in label_counts:
                label_counts[text_item.label] += 1

        # Calculate percentage of pictures that would trigger OCR based on area coverage
        num_pictures_for_ocr = 0
        for picture in doc.pictures:
            # Get picture's bounding box area from provenance
            if picture.prov and len(picture.prov) > 0:
                prov = picture.prov[0]  # Use first provenance item
                bbox = prov.bbox
                picture_area = bbox.width * bbox.height

                # Get page size
                page_no = prov.page_no
                if page_no in doc.pages:
                    page = doc.pages[page_no]
                    page_area = page.size.width * page.size.height

                    # Calculate coverage ratio
                    if page_area > 0:
                        coverage_ratio = picture_area / page_area

                        # Check if coverage exceeds threshold
                        if coverage_ratio >= bitmap_coverage_threshold:
                            num_pictures_for_ocr += 1

        return DocumentStats(
            name=doc.name,
            num_pages=len(doc.pages),
            num_tables=len(doc.tables),
            num_pictures=len(doc.pictures),
            num_texts=len(doc.texts),
            num_key_value_items=len(doc.key_value_items),
            num_form_items=len(doc.form_items),
            num_section_headers=label_counts[DocItemLabel.SECTION_HEADER],
            num_list_items=label_counts[DocItemLabel.LIST_ITEM],
            num_code_items=label_counts[DocItemLabel.CODE],
            num_formulas=label_counts[DocItemLabel.FORMULA],
            origin_mimetype=doc.origin.mimetype if doc.origin else None,
            num_pictures_for_ocr=num_pictures_for_ocr,
        )

    @staticmethod
    def profile_collection(
        documents: Iterable[DoclingDocument] | DoclingDocument,
        include_individual_stats: bool = False,
        bitmap_coverage_threshold: float = 0.05,
        num_bins: int = 10,
    ) -> CollectionStats:
        """Extract statistics from a collection of DoclingDocument objects.

        Args:
            documents: An iterable of DoclingDocument objects, or a single document
            include_individual_stats: Whether to include individual document statistics
                in the result (useful for detailed analysis but increases memory usage)
            bitmap_coverage_threshold: Threshold for picture area coverage (0-1) to
                trigger OCR. Pictures with area coverage above this threshold are counted
                as requiring OCR. Default is 0.05 (5% of page area).
            num_bins: Number of bins for histograms. Default is 10.

        Returns:
            CollectionStatistics containing the aggregated metrics
        """
        # Handle single document case
        if isinstance(documents, DoclingDocument):
            documents = [documents]

        # Collect statistics
        doc_stats_list: list[DocumentStats] = []
        pages_list: list[int] = []
        tables_list: list[int] = []
        pictures_list: list[int] = []
        texts_list: list[int] = []
        pictures_for_ocr_list: list[int] = []

        total_pages = 0
        total_tables = 0
        total_pictures = 0
        total_texts = 0
        total_key_value_items = 0
        total_form_items = 0
        total_section_headers = 0
        total_list_items = 0
        total_code_items = 0
        total_formulas = 0
        total_pictures_for_ocr = 0

        mimetype_distribution: dict[str, int] = {}

        # Process each document
        for doc in documents:
            doc_stats = DocumentProfiler.profile_document(doc, bitmap_coverage_threshold=bitmap_coverage_threshold)

            if include_individual_stats:
                doc_stats_list.append(doc_stats)

            # Collect values for statistics
            pages_list.append(doc_stats.num_pages)
            tables_list.append(doc_stats.num_tables)
            pictures_list.append(doc_stats.num_pictures)
            texts_list.append(doc_stats.num_texts)
            pictures_for_ocr_list.append(doc_stats.num_pictures_for_ocr)

            # Accumulate totals
            total_pages += doc_stats.num_pages
            total_tables += doc_stats.num_tables
            total_pictures += doc_stats.num_pictures
            total_texts += doc_stats.num_texts
            total_key_value_items += doc_stats.num_key_value_items
            total_form_items += doc_stats.num_form_items
            total_section_headers += doc_stats.num_section_headers
            total_list_items += doc_stats.num_list_items
            total_code_items += doc_stats.num_code_items
            total_formulas += doc_stats.num_formulas
            total_pictures_for_ocr += doc_stats.num_pictures_for_ocr

            # Track MIME types
            if doc_stats.origin_mimetype:
                mimetype_distribution[doc_stats.origin_mimetype] = (
                    mimetype_distribution.get(doc_stats.origin_mimetype, 0) + 1
                )

        num_documents = len(pages_list)

        # Handle edge case of empty collection
        if num_documents == 0:
            return CollectionStats()

        # Calculate statistics
        return CollectionStats(
            num_documents=num_documents,
            # Page statistics
            total_pages=total_pages,
            min_pages=min(pages_list),
            max_pages=max(pages_list),
            deciles_pages=DocumentProfiler._calculate_deciles(pages_list),
            histogram_pages=DocumentProfiler._calculate_histogram(pages_list, num_bins=num_bins),
            mean_pages=statistics.mean(pages_list),
            std_pages=statistics.stdev(pages_list) if num_documents > 1 else 0.0,
            # Table statistics
            total_tables=total_tables,
            min_tables=min(tables_list),
            max_tables=max(tables_list),
            deciles_tables=DocumentProfiler._calculate_deciles(tables_list),
            histogram_tables=DocumentProfiler._calculate_histogram(tables_list, num_bins=num_bins),
            mean_tables=statistics.mean(tables_list),
            std_tables=statistics.stdev(tables_list) if num_documents > 1 else 0.0,
            # Picture statistics
            total_pictures=total_pictures,
            min_pictures=min(pictures_list),
            max_pictures=max(pictures_list),
            deciles_pictures=DocumentProfiler._calculate_deciles(pictures_list),
            histogram_pictures=DocumentProfiler._calculate_histogram(pictures_list, num_bins=num_bins),
            mean_pictures=statistics.mean(pictures_list),
            std_pictures=statistics.stdev(pictures_list) if num_documents > 1 else 0.0,
            # Text statistics
            total_texts=total_texts,
            min_texts=min(texts_list),
            max_texts=max(texts_list),
            deciles_texts=DocumentProfiler._calculate_deciles(texts_list),
            histogram_texts=DocumentProfiler._calculate_histogram(texts_list, num_bins=num_bins),
            mean_texts=statistics.mean(texts_list),
            std_texts=statistics.stdev(texts_list) if num_documents > 1 else 0.0,
            # Additional totals
            total_key_value_items=total_key_value_items,
            total_form_items=total_form_items,
            total_section_headers=total_section_headers,
            total_list_items=total_list_items,
            total_code_items=total_code_items,
            total_formulas=total_formulas,
            # Document characteristics
            # Pictures for OCR statistics
            total_pictures_for_ocr=total_pictures_for_ocr,
            min_pictures_for_ocr=min(pictures_for_ocr_list),
            max_pictures_for_ocr=max(pictures_for_ocr_list),
            deciles_pictures_for_ocr=DocumentProfiler._calculate_deciles(pictures_for_ocr_list),
            histogram_pictures_for_ocr=DocumentProfiler._calculate_histogram(pictures_for_ocr_list, num_bins=num_bins),
            mean_pictures_for_ocr=statistics.mean(pictures_for_ocr_list),
            std_pictures_for_ocr=(statistics.stdev(pictures_for_ocr_list) if num_documents > 1 else 0.0),
            mimetype_distribution=mimetype_distribution,
            document_stats=doc_stats_list if include_individual_stats else [],
        )
