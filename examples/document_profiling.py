"""Example usage of the document profiler for extracting statistics."""

import time
from pathlib import Path

from docling_core.transforms.profiler import DocumentProfiler
from docling_core.types.doc import DoclingDocument


def profile_single_document():
    """Example: Profile a single document."""
    print("=" * 80)
    print("Example 1: Profiling a Single Document")
    print("=" * 80)

    # Load a document
    doc_path = Path("./examples/2408.09869v3.json")
    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        return

    doc = DoclingDocument.load_from_json(doc_path)

    # Profile the document
    stats = DocumentProfiler.profile_document(doc)

    # Print statistics
    print(f"\nDocument: {stats.name}")
    print(f"Pages: {stats.num_pages}")
    print(f"Tables: {stats.num_tables}")
    print(f"Pictures: {stats.num_pictures}")
    print(f"Text items: {stats.num_texts}")
    print(f"  - Section headers: {stats.num_section_headers}")
    print(f"  - List items: {stats.num_list_items}")
    print(f"  - Code blocks: {stats.num_code_items}")
    print(f"  - Formulas: {stats.num_formulas}")
    print(f"\nTotal items: {stats.total_items}")
    print(f"Average items per page: {stats.avg_items_per_page:.2f}")
    print(f"\nOrigin MIME type: {stats.origin_mimetype}")
    print(f"Likely scanned: {stats.is_likely_scanned}")
    print(f"Pictures requiring OCR: {stats.num_pictures_for_ocr}")

    # Export to JSON
    json_output = stats.model_dump_json(indent=2)
    print(f"\nJSON export (first 500 chars):\n{json_output[:500]}...")


def profile_document_collection():
    """Example: Profile a collection of documents."""
    print("\n" + "=" * 80)
    print("Example 2: Profiling a Document Collection")
    print("=" * 80)

    # Load multiple documents
    doc_dir = Path("./test/data/doc")
    if not doc_dir.exists():
        print(f"Directory not found: {doc_dir}")
        return

    # Load all JSON documents
    docs = []
    for json_file in doc_dir.glob("*.json"):
        try:
            doc = DoclingDocument.load_from_json(json_file)
            docs.append(doc)
        except Exception as e:
            print(f"Skipping {json_file.name}: {e}")

    if not docs:
        print("No documents found")
        return

    print(f"\nLoaded {len(docs)} documents")

    # Profile the collection
    stats = DocumentProfiler.profile_collection(docs, include_individual_stats=True)

    # Print collection statistics
    print("\nCollection Statistics:")
    print(f"Number of documents: {stats.num_documents}")
    print("\nPages:")
    print(f"  Total: {stats.total_pages}")
    print(f"  Min: {stats.min_pages}, Max: {stats.max_pages}")
    print(f"  Median: {stats.median_pages:.1f}, Mean: {stats.mean_pages:.2f}")
    print(f"  Std Dev: {stats.std_pages:.2f}")

    print("\nTables:")
    print(f"  Total: {stats.total_tables}")
    print(f"  Min: {stats.min_tables}, Max: {stats.max_tables}")
    print(f"  Median: {stats.median_tables:.1f}, Mean: {stats.mean_tables:.2f}")
    print(f"  Std Dev: {stats.std_tables:.2f}")

    print("\nPictures:")
    print(f"  Total: {stats.total_pictures}")
    print(f"  Min: {stats.min_pictures}, Max: {stats.max_pictures}")
    print(f"  Median: {stats.median_pictures:.1f}, Mean: {stats.mean_pictures:.2f}")
    print(f"  Std Dev: {stats.std_pictures:.2f}")

    print("\nText Items:")
    print(f"  Total: {stats.total_texts}")
    print(f"  Min: {stats.min_texts}, Max: {stats.max_texts}")
    print(f"  Median: {stats.median_texts:.1f}, Mean: {stats.mean_texts:.2f}")
    print(f"  Std Dev: {stats.std_texts:.2f}")

    print("\nDocument Characteristics:")
    print(f"  Likely scanned documents: {stats.num_likely_scanned}")

    print("\nPictures Requiring OCR:")
    print(f"  Total: {stats.total_pictures_for_ocr}")
    print(f"  Min: {stats.min_pictures_for_ocr}, Max: {stats.max_pictures_for_ocr}")
    print(f"  Median: {stats.median_pictures_for_ocr:.1f}, Mean: {stats.mean_pictures_for_ocr:.2f}")
    print(f"  Std Dev: {stats.std_pictures_for_ocr:.2f}")

    if stats.mimetype_distribution:
        print("\nMIME Type Distribution:")
        for mimetype, count in sorted(stats.mimetype_distribution.items()):
            print(f"  {mimetype}: {count}")

    print("\nComputed Metrics:")
    print(f"  Total items: {stats.total_items}")
    print(f"  Avg items per document: {stats.avg_items_per_document:.2f}")
    print(f"  Avg items per page: {stats.avg_items_per_page:.2f}")

    # Show individual document stats
    if stats.document_stats:
        print("\nIndividual Document Statistics:")
        for i, doc_stat in enumerate(stats.document_stats[:3], 1):  # Show first 3
            print(f"\n  Document {i}: {doc_stat.name}")
            print(f"    Pages: {doc_stat.num_pages}, Tables: {doc_stat.num_tables}, "
                  f"Pictures: {doc_stat.num_pictures}, Texts: {doc_stat.num_texts}")


def profile_with_generator():
    """Example: Profile documents using a generator (memory efficient)."""
    print("\n" + "=" * 80)
    print("Example 3: Profiling with Generator (Memory Efficient)")
    print("=" * 80)

    doc_dir = Path("./test/data/doc")
    if not doc_dir.exists():
        print(f"Directory not found: {doc_dir}")
        return

    def document_generator():
        """Generator that yields documents one at a time."""
        for json_file in doc_dir.glob("*.json"):
            try:
                doc = DoclingDocument.load_from_json(json_file)
                yield doc
            except Exception:
                pass  # Skip invalid documents

    # Profile using generator - documents are not all loaded into memory
    start_time = time.time()
    stats = DocumentProfiler.profile_collection(
        document_generator(),
        include_individual_stats=False  # Don't store individual stats to save memory
    )
    elapsed_time = time.time() - start_time

    print(f"\nProcessed {stats.num_documents} documents in {elapsed_time:.2f} seconds")
    print(f"Total pages: {stats.total_pages}")
    print(f"Total tables: {stats.total_tables}")
    print(f"Total pictures: {stats.total_pictures}")
    print(f"Mean pages per document: {stats.mean_pages:.2f}")


def export_statistics_report():
    """Example: Export statistics to a JSON report."""
    print("\n" + "=" * 80)
    print("Example 4: Exporting Statistics Report")
    print("=" * 80)

    doc_dir = Path("./test/data/doc")
    if not doc_dir.exists():
        print(f"Directory not found: {doc_dir}")
        return

    # Load documents
    docs = []
    for json_file in doc_dir.glob("*.json"):
        try:
            docs.append(DoclingDocument.load_from_json(json_file))
        except Exception:
            pass

    if not docs:
        print("No documents found")
        return

    # Profile collection
    stats = DocumentProfiler.profile_collection(docs, include_individual_stats=True)

    # Export to JSON file
    output_file = Path("./document_statistics_report.json")
    with open(output_file, "w") as f:
        f.write(stats.model_dump_json(indent=2))

    print(f"\nStatistics report exported to: {output_file}")
    print(f"File size: {output_file.stat().st_size} bytes")

    # Also export as Python dict for further processing
    stats_dict = stats.model_dump()
    print(f"\nStatistics as dict (keys): {list(stats_dict.keys())[:10]}...")


def analyze_document_characteristics():
    """Example: Analyze specific document characteristics."""
    print("\n" + "=" * 80)
    print("Example 5: Analyzing Document Characteristics")
    print("=" * 80)

    doc_dir = Path("./test/data/doc")
    if not doc_dir.exists():
        print(f"Directory not found: {doc_dir}")
        return

    # Profile each document individually
    scanned_docs = []
    ocr_candidate_docs = []

    for json_file in doc_dir.glob("*.json"):
        try:
            doc = DoclingDocument.load_from_json(json_file)
            stats = DocumentProfiler.profile_document(doc)

            if stats.is_likely_scanned:
                scanned_docs.append(stats.name)

            if stats.num_pictures_for_ocr > 0:
                ocr_candidate_docs.append((stats.name, stats.num_pictures_for_ocr))
        except Exception:
            pass

    print(f"\nDocuments likely scanned: {len(scanned_docs)}")
    if scanned_docs:
        for name in scanned_docs[:5]:  # Show first 5
            print(f"  - {name}")

    print(f"\nDocuments with OCR requirements: {len(ocr_candidate_docs)}")
    if ocr_candidate_docs:
        for name, count in sorted(ocr_candidate_docs, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {name}: {count} pictures require OCR")


if __name__ == "__main__":
    # Run all examples
    profile_single_document()
    profile_document_collection()
    profile_with_generator()
    export_statistics_report()
    analyze_document_characteristics()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)

