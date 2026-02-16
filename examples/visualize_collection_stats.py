"""Example: Visualizing Collection Statistics with Charts.

This example demonstrates how to use the StatsVisualizer to create
various charts from CollectionStats data.

Requirements:
    pip install docling-core[examples]  # Includes matplotlib
"""

from pathlib import Path

from stats_visualizer import StatsVisualizer

from docling_core.transforms.profiler import CollectionStats, DocumentProfiler
from docling_core.types.doc import DoclingDocument


def load_documents_and_profile(doc_dir: Path) -> CollectionStats | None:
    """Load documents from directory and profile them.

    Args:
        doc_dir: Directory containing JSON documents

    Returns:
        CollectionStats object or None if no documents found
    """
    if not doc_dir.exists():
        print(f"Directory not found: {doc_dir}")
        return None

    docs = []
    for json_file in doc_dir.glob("*.json"):
        try:
            docs.append(DoclingDocument.load_from_json(json_file))
        except Exception:
            pass

    if not docs:
        print("No documents found")
        return None

    # Profile collection
    stats = DocumentProfiler.profile_collection(docs)
    print(f"Loaded and profiled {stats.num_documents} documents")
    return stats


def visualize_single_histogram(stats: CollectionStats):
    """Example 1: Plot a single histogram."""
    print("\n" + "=" * 80)
    print("Example 1: Single Histogram Plot")
    print("=" * 80)

    # Create histogram plot for pages (linear scale)
    fig = StatsVisualizer.plot_histogram(
        histogram=stats.histogram_pages,
        title="Distribution of Pages per Document",
        xlabel="Number of Pages",
        ylabel="Number of Documents",
        color="steelblue",
    )

    # Save the figure
    output_file = Path("./pages_histogram.png")
    StatsVisualizer.save_figure(fig, output_file)
    print(f"Saved histogram to: {output_file}")

    # Create histogram plot for pages (logarithmic scale)
    fig_log = StatsVisualizer.plot_histogram(
        histogram=stats.histogram_pages,
        title="Distribution of Pages per Document (Log Scale)",
        xlabel="Number of Pages",
        ylabel="Number of Documents",
        color="steelblue",
        log_scale=True,
    )

    # Save the figure
    output_file_log = Path("./pages_histogram_log.png")
    StatsVisualizer.save_figure(fig_log, output_file_log)
    print(f"Saved histogram (log scale) to: {output_file_log}")


def visualize_deciles(stats: CollectionStats):
    """Example 2: Plot deciles."""
    print("\n" + "=" * 80)
    print("Example 2: Decile Distribution Plot")
    print("=" * 80)

    # Create decile plot for tables
    fig = StatsVisualizer.plot_deciles(
        deciles=stats.deciles_tables,
        title="Decile Distribution of Tables per Document",
        ylabel="Number of Tables",
        color="forestgreen",
    )

    # Save the figure
    output_file = Path("./tables_deciles.png")
    StatsVisualizer.save_figure(fig, output_file)
    print(f"Saved decile plot to: {output_file}")


def visualize_collection_overview(stats: CollectionStats):
    """Example 3: Create comprehensive overview with multiple metrics."""
    print("\n" + "=" * 80)
    print("Example 3: Collection Overview (Multiple Histograms)")
    print("=" * 80)

    # Create overview plot with all metrics (linear scale)
    fig = StatsVisualizer.plot_collection_overview(
        stats=stats,
        metrics=["pages", "tables", "pictures", "texts"],
        figsize=(16, 10),
    )

    # Save the figure
    output_file = Path("./collection_overview.png")
    StatsVisualizer.save_figure(fig, output_file)
    print(f"Saved collection overview to: {output_file}")

    # Create overview plot with all metrics (logarithmic scale)
    fig_log = StatsVisualizer.plot_collection_overview(
        stats=stats,
        metrics=["pages", "tables", "pictures", "texts"],
        figsize=(16, 10),
        log_scale=True,
    )

    # Save the figure
    output_file_log = Path("./collection_overview_log.png")
    StatsVisualizer.save_figure(fig_log, output_file_log)
    print(f"Saved collection overview (log scale) to: {output_file_log}")


def visualize_deciles_comparison(stats: CollectionStats):
    """Example 4: Compare deciles across multiple metrics."""
    print("\n" + "=" * 80)
    print("Example 4: Decile Comparison Across Metrics")
    print("=" * 80)

    # Create comparison plot
    fig = StatsVisualizer.plot_deciles_comparison(
        stats=stats,
        metrics=["pages", "tables", "pictures", "texts"],
        figsize=(12, 6),
    )

    # Save the figure
    output_file = Path("./deciles_comparison.png")
    StatsVisualizer.save_figure(fig, output_file)
    print(f"Saved decile comparison to: {output_file}")


def create_custom_visualization(stats: CollectionStats):
    """Example 5: Create custom visualization for specific metrics."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Visualization")
    print("=" * 80)

    # Create histogram for pictures only (with log scale for high frequency on low values)
    fig1 = StatsVisualizer.plot_histogram(
        histogram=stats.histogram_pictures,
        title="Picture Distribution (Log Scale)",
        xlabel="Pictures per Document",
        ylabel="Frequency",
        color="coral",
        figsize=(10, 6),
        log_scale=True,
    )
    StatsVisualizer.save_figure(fig1, "./pictures_histogram_log.png")
    print("Saved pictures histogram (log scale)")

    # Create decile plot for texts only
    fig2 = StatsVisualizer.plot_deciles(
        deciles=stats.deciles_texts,
        title="Text Items Decile Distribution",
        ylabel="Number of Text Items",
        color="mediumpurple",
        figsize=(10, 6),
    )
    StatsVisualizer.save_figure(fig2, "./texts_deciles.png")
    print("Saved texts decile plot")

    # Create overview with selected metrics (log scale)
    fig3 = StatsVisualizer.plot_collection_overview(
        stats=stats,
        metrics=["pages", "tables"],  # Only pages and tables
        figsize=(12, 6),
        log_scale=True,
    )
    StatsVisualizer.save_figure(fig3, "./pages_tables_overview_log.png")
    print("Saved pages and tables overview (log scale)")


def display_statistics_summary(stats: CollectionStats):
    """Example 6: Display statistics summary with key insights."""
    print("\n" + "=" * 80)
    print("Example 6: Statistics Summary")
    print("=" * 80)

    print(f"\nCollection Summary ({stats.num_documents} documents):")
    print("\nPages:")
    print(f"  Range: {stats.min_pages} - {stats.max_pages}")
    print(f"  Median (d5): {stats.deciles_pages[4]:.1f}")
    print(f"  Mean: {stats.mean_pages:.2f}")
    print(f"  Deciles: d1={stats.deciles_pages[0]:.1f}, "
          f"d5={stats.deciles_pages[4]:.1f}, d9={stats.deciles_pages[8]:.1f}")

    print("\nTables:")
    print(f"  Range: {stats.min_tables} - {stats.max_tables}")
    print(f"  Median (d5): {stats.deciles_tables[4]:.1f}")
    print(f"  Mean: {stats.mean_tables:.2f}")

    print("\nPictures:")
    print(f"  Range: {stats.min_pictures} - {stats.max_pictures}")
    print(f"  Median (d5): {stats.deciles_pictures[4]:.1f}")
    print(f"  Mean: {stats.mean_pictures:.2f}")

    print("\nText Items:")
    print(f"  Range: {stats.min_texts} - {stats.max_texts}")
    print(f"  Median (d5): {stats.deciles_texts[4]:.1f}")
    print(f"  Mean: {stats.mean_texts:.2f}")


if __name__ == "__main__":
    try:
        # Load documents once and profile them
        doc_dir = Path("./test/data/doc")
        stats = load_documents_and_profile(doc_dir)

        if stats is None:
            print("Failed to load documents. Exiting.")
            exit(1)

        # Run all examples with the same stats object
        visualize_single_histogram(stats)
        visualize_deciles(stats)
        visualize_collection_overview(stats)
        # visualize_deciles_comparison(stats)
        create_custom_visualization(stats)
        display_statistics_summary(stats)

        print("\n" + "=" * 80)
        print("All visualizations created successfully!")
        print("Check the current directory for generated PNG files.")
        print("=" * 80)

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo run this example, install matplotlib:")
        print("  pip install docling-core[examples]")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

