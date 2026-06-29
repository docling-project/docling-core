This document demonstrates footnote anchoring across multiple element types.

## Observations

Our analysis relies on a well-established methodology. Regional revenue data was collected across two consecutive quarters from a standardised reporting framework used by all participating business units. The figures presented in Table 1 summarise gross revenue by region before tax adjustments, allowing direct comparison between the North and South territories over the reference period.

The North region consistently outperformed expectations driven by strong consumer demand and favourable currency movements, while the South region recorded more modest growth constrained by supply chain disruptions in the second quarter.

Table 1: Quarterly revenue by region.

| Region   | Q1 Revenue   | Q2 Revenue   |
|----------|--------------|--------------|
| North    | 1.2M         | 1.4M         |
| South    | 0.9M         | 1.1M         |

[^1]

[^1]: Revenue figures are in USD and exclude tax.

[^2]

[^2]: Figures are unaudited estimates.

The revenue growth observed in both regions aligns with the broader market trends reported in the same period. Variance between Q1 and Q2 is within acceptable tolerance bounds and does not indicate systemic reporting issues.

## Pipeline Architecture

The pipeline architecture is illustrated below. The diagram captures the primary processing stages from document ingestion through to structured output generation.

Figure 1: Pipeline overview diagram.

<!-- image -->

[^3]

[^3]: Diagram reproduced with permission from the original authors.

The diagram above illustrates the end-to-end flow of documents through the system. Each stage operates independently, enabling modular replacement of individual components without disrupting the overall pipeline. Further details are available in the appendix.
