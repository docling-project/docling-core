# Docling Core

[![PyPI version](https://img.shields.io/pypi/v/docling-core)](https://pypi.org/project/docling-core/)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%20%203.11%20%7C%203.12%20%7C%203.13-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling-core)](https://opensource.org/licenses/MIT)

Docling Core is a library that defines core data types and transformations in [Docling](https://github.com/docling-project/docling).

## Installation

To use Docling Core, simply install `docling-core` from your package manager, e.g. pip:
```bash
pip install docling-core
```

### Development setup

To develop for Docling Core, you need Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13 and uv. You can then install from your local clone's root dir:
```bash
uv sync --all-extras
```

To run the pytest suite, execute:
```
uv run pytest -s test
```

## Main features

Docling Core provides the foundational DoclingDocument data model and API, as well as
additional APIs for tasks like serialization and chunking, which are key to developing
generative AI applications using Docling.

### DoclingDocument

Docling Core defines the DoclingDocument as a Pydantic model, allowing for advanced
data model control, customizability, and interoperability.

In addition to specifying the schema, it provides a handy API for building documents,
as well as for basic operations, e.g. exporting to various formats, like Markdown, HTML,
and others.

👉 More details:
- [Architecture docs](https://docling-project.github.io/docling/concepts/architecture/)
- [DoclingDocument docs](https://docling-project.github.io/docling/concepts/docling_document/)

### Serialization

Different users can have varying requirements when it comes to serialization.
To address this, the Serialization API introduces a design that allows easy extension,
while providing feature-rich built-in implementations (on which the respective
DoclingDocument helpers are actually based).

👉 More details:
- [Serialization docs](https://docling-project.github.io/docling/concepts/serialization/)
- [Serialization example](https://docling-project.github.io/docling/examples/serialization/)

### Chunking

Similarly to above, the Chunking API provides built-in chunking capabilities as well as
a design that enables easy extension, this way tackling customization requirements of
different use cases.

👉 More details:
- [Chunking docs](https://docling-project.github.io/docling/concepts/chunking/)
- [Hybrid chunking example](https://docling-project.github.io/docling/examples/hybrid_chunking/)
- [Advanced chunking and serialization](https://docling-project.github.io/docling/examples/advanced_chunking_and_serialization/)

## Contributing

Please read [Contributing to Docling Core](./CONTRIBUTING.md) for details.

## References

If you use Docling Core in your projects, please consider citing the following:

```bib
@techreport{Docling,
  author = "Deep Search Team",
  month = 8,
  title = "Docling Technical Report",
  url = "https://arxiv.org/abs/2408.09869",
  eprint = "2408.09869",
  doi = "10.48550/arXiv.2408.09869",
  version = "1.0.0",
  year = 2024
}
```

## License

The Docling Core codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.
