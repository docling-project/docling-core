[![PyPI version](https://img.shields.io/pypi/v/docling-core)](https://pypi.org/project/docling-core/)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/ds4sd/deepsearch-toolkit)](https://opensource.org/licenses/MIT)

# Docling Core

Docling Core is a library that defines the data types in [Docling](https://ds4sd.github.io), leveraging pydantic models.

## Installation

Using [Poetry](https://python-poetry.org), create and activate a virtual environment.

```
poetry shell
```

Install the defined dependencies of the project.

```
poetry install
```

Test the installation running the pytest suite.

```
poetry run pytest test
```

## Basic Usage

- You can validate your JSON objects using the pydantic class definition.

  ```py
  from docling_core.types import Document

  data_dict = {...}  # here the object you want to validate, as a dictionary
  Document.model_validate(data_dict)

  data_str = {...}  # here the object as a JSON string
  Document.model_validate_json(data_str)
  ```

- You can generate the JSON schema of a model with the script `ds_generate_jsonschema`.

  ```py
  # for the `Document` type
  ds_generate_jsonschema Document

  # for the use `Record` type
  ds_generate_jsonschema Record
  ```

## Documentation

Docling supports 3 main data types:

- **Document** for publications like books, articles, reports, or patents. When Docling converts an unstructured PDF document, the generated JSON follows this schema.
  The Document type also models the metadata that may be attached to the converted document.
  Check [Document](docs/Document.md) for the full JSON schema. 
- **Record** for structured database records, centered on an entity or _subject_ that is provided with a list of attributes.
  Related to records, the statements can represent annotations on text by Natural Language Processing (NLP) tools.
  Check [Record](docs/Record.md) for the full JSON schema. 
- **Generic** for any data representation, ensuring minimal configuration and maximum flexibility.
  Check [Generic](docs/Generic.md) for the full JSON schema. 

The data schemas are defined using [pydantic](https://pydantic-docs.helpmanual.io/) models, which provide built-in processes to support the creation of data that adhere to those models.

## Contributing

Please read [Contributing to Docling Core](./CONTRIBUTING.md) for details.

## References

If you use `Docling Core` in your projects, please consider citing the following:

```bib
@software{Docling,
author = {Deep Search Team},
month = {7},
title = {{Docling}},
url = {https://github.com/DS4SD/docling},
version = {main},
year = {2024}
}
```

## License

The `Docling Core` codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.
