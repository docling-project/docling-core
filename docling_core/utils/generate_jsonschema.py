#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Generate the JSON Schema of pydantic models and export them to files.

Example:
    python docling_core/utils/generate_jsonschema.py doc.document.TableCell

"""
import argparse
import json
from typing import Any, Union

from pydantic import BaseModel


def _import_class(class_reference: str) -> Any:
    components = class_reference.split(".")
    module_ref = ".".join(components[:-1])
    class_name = components[-1]
    mod = __import__(module_ref, fromlist=[class_name])
    class_type = getattr(mod, class_name)

    return class_type


def generate_json_schema(class_reference: str) -> Union[dict, None]:
    """Generate a jsonable dict of a model's schema from a data type.

    Args:
        class_reference: The reference to a class in 'docling_core.types'.

    Returns:
        A jsonable dict of the model's schema.
    """
    if not class_reference.startswith("docling_core.types."):
        class_reference = "docling_core.types." + class_reference
    class_type = _import_class(class_reference)
    if issubclass(class_type, BaseModel):
        return class_type.model_json_schema()
    else:
        return None


def main() -> None:
    """Print the JSON Schema of a model."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "class_ref", help="Class reference, e.g., doc.document.TableCell"
    )
    args = argparser.parse_args()

    json_schema = generate_json_schema(args.class_ref)
    print(
        json.dumps(json_schema, ensure_ascii=False, indent=2).encode("utf-8").decode()
    )


if __name__ == "__main__":
    main()
