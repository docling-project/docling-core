#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Methods to define fields in an index mapping of a search database."""
from typing import Any, Optional


def es_field(
    *,
    type: Optional[str] = None,
    ignore_above: Optional[int] = None,
    term_vector: Optional[str] = None,
    **kwargs: Any,
):
    """Create x-es kwargs to be passed to a `pydantic.Field` via unpacking."""
    all_kwargs = {**kwargs}

    if type is not None:
        all_kwargs["type"] = type

    if ignore_above is not None:
        all_kwargs["ignore_above"] = ignore_above

    if term_vector is not None:
        all_kwargs["term_vector"] = term_vector

    return {f"x-es-{k}": v for k, v in all_kwargs.items()}
