#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the ID generator types."""

from docling_core.transforms.id_generator.base import BaseIDGenerator  # noqa
from docling_core.transforms.id_generator.doc_hash_id_generator import (  # noqa
    DocHashIDGenerator,
)
from docling_core.transforms.id_generator.uuid_generator import UUIDGenerator  # noqa
