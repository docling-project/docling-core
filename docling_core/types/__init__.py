#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define the main types."""

from docling_core.types.gen.generic import Generic  # noqa
from docling_core.types.legacy_doc.base import BoundingBox  # noqa
from docling_core.types.legacy_doc.base import Table  # noqa
from docling_core.types.legacy_doc.base import TableCell  # noqa
from docling_core.types.legacy_doc.base import (  # noqa
    BaseCell,
    BaseText,
    PageDimensions,
    PageReference,
    Prov,
    Ref,
)
from docling_core.types.legacy_doc.document import (  # noqa
    CCSDocumentDescription as DocumentDescription,
)
from docling_core.types.legacy_doc.document import (  # noqa
    CCSFileInfoObject as FileInfoObject,
)
from docling_core.types.legacy_doc.document import (  # noqa
    ExportedCCSDocument as Document,
)
from docling_core.types.rec.record import Record  # noqa
