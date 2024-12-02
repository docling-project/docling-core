#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Models for io."""

from io import BytesIO

from pydantic import BaseModel, ConfigDict


class DocumentStream(BaseModel):
    """Wrapper class for a bytes stream with a filename."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    stream: BytesIO
