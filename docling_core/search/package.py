#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Models and methods to define a package model."""

import importlib.metadata
import re
from typing import Final

from pydantic import BaseModel, StrictStr, StringConstraints
from typing_extensions import Annotated

VERSION_PATTERN: Final = (
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+"
    r"(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


class Package(BaseModel, extra="forbid"):
    """Representation of a software package.

    The version needs to comply with Semantic Versioning 2.0.0.
    """

    name: StrictStr = "docling-core"
    version: Annotated[str, StringConstraints(strict=True, pattern=VERSION_PATTERN)] = (
        importlib.metadata.version("docling-core")
    )

    def __hash__(self):
        """Return the hash value for this S3Path object."""
        return hash((type(self),) + tuple(self.__dict__.values()))

    def get_major(self):
        """Get the major version of this package."""
        return re.match(VERSION_PATTERN, self.version)["major"]

    def get_minor(self):
        """Get the major version of this package."""
        return re.match(VERSION_PATTERN, self.version)["minor"]

    def get_patch(self):
        """Get the major version of this package."""
        return re.match(VERSION_PATTERN, self.version)["patch"]

    def get_pre_release(self):
        """Get the pre-release version of this package."""
        return re.match(VERSION_PATTERN, self.version)["prerelease"]

    def get_build_metadata(self):
        """Get the build metadata version of this package."""
        return re.match(VERSION_PATTERN, self.version)["buildmetadata"]
