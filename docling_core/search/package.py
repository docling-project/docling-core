"""Models and methods to define a package model."""

import importlib.metadata
import re
from typing import Annotated

from pydantic import BaseModel, StrictStr, StringConstraints

from docling_core.types.base import VERSION_PATTERN


class Package(BaseModel, extra="forbid"):
    """Representation of a software package.

    The version needs to comply with Semantic Versioning 2.0.0.
    """

    name: StrictStr = "docling-core"
    version: Annotated[str, StringConstraints(strict=True, pattern=VERSION_PATTERN)] = importlib.metadata.version(
        "docling-core"
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
