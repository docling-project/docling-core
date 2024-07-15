#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Define utility models and types related to field aliases."""
from pydantic import BaseModel, ConfigDict


class AliasModel(BaseModel):
    """Model for alias fields to ensure instantiation and serialization by alias."""

    model_config = ConfigDict(populate_by_name=True)

    def model_dump(self, **kwargs) -> dict:
        """Generate a dictionary representation of the model using field aliases."""
        if "by_alias" not in kwargs:
            kwargs = {**kwargs, "by_alias": True}

        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        """Generate a JSON representation of the model using field aliases."""
        if "by_alias" not in kwargs:
            kwargs = {**kwargs, "by_alias": True}

        return super().model_dump_json(**kwargs)
