"""Group (container) document items."""

import typing
from typing import TYPE_CHECKING

from pydantic import field_validator
from typing_extensions import deprecated

from docling_core.types.doc.items.node import NodeItem
from docling_core.types.doc.items.text import ListItem
from docling_core.types.doc.labels import GroupLabel

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class GroupItem(NodeItem):  # Container type, can't be a leaf node
    """GroupItem."""

    name: str = (
        "group"  # Name of the group, e.g. "Introduction Chapter",
        # "Slide 5", "Navigation menu list", ...
    )
    # TODO narrow down to allowed values, i.e. excluding those used for subtypes
    label: GroupLabel = GroupLabel.UNSPECIFIED


class ListGroup(GroupItem):
    """ListGroup."""

    label: typing.Literal[GroupLabel.LIST] = GroupLabel.LIST  # type: ignore[assignment]

    @field_validator("label", mode="before")
    @classmethod
    def patch_ordered(cls, value):
        """patch_ordered."""
        return GroupLabel.LIST if value == GroupLabel.ORDERED_LIST else value

    def first_item_is_enumerated(self, doc: "DoclingDocument"):
        """Whether the first list item is enumerated."""
        return (
            len(self.children) > 0
            and isinstance(first_child := self.children[0].resolve(doc), ListItem)
            and first_child.enumerated
        )


@deprecated("Use ListGroup instead.")
class OrderedList(GroupItem):
    """OrderedList."""

    label: typing.Literal[GroupLabel.ORDERED_LIST] = GroupLabel.ORDERED_LIST  # type: ignore[assignment]


class InlineGroup(GroupItem):
    """InlineGroup."""

    label: typing.Literal[GroupLabel.INLINE] = GroupLabel.INLINE


UnorderedList = ListGroup
