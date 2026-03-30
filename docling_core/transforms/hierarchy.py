"""Document hierarchy transforms: flatten and reconstruct section nesting."""

import logging
from typing import Optional

from docling_core.types.doc.document import (
    DoclingDocument,
    FieldItem,
    FieldRegionItem,
    FormItem,
    GroupItem,
    KeyValueItem,
    ListGroup,
    ListItem,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

_logger = logging.getLogger(__name__)


class DocHierarchyTransform:
    """Transform a DoclingDocument between flat and hierarchical section nesting.

    ``to_flat`` moves every item to be a direct child of body, dissolving
    section → content nesting while preserving list, table, and picture
    internal structure.

    ``to_hierarchical`` rebuilds maximal section nesting: each non-header item
    becomes a child of the deepest open section header whose level is lower,
    or of body if no header has been seen yet.
    """

    @staticmethod
    def _copy_list_group(
        source: ListGroup,
        source_doc: DoclingDocument,
        target_doc: DoclingDocument,
        parent: NodeItem,
    ) -> ListGroup:
        new_group = target_doc.add_list_group(
            parent=parent, content_layer=source.content_layer
        )
        new_group.meta = source.meta
        for child_ref in source.children or []:
            try:
                child = child_ref.resolve(source_doc)
                if isinstance(child, ListItem):
                    new_item = target_doc.add_list_item(
                        text=child.text,
                        enumerated=child.enumerated,
                        parent=new_group,
                        content_layer=child.content_layer,
                    )
                    new_item.meta = child.meta
                    # Recursively copy nested list groups
                    for nested_ref in child.children or []:
                        try:
                            nested = nested_ref.resolve(source_doc)
                            if isinstance(nested, ListGroup):
                                DocHierarchyTransform._copy_list_group(
                                    nested, source_doc, target_doc, new_item
                                )
                        except Exception:
                            pass
            except Exception as exc:
                _logger.warning(f"Could not copy list child: {exc}")
        return new_group

    @staticmethod
    def _copy_table(
        source: TableItem,
        source_doc: DoclingDocument,
        target_doc: DoclingDocument,
        parent: NodeItem,
    ) -> TableItem:
        new_table = target_doc.add_table(
            data=source.data, parent=parent, content_layer=source.content_layer
        )
        new_table.meta = source.meta
        for cap_ref in source.captions:
            try:
                cap = cap_ref.resolve(source_doc)
                if isinstance(cap, TextItem):
                    new_cap = target_doc.add_text(
                        label=cap.label,
                        text=cap.text,
                        parent=new_table,
                        content_layer=cap.content_layer,
                    )
                    new_cap.meta = cap.meta
                    new_table.captions.append(new_cap.get_ref())
            except Exception as exc:
                _logger.warning(f"Could not copy table caption: {exc}")
        return new_table

    @staticmethod
    def _copy_picture(
        source: PictureItem,
        source_doc: DoclingDocument,
        target_doc: DoclingDocument,
        parent: NodeItem,
    ) -> PictureItem:
        new_pic = target_doc.add_picture(
            image=source.image, parent=parent, content_layer=source.content_layer
        )
        new_pic.meta = source.meta
        for cap_ref in source.captions:
            try:
                cap = cap_ref.resolve(source_doc)
                if isinstance(cap, TextItem):
                    new_cap = target_doc.add_text(
                        label=cap.label,
                        text=cap.text,
                        parent=new_pic,
                        content_layer=cap.content_layer,
                    )
                    new_cap.meta = cap.meta
                    new_pic.captions.append(new_cap.get_ref())
            except Exception as exc:
                _logger.warning(f"Could not copy picture caption: {exc}")
        return new_pic

    @staticmethod
    def _flatten_into(
        node: NodeItem,
        source_doc: DoclingDocument,
        target_doc: DoclingDocument,
        target_parent: NodeItem,
    ) -> None:
        """Recursively add node's children to target_parent, preserving atomic units."""
        for child_ref in node.children or []:
            try:
                child = child_ref.resolve(source_doc)
            except Exception as exc:
                _logger.warning(f"Could not resolve child {child_ref}: {exc}")
                continue

            if isinstance(child, ListGroup):
                DocHierarchyTransform._copy_list_group(child, source_doc, target_doc, target_parent)
            elif isinstance(child, TableItem):
                DocHierarchyTransform._copy_table(child, source_doc, target_doc, target_parent)
            elif isinstance(child, PictureItem):
                DocHierarchyTransform._copy_picture(child, source_doc, target_doc, target_parent)
            elif isinstance(child, TitleItem):
                new_item = target_doc.add_title(
                    text=child.text,
                    parent=target_parent,
                    content_layer=child.content_layer,
                )
                new_item.meta = child.meta
                DocHierarchyTransform._flatten_into(child, source_doc, target_doc, target_parent)
            elif isinstance(child, SectionHeaderItem):
                new_item = target_doc.add_heading(
                    text=child.text,
                    level=child.level,
                    parent=target_parent,
                    content_layer=child.content_layer,
                )
                new_item.meta = child.meta
                DocHierarchyTransform._flatten_into(child, source_doc, target_doc, target_parent)
            elif isinstance(child, ListItem):
                _logger.warning(f"ListItem {child.self_ref} found outside a ListGroup; skipping")
            elif isinstance(child, TextItem):
                new_item = target_doc.add_text(
                    label=child.label,
                    text=child.text,
                    parent=target_parent,
                    content_layer=child.content_layer,
                )
                new_item.meta = child.meta
            elif isinstance(child, KeyValueItem | FormItem):
                _logger.warning(
                    f"Item of type {type(child).__name__} at {child.self_ref} is not yet "
                    f"supported by DocHierarchyTransform; skipping"
                )
            elif isinstance(child, FieldRegionItem | FieldItem):
                _logger.warning(
                    f"Item of type {type(child).__name__} at {child.self_ref} is not yet "
                    f"supported by DocHierarchyTransform; skipping"
                )
            elif isinstance(child, GroupItem):
                # Dissolve other groups (InlineGroup, OrderedList, …) by recursing
                # into their children without adding the group itself.
                DocHierarchyTransform._flatten_into(child, source_doc, target_doc, target_parent)
            else:
                _logger.warning(f"Unhandled item type {type(child).__name__} at {child.self_ref}; skipping")

    @staticmethod
    def to_flat(doc: DoclingDocument) -> DoclingDocument:
        """Return a new document where every item is a direct child of body.

        Iterates ``doc`` in document order and appends each item to the new body,
        preserving:

        - ``SectionHeaderItem.level``  (needed for ``to_hierarchical`` to invert)
        - List internal structure  (``ListGroup`` → ``ListItem`` nesting is kept)
        - Table / picture caption children
        - ``content_layer`` of every item

        All other parent-child links (section → text) are dissolved.
        """
        _logger.info(f"DocHierarchyTransform.to_flat: doc={doc.name!r}")
        new_doc = DoclingDocument(name=doc.name)
        DocHierarchyTransform._flatten_into(doc.body, doc, new_doc, new_doc.body)
        return new_doc

    @staticmethod
    def to_hierarchical(doc: DoclingDocument) -> DoclingDocument:
        """Return a new document with maximal section nesting.

        Iterates ``doc`` in document order (after flattening first).  Maintains a
        stack of open section headers keyed by their level.  Each non-header item
        (text, table, picture, list) is appended as a child of the most recently
        opened section header (or of body if no header has been seen yet).
        A section header at level N is appended as a child of the nearest ancestor
        whose level is strictly less than N.

        Lists, table-caption pairs, and picture-caption pairs are treated as atomic
        units and are not split across parent boundaries.
        """
        _logger.info(f"DocHierarchyTransform.to_hierarchical: doc={doc.name!r}")
        flat = DocHierarchyTransform.to_flat(doc)
        new_doc = DoclingDocument(name=doc.name)

        # open_sections maps level -> NodeItem (only section headers, not title)
        open_sections: dict[int, NodeItem] = {}
        # title_node is the most recently seen TitleItem; text before any section
        # header becomes a child of the title rather than of body.
        title_node: Optional[NodeItem] = None

        def _current_parent() -> NodeItem:
            if open_sections:
                return open_sections[max(open_sections)]
            if title_node is not None:
                return title_node
            return new_doc.body

        def _parent_for_level(level: int) -> NodeItem:
            # Section headers nest only under other section headers, never under title.
            candidates = [lv for lv in open_sections if lv < level]
            if not candidates:
                return new_doc.body
            return open_sections[max(candidates)]

        for child_ref in flat.body.children or []:
            try:
                child = child_ref.resolve(flat)
            except Exception as exc:
                _logger.warning(f"Could not resolve body child {child_ref}: {exc}")
                continue

            if isinstance(child, TitleItem):
                new_item = new_doc.add_title(
                    text=child.text,
                    parent=new_doc.body,
                    content_layer=child.content_layer,
                )
                new_item.meta = child.meta
                title_node = new_item
                open_sections = {}

            elif isinstance(child, SectionHeaderItem):
                level = child.level
                parent = _parent_for_level(level)
                new_item = new_doc.add_heading(
                    text=child.text,
                    level=level,
                    parent=parent,
                    content_layer=child.content_layer,
                )
                new_item.meta = child.meta
                # Close all open sections at >= this level
                open_sections = {lv: n for lv, n in open_sections.items() if lv < level}
                open_sections[level] = new_item

            elif isinstance(child, ListGroup):
                DocHierarchyTransform._copy_list_group(child, flat, new_doc, _current_parent())

            elif isinstance(child, TableItem):
                DocHierarchyTransform._copy_table(child, flat, new_doc, _current_parent())

            elif isinstance(child, PictureItem):
                DocHierarchyTransform._copy_picture(child, flat, new_doc, _current_parent())

            elif isinstance(child, TextItem):
                new_item = new_doc.add_text(
                    label=child.label,
                    text=child.text,
                    parent=_current_parent(),
                    content_layer=child.content_layer,
                )
                new_item.meta = child.meta

            elif isinstance(child, KeyValueItem | FormItem):
                _logger.warning(
                    f"Item of type {type(child).__name__} at {child.self_ref} is not yet "
                    f"supported by DocHierarchyTransform; skipping"
                )

            elif isinstance(child, FieldRegionItem | FieldItem):
                _logger.warning(
                    f"Item of type {type(child).__name__} at {child.self_ref} is not yet "
                    f"supported by DocHierarchyTransform; skipping"
                )

            else:
                _logger.warning(f"Unhandled item type {type(child).__name__} at {child.self_ref}; skipping")

        return new_doc
