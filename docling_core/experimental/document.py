import base64
import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Annotated, Final, Iterator, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    field_serializer,
)

from docling_core.search.package import VERSION_PATTERN
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import ContentLayer, DocItemLabel
from docling_core.types.doc.document import DoclingDocument as DoclingDocumentV2
from docling_core.types.doc.document import (
    ListGroup,
    ListItem,
    NodeItem,
    RefItem,
    TextItem,
)
from docling_core.types.doc.labels import GroupLabel

CURRENT_VERSION: Final = "2.0.0"

V2_ROOT_REF: Final = "#/body"


class DocMeta(BaseModel):
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = (
        CURRENT_VERSION
    )
    name: str = "Document"


class _FieldAlias(str, Enum):
    TREE = "tree"
    REF = "$ref"


def _create_ref_from_pos(pos: int) -> str:
    return f"#/{_FieldAlias.TREE.value}/{pos}"


def _parse_pos_from_ref(ref: str) -> int:
    return int(ref.split("/")[-1])


def _generate_id() -> str:
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("ascii").rstrip("=")


class BaseNode(BaseModel):
    _id: str = PrivateAttr(
        default_factory=_generate_id
    )  # perhaps can use id(self) instead
    label: Union[DocItemLabel, GroupLabel]  # TODO: streamline this
    layer: ContentLayer = ContentLayer.BODY

    def register_as_parent_of(self, *, other: "BaseNode", doc: "Document") -> None:
        doc._parent_by_node_id[other._get_id()] = self

    def register_as_child_of(self, *, other: "BaseNode", doc: "Document") -> None:
        doc._children_by_node_id.setdefault(other._get_id(), []).append(self)

    def _get_id(self) -> str:
        return self._id  # could use str(id(self)) if deserialization allowed it

    @field_serializer("label")
    def serialize_label(self, label: Union[DocItemLabel, GroupLabel]) -> str:
        return label.value

    def iterate_with_level(
        self, doc: "Document", level: int = 0
    ) -> Iterator[tuple["BaseNode", int]]:
        """Depth-first search."""
        yield self, level
        for child in doc.get_children(self):
            yield from child.iterate_with_level(doc, level + 1)

    def iterate(self, doc: "Document") -> Iterator["BaseNode"]:
        for node, _ in self.iterate_with_level(doc):
            yield node


class RootNode(BaseNode):
    label: Literal[GroupLabel.ROOT] = GroupLabel.ROOT

    def register_as_child_of(self, *, other: "BaseNode", doc: "Document") -> None:
        raise ValueError(f"Cannot register root as child (of: {other})")


class BaseTextNode(BaseNode):
    text: str


class ListNode(BaseNode):
    label: Literal[GroupLabel.LIST] = GroupLabel.LIST

    def register_as_parent_of(self, *, other: "BaseNode", doc: "Document") -> None:
        if not isinstance(other, ListItemNode):
            raise ValueError(f"Invalid parent for {other}: {self}")
        super().register_as_parent_of(other=other, doc=doc)


class ListItemNode(BaseTextNode):
    label: Literal[DocItemLabel.LIST_ITEM] = DocItemLabel.LIST_ITEM
    enumerated: bool = False

    def register_as_child_of(self, *, other: "BaseNode", doc: "Document") -> None:
        if not isinstance(other, ListNode):
            raise ValueError(f"Invalid child for {other}: {self}")
        super().register_as_child_of(other=other, doc=doc)


class TextNode(BaseTextNode):
    label: Literal[DocItemLabel.TEXT] = DocItemLabel.TEXT


class TableCellNode(BaseNode):
    """TableCell."""

    label: Literal[GroupLabel.TABLE_CELL] = GroupLabel.TABLE_CELL
    bbox: Optional[BoundingBox] = None
    # row_span: int = 1  # cannot update this
    # col_span: int = 1  # cannot update this
    # start_row_offset_idx: int
    # end_row_offset_idx: int
    # start_col_offset_idx: int
    # end_col_offset_idx: int
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False


class TableGrid:
    """TableGrid."""

    def __init__(self, num_rows: int, num_cols: int):
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._id_grid: list[list[str]] = []
        self._node_by_id: dict[str, BaseNode] = {}
        for i in range(self._num_rows):
            for _ in range(self._num_cols):
                table_cell = TableCellNode()
                self._node_by_id[table_cell._get_id()] = table_cell
                if len(self._id_grid) <= i:
                    self._id_grid.append([])
                self._id_grid[i].append(table_cell._get_id())

    def get_num_rows(self) -> int:
        return self._num_rows

    def get_num_cols(self) -> int:
        return self._num_cols

    def set_cell(
        self,
        cell: TableCellNode,
        *,
        start_row_idx: int,
        start_col_idx: int,
        row_span: int = 1,
        col_span: int = 1,
    ) -> None:
        if cell._get_id() in self._node_by_id:
            raise ValueError(f"Cell {cell._get_id()} already exists")
        self._node_by_id[cell._get_id()] = cell
        for i in range(start_row_idx, start_row_idx + row_span):
            for j in range(start_col_idx, start_col_idx + col_span):
                self._id_grid[i][j] = cell._get_id()
                # old cell not removed as it may still be referenced in other cells

    def get_grid(self) -> list[list[BaseNode]]:
        return [[self._node_by_id[cell_id] for cell_id in row] for row in self._id_grid]

    def print_ids(self) -> None:
        for cell_id in self._id_grid:
            print(" ".join(cell_id))

    def print_grid(self) -> None:
        for row in self.get_grid():
            print(" ".join([c.model_dump_json() for c in row]))


class TableNode(BaseNode):
    # TODO: implement serialization & deserialization
    model_config = ConfigDict(arbitrary_types_allowed=True)
    label: Literal[DocItemLabel.TABLE] = DocItemLabel.TABLE
    data: TableGrid

    @field_serializer("data")
    def serialize_data(self, data: TableGrid) -> dict:
        return {"num_rows": data.get_num_rows(), "num_cols": data.get_num_cols()}

    def iterate_with_level(
        self, doc: "Document", level: int = 0
    ) -> Iterator[tuple["BaseNode", int]]:
        yield self, level
        for row_idx, row in enumerate(self.data._id_grid):
            for col_idx, cell_id in enumerate(row):
                if (
                    row_idx > 0 and self.data._id_grid[row_idx - 1][col_idx] == cell_id
                ) or (
                    col_idx > 0 and self.data._id_grid[row_idx][col_idx - 1] == cell_id
                ):
                    continue  # skip if cell is a merge cell
                node_id = self.data._id_grid[row_idx][col_idx]
                yield from doc._node_by_id[node_id].iterate_with_level(doc, level + 1)

    def register_as_child_of(self, *, other: "BaseNode", doc: "Document") -> None:
        # copy cells to document:
        for k in self.data._node_by_id:
            doc._node_by_id[k] = self.data._node_by_id[k]
        # TODO: perhaps you must keep all 3 mappings?

        self.data._node_by_id = doc._node_by_id  # is this needed?

        # register self as child of parent
        return super().register_as_child_of(other=other, doc=doc)

    def get_cell(self, row_idx: int, col_idx: int) -> BaseNode:
        return self.data._node_by_id[self.data._id_grid[row_idx][col_idx]]


ConcreteNode = Annotated[
    Union[
        RootNode,
        TextNode,
        ListNode,
        ListItemNode,
        TableNode,
        TableCellNode,
    ],
    Field(discriminator="label"),
]

RefType = Annotated[
    str,
    Field(
        alias=_FieldAlias.REF.value,
        # pattern=_JSON_POINTER_REGEX,
    ),
]


class _StaticBase(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class _StaticRef(_StaticBase):
    ref: RefType


class _StaticNode(_StaticBase):
    _self_ref: RefType
    node: ConcreteNode
    children: list[_StaticRef] = []
    xref: Optional[_StaticRef] = None


class _StaticDoc(_StaticBase):
    meta: DocMeta
    tree: Annotated[list[_StaticNode], Field(alias=_FieldAlias.TREE)]


class Document:
    def __init__(
        self, *, meta: Optional[DocMeta] = None, root_node: Optional[RootNode] = None
    ):
        self._meta: DocMeta = meta or DocMeta()
        self._root: RootNode = root_node or RootNode()
        self._node_by_id: dict[str, BaseNode] = {self._root._get_id(): self._root}
        self._children_by_node_id: dict[str, list[BaseNode]] = {}
        self._parent_by_node_id: dict[str, BaseNode] = {}

        # TODO: implement deserialization
        self._xref_anc_by_ref_node_id: dict[str, BaseNode] = {}
        # self._xref_ref_by_anc_node_id: dict[str, BaseNode] = {}

    def get_parent(self, node: BaseNode) -> Optional[BaseNode]:
        return self._parent_by_node_id.get(node._get_id())

    def get_children(self, node: BaseNode) -> list[BaseNode]:
        return self._children_by_node_id.get(node._get_id(), [])

    def add_node(self, node: BaseNode, *, parent: Optional[BaseNode] = None) -> None:
        actual_parent: BaseNode = parent or self._root
        if self.get_parent(node) is not None:
            raise ValueError(f"Node {node} already has a parent")

        if actual_parent._get_id() not in self._node_by_id:
            raise ValueError(f"Parent {actual_parent} not found in document")

        # # commented out to allow building document bottom-up (deserialization)
        # if node.get_id() in self.node_by_id:
        #     raise ValueError(f"Node {node} already in document")

        self._node_by_id[node._get_id()] = node
        actual_parent.register_as_parent_of(other=node, doc=self)
        node.register_as_child_of(other=actual_parent, doc=self)

    def capture_xref(self, *, anchor: BaseNode, reference: BaseNode) -> None:
        if anchor._get_id() not in self._node_by_id:
            raise ValueError(f"Anchor {anchor} not found in document")
        if reference._get_id() not in self._node_by_id:
            raise ValueError(f"Reference node {reference} not found in document")
        self._xref_anc_by_ref_node_id[reference._get_id()] = anchor
        # self._xref_ref_by_anc_node_id[anchor._get_id()] = reference

    def _export_to_static_doc(self) -> _StaticDoc:
        static_doc = _StaticDoc(meta=self._meta, tree=[])
        position_by_node_id: dict[str, int] = {}
        for st_node in self.iterate():
            node_pos = len(static_doc.tree)
            pos_ref = _create_ref_from_pos(node_pos)
            if parent := self.get_parent(st_node):
                # register node with parent:
                parent_pos = position_by_node_id[parent._get_id()]
                static_doc.tree[parent_pos].children.append(_StaticRef(ref=pos_ref))

            # register node:
            static_node = _StaticNode(
                node=st_node,  # type: ignore[arg-type]
                _self_ref=pos_ref,
            )
            static_doc.tree.append(static_node)
            position_by_node_id[st_node._get_id()] = node_pos

        for ref_node_id, anc_node in self._xref_anc_by_ref_node_id.items():
            ref_pos = position_by_node_id[ref_node_id]
            anc_pos = position_by_node_id[anc_node._get_id()]
            static_doc.tree[ref_pos].xref = _StaticRef(
                ref=_create_ref_from_pos(anc_pos)
            )

        return static_doc

    @classmethod
    def _import_from_static_doc(cls, static_doc: _StaticDoc) -> "Document":

        if not static_doc.tree:
            return Document(meta=static_doc.meta)

        # prepare IDs for all nodes:
        ids = [_generate_id() for _ in range(len(static_doc.tree))]
        node_by_id: dict[str, BaseNode] = {}
        # create root node:
        first_static_node = static_doc.tree[0]
        if not isinstance(first_static_node.node, RootNode):
            raise ValueError(f"First node is not a RootNode: {first_static_node.node}")
        root_node = first_static_node.node
        root_node._id = ids[0]
        doc = Document(meta=static_doc.meta, root_node=root_node)

        # build document bottom-up, i.e. children before parents, as that is the way
        # of traversing the serialized document without unresolved references:
        for i, static_node in enumerate(reversed(static_doc.tree)):
            pos = len(static_doc.tree) - i - 1
            node = static_node.node
            node._id = ids[pos]

            node_by_id[node._get_id()] = node
            for child_id in static_node.children:
                child_pos = _parse_pos_from_ref(child_id.ref)
                if child := node_by_id.get(ids[child_pos]):
                    parent_node = static_node.node
                    doc._node_by_id[parent_node._get_id()] = parent_node
                    doc.add_node(node=child, parent=parent_node)
                else:
                    raise ValueError(f"Child node {child_id} not found")
        return doc

    def export_to_dict(self) -> dict:
        static_doc = self._export_to_static_doc()
        return static_doc.model_dump(
            mode="json",
            by_alias=True,
            # TODO: prevent some fields from being ommitted, e.g. version
            exclude_defaults=True,
            # exclude={'body': {'__all__': {'node': {'id'}}}},
        )

    def save_to_yaml(self, path: Union[str, Path]) -> None:
        static_doc_dict = self.export_to_dict()
        with open(path, "w", encoding="utf-8") as fw:
            yaml.dump(static_doc_dict, fw, sort_keys=False)

    def save_to_json(self, path: Union[str, Path]) -> None:
        static_doc_dict = self.export_to_dict()
        with open(path, "w", encoding="utf-8") as fw:
            json.dump(static_doc_dict, fw, indent=2)

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> "Document":
        with open(path, "r", encoding="utf-8") as f:
            static_doc_dict = yaml.load(f, Loader=yaml.FullLoader)
        static_doc = _StaticDoc.model_validate(static_doc_dict)
        return cls._import_from_static_doc(static_doc)

    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> "Document":
        with open(path, "r", encoding="utf-8") as f:
            static_doc_dict = json.load(f)
        static_doc = _StaticDoc.model_validate(static_doc_dict)
        return cls._import_from_static_doc(static_doc)

    def iterate_with_level(
        self, node: Optional[BaseNode] = None, level: int = 0
    ) -> Iterator[tuple[BaseNode, int]]:
        node = node or self._root
        yield from node.iterate_with_level(self, level=level)

    def iterate(self, node: Optional[BaseNode] = None) -> Iterator[BaseNode]:
        node = node or self._root
        yield from node.iterate(self)

    def _get_tree_str(self) -> str:
        lines = []
        for node, level in self.iterate_with_level():
            lines.append(f'{"  " * level} [{node._get_id()}] {node.model_dump_json()}')
        return "\n".join(lines)

    def print_tree(self) -> None:
        print("tree:")
        print(self._get_tree_str())

    def print_maps(self) -> None:
        print("node_by_id:")
        for k in self._node_by_id:
            print(f"{k}: {self._node_by_id[k].model_dump_json()}")
        print()
        print("children_by_node_id:")
        for k in self._children_by_node_id:
            print(f"{k}:")
            for child in self._children_by_node_id[k]:
                print(f"\t{child._get_id()}: {child.model_dump_json()}")
        print()
        print("parent_by_node_id:")
        for k in self._parent_by_node_id:
            print(
                f"{k}:\t{self._parent_by_node_id[k]._get_id()}: {self._parent_by_node_id[k].model_dump_json()}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return False
        return self.export_to_dict() == other.export_to_dict()

    @classmethod
    def _create_node_from_v2_item(cls, item: NodeItem) -> Optional[BaseNode]:
        if isinstance(item, ListGroup):
            return ListNode(layer=item.content_layer)
        elif isinstance(item, ListItem):
            return ListItemNode(
                text=item.text, enumerated=item.enumerated, layer=item.content_layer
            )
        elif isinstance(item, TextItem) and item.label == DocItemLabel.TEXT:
            return TextNode(text=item.text, layer=item.content_layer)
        # TODO: handle other item types
        else:
            return None

    def export_to_v2(self) -> DoclingDocumentV2:
        v2_doc = DoclingDocumentV2(
            name=self._meta.name,
            # TODO: complete missing fields
        )

        ref_by_node_id: dict[str, str] = {self._root._get_id(): V2_ROOT_REF}

        for node in self.iterate():
            if not (v3_parent := self.get_parent(node)):
                continue  # root node
            v2_parent = RefItem(cref=ref_by_node_id[v3_parent._get_id()]).resolve(
                v2_doc
            )

            if isinstance(node, ListNode):
                item = v2_doc.add_list_group(
                    parent=v2_parent,
                    content_layer=node.layer,
                )
            elif isinstance(node, ListItemNode):
                item = v2_doc.add_list_item(
                    text=node.text,
                    enumerated=node.enumerated,
                    parent=v2_parent,
                )
            elif isinstance(node, TextNode):
                item = v2_doc.add_text(
                    text=node.text,
                    label=DocItemLabel.TEXT,
                    parent=v2_parent,
                )
            else:
                continue

            # register self-ref:
            ref_by_node_id[node._get_id()] = item.self_ref

        return v2_doc

    @classmethod
    def import_from_v2(cls, doc: DoclingDocumentV2) -> "Document":
        root_node = RootNode()
        new_doc = Document(
            root_node=root_node,
            meta=DocMeta(
                name=doc.name,
                # TODO complete missing fields
            ),
        )
        node_by_v2_self_ref: dict[str, BaseNode] = {V2_ROOT_REF: root_node}

        for item, _ in doc.iterate_items(
            with_groups=True,
            traverse_pictures=True,
            included_content_layers={c for c in ContentLayer},
        ):

            # register node:
            node = cls._create_node_from_v2_item(item)
            if node is None:
                continue
            node_by_v2_self_ref[item.self_ref] = node

            if item.parent and (parent := node_by_v2_self_ref.get(item.parent.cref)):
                # register with parent:
                new_doc.add_node(node=node, parent=parent)

        return new_doc


def create_basic_doc_v3() -> Document:
    """Create a document with a list, list items, and text."""
    # Create document
    doc = Document(meta=DocMeta(name="My document"))
    doc.add_node(node=(list_node := ListNode()))
    doc.add_node(node=ListItemNode(text="foo", enumerated=True), parent=list_node)
    doc.add_node(node=ListItemNode(text="bar"), parent=list_node)
    doc.add_node(node=TextNode(text="some text"))
    return doc


def serialize_and_deserialize_doc_v3(
    doc: Document, file_stem: str = "doc_v3"
) -> Document:
    doc.save_to_yaml(yaml_path := Path(f"{file_stem}.yaml"))
    deserialized_doc = Document.load_from_yaml(yaml_path)
    return deserialized_doc


print(f"{80 * '-'}\n### create_basic_doc_v3 ###\n{80 * '-'}")
doc = create_basic_doc_v3()

doc.print_tree()
print()
doc.print_maps()

print(f"{80 * '-'}\n### serialize_and_deserialize_doc_v3 ###\n{80 * '-'}")
deserialized_doc = serialize_and_deserialize_doc_v3(doc)
assert deserialized_doc == doc

print(f"{80 * '-'}\n### doc_from_v2 ###\n{80 * '-'}")


def create_basic_doc_v2() -> DoclingDocumentV2:
    v2_doc = DoclingDocumentV2(name="My document")
    list_group = v2_doc.add_list_group()
    v2_doc.add_list_item(text="foo", enumerated=True, parent=list_group)
    v2_doc.add_list_item(text="bar", parent=list_group)
    v2_doc.add_text(text="some text", label=DocItemLabel.TEXT)
    return v2_doc


v2_doc = create_basic_doc_v2()
doc_from_v2 = Document.import_from_v2(v2_doc)

doc_from_v2.print_tree()
print()
doc_from_v2.print_maps()

# Test equality
assert doc_from_v2 == doc

print(f"{80 * '-'}\n### export_to_v2 ###\n{80 * '-'}")
v2_doc_from_v3 = doc.export_to_v2()

print(v2_doc_from_v3.export_to_markdown())


doc.add_node(node=(anc_1 := TextNode(text="anchor 1")))
doc.add_node(node=(ref_1 := TextNode(text="pointing to a pre-occurring node")))
doc.capture_xref(anchor=anc_1, reference=ref_1)

doc.add_node(node=(ref_2 := TextNode(text="will point to an upcoming node")))
doc.add_node(node=(anc_2 := TextNode(text="anchor 2")))
doc.capture_xref(anchor=anc_2, reference=ref_2)

serialize_and_deserialize_doc_v3(doc, file_stem="doc_v3_xref")

# # check later
# table_cell2 = TableCellNode()
# text_node2 = TextNode(text="disconnected")
# doc.add_node(node=text_node2, parent=table_cell2)  # disconnected tree allowed for the moment

print(f"{80 * '-'}\n### initialize table grid ###\n{80 * '-'}")

table_grid = TableGrid(
    num_rows=2,
    num_cols=3,
)

table_grid.print_ids()
print()
table_grid.print_grid()
print()
doc.print_tree()
print()
doc.print_maps()

print(f"{80 * '-'}\n### set_cell and add table node ###\n{80 * '-'}")

doc.add_node(node=(table_node := TableNode(data=table_grid)))

# add child to an existing table cell:
doc.add_node(
    node=TextNode(text="text in preexisting cell"), parent=table_node.get_cell(1, 0)
)

# merge cells and add child to the new cell:
table_grid.set_cell(
    new_cell := TableCellNode(), start_row_idx=0, start_col_idx=2, row_span=2
)
doc.add_node(node=(list_in_cell := ListNode()), parent=new_cell)
doc.add_node(node=ListItemNode(text="list item in merged cell"), parent=list_in_cell)


table_grid.print_ids()
print()
table_grid.print_grid()
print()
doc.print_tree()
print()
doc.print_maps()


# doc.save_to_yaml(Path("doc_v3_tables.yaml"))
