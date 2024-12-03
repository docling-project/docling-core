import os
from collections import deque
from pathlib import Path
from typing import List
from unittest.mock import Mock

import pytest
import yaml
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import AnyUrl, ValidationError

from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import (
    CURRENT_VERSION,
    BoundingBox,
    DocItem,
    DoclingDocument,
    DocumentOrigin,
    FloatingItem,
    ImageRef,
    KeyValueItem,
    ListItem,
    PictureItem,
    ProvenanceItem,
    SectionHeaderItem,
    Size,
    TableCell,
    TableData,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel, GroupLabel

GENERATE = False


def test_doc_origin():
    doc_origin = DocumentOrigin(
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="myfile.pdf",
        binary_hash="50115d582a0897fe1dd520a6876ec3f9321690ed0f6cfdc99a8d09019be073e8",
    )


def test_docitems():

    # Iterative function to find all subclasses
    def find_all_subclasses_iterative(base_class):
        subclasses = deque(
            [base_class]
        )  # Use a deque for efficient popping from the front
        all_subclasses = []

        while subclasses:
            current_class = subclasses.popleft()  # Get the next class to process
            for subclass in current_class.__subclasses__():
                all_subclasses.append(subclass)
                subclasses.append(subclass)  # Add the subclass for further exploration

        return all_subclasses

    def serialise(obj):
        return yaml.safe_dump(obj.model_dump(mode="json", by_alias=True))

    def write(name: str, serialisation: str):
        with open(f"./test/data/docling_document/unit/{name}.yaml", "w") as fw:
            fw.write(serialisation)

    def read(name: str):
        with open(f"./test/data/docling_document/unit/{name}.yaml", "r") as fr:
            gold = fr.read()
        return gold

    def verify(dc, obj):
        pred = serialise(obj).strip()
        # print(f"\t{dc.__name__}:\n {pred}")
        gold = read(dc.__name__).strip()

        assert pred == gold, f"pred!=gold for {dc.__name__}"

    # Iterate over the derived classes of the BaseClass
    derived_classes = find_all_subclasses_iterative(DocItem)
    for dc in derived_classes:

        if dc is TextItem:
            obj = dc(
                text="whatever",
                orig="whatever",
                label=DocItemLabel.TEXT,
                self_ref="#",
            )
            verify(dc, obj)
        elif dc is ListItem:
            obj = dc(
                text="whatever",
                orig="whatever",
                marker="(1)",
                enumerated=True,
                self_ref="#",
            )
            verify(dc, obj)
        elif dc is FloatingItem:
            obj = dc(
                label=DocItemLabel.TEXT,
                self_ref="#",
            )
            verify(dc, obj)

        elif dc is KeyValueItem:
            obj = dc(
                label=DocItemLabel.TEXT,
                self_ref="#",
            )
            verify(dc, obj)

        elif dc is SectionHeaderItem:
            obj = dc(
                text="whatever",
                orig="whatever",
                label=DocItemLabel.SECTION_HEADER,
                self_ref="#",
                level=2,
            )
            verify(dc, obj)

        elif dc is PictureItem:
            obj = dc(
                self_ref="#",
            )
            verify(dc, obj)

        elif dc is TableItem:
            obj = dc(
                self_ref="#",
                data=TableData(num_rows=3, num_cols=5, table_cells=[]),
            )
            verify(dc, obj)

        else:
            # print(f"{dc.__name__} is not known")
            assert False, "new derived class detected {dc.__name__}: {e}"


def test_reference_doc():

    filename = "test/data/doc/dummy_doc.yaml"

    # Read YAML file of manual reference doc
    with open(filename, "r") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    doc = DoclingDocument.model_validate(dict_from_yaml)

    # Objects can be accessed
    text_item = doc.texts[0]

    # access members
    text_item.text
    text_item.prov[0].page_no

    # Objects that are references need explicit resolution for now:
    obj = doc.texts[2]  # Text item with parent
    parent = obj.parent.resolve(doc=doc)  # it is a figure

    obj2 = parent.children[0].resolve(
        doc=doc
    )  # Child of figure must be the same as obj

    assert obj == obj2
    assert obj is obj2

    # Iterate all elements

    for item, level in doc.iterate_items():
        _ = f"Item: {item} at level {level}"
        # print(f"Item: {item} at level {level}")

    # Serialize and reload
    _test_serialize_and_reload(doc)

    # Call Export methods
    _test_export_methods(doc, filename=filename)


def test_parse_doc():

    filename = "test/data/doc/2206.01062.yaml"

    with open(filename, "r") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    doc = DoclingDocument.model_validate(dict_from_yaml)

    _test_export_methods(doc, filename=filename)
    _test_serialize_and_reload(doc)


def test_construct_doc():

    filename = "test/data/doc/constructed_document.yaml"

    doc = _construct_doc()

    assert doc.validate_tree(doc.body)
    assert doc.validate_tree(doc.furniture)

    _test_export_methods(doc, filename=filename)
    _test_serialize_and_reload(doc)


def test_construct_bad_doc():

    filename = "test/data/doc/bad_doc.yaml"

    doc = _construct_bad_doc()
    assert doc.validate_tree(doc.body) == False

    _test_export_methods(doc, filename=filename)
    with pytest.raises(ValueError):
        _test_serialize_and_reload(doc)


def _test_serialize_and_reload(doc):
    ### Serialize and deserialize stuff
    yaml_dump = yaml.safe_dump(doc.model_dump(mode="json", by_alias=True))
    # print(f"\n\n{yaml_dump}")
    doc_reload = DoclingDocument.model_validate(yaml.safe_load(yaml_dump))

    yaml_dump_reload = yaml.safe_dump(doc_reload.model_dump(mode="json", by_alias=True))

    assert yaml_dump == yaml_dump_reload, "yaml_dump!=yaml_dump_reload"

    """
    for item, level in doc.iterate_items():
        if isinstance(item, PictureItem):
            _ = item.get_image(doc)
        
    assert doc_reload == doc  # must be equal
    """

    assert doc_reload is not doc  # can't be identical


def _verify_regression_test(pred: str, filename: str, ext: str):

    if os.path.exists(filename + f".{ext}") and not GENERATE:
        with open(filename + f".{ext}", "r") as fr:
            gt_true = fr.read()

        assert gt_true == pred, f"Does not pass regression-test for {filename}.{ext}"
    else:
        with open(filename + f".{ext}", "w") as fw:
            fw.write(pred)

        assert True, "generating the ground-truth for regression test"


def _test_export_methods(doc: DoclingDocument, filename: str):
    ### Iterate all elements
    et_pred = doc.export_to_element_tree()
    _verify_regression_test(et_pred, filename=filename, ext="et")

    ## Export stuff
    md_pred = doc.export_to_markdown()
    _verify_regression_test(md_pred, filename=filename, ext="md")

    # Test HTML export ...
    html_pred = doc.export_to_html()
    _verify_regression_test(html_pred, filename=filename, ext="html")

    # Test DocTags export ...
    dt_pred = doc.export_to_document_tokens()
    _verify_regression_test(dt_pred, filename=filename, ext="dt")

    # Test Tables export ...
    for table in doc.tables:
        table.export_to_markdown()
        table.export_to_html(doc)
        table.export_to_dataframe()
        table.export_to_document_tokens(doc)

    # Test Images export ...

    for fig in doc.pictures:
        fig.export_to_document_tokens(doc)


def _construct_bad_doc():
    doc = DoclingDocument(name="Bad doc")

    title = doc.add_text(label=DocItemLabel.TITLE, text="This is the title")
    group = doc.add_group(parent=title, name="chapter 1")
    text = doc.add_text(
        parent=group,
        label=DocItemLabel.SECTION_HEADER,
        text="This is the first section",
    )

    # Bend the parent of an element to be another.
    text.parent = title.get_ref()

    return doc


def _construct_doc() -> DoclingDocument:

    doc = DoclingDocument(name="Untitled 1")

    title = doc.add_title(
        text="Title of the Document"
    )  # can be done if such information is present, or ommitted.

    # group, heading, paragraph, table, figure, title, list, provenance
    doc.add_text(parent=title, label=DocItemLabel.TEXT, text="Author 1\nAffiliation 1")
    doc.add_text(parent=title, label=DocItemLabel.TEXT, text="Author 2\nAffiliation 2")

    chapter1 = doc.add_group(
        label=GroupLabel.CHAPTER, name="Introduction"
    )  # can be done if such information is present, or ommitted.

    doc.add_heading(
        parent=chapter1,
        text="1. Introduction",
        level=1,
    )
    doc.add_text(
        parent=chapter1,
        label=DocItemLabel.TEXT,
        text="This paper introduces the biggest invention ever made. ...",
    )

    mylist_level_1 = doc.add_group(parent=chapter1, label=GroupLabel.LIST)

    doc.add_list_item(
        parent=mylist_level_1,
        text="list item 1",
    )
    doc.add_list_item(parent=mylist_level_1, text="list item 2")
    doc.add_list_item(
        parent=mylist_level_1,
        text="list item 3",
    )

    mylist_level_2 = doc.add_group(parent=mylist_level_1, label=GroupLabel.ORDERED_LIST)

    doc.add_list_item(
        parent=mylist_level_2,
        text="list item 3.a",
    )
    doc.add_list_item(parent=mylist_level_2, text="list item 3.b")
    doc.add_list_item(
        parent=mylist_level_2,
        text="list item 3.c",
    )

    doc.add_list_item(
        parent=mylist_level_1,
        text="list item 4",
    )

    tab_caption = doc.add_text(
        label=DocItemLabel.CAPTION, text="This is the caption of table 1."
    )

    # Make some table cells
    table_cells = []
    table_cells.append(
        TableCell(
            row_span=2,
            start_row_offset_idx=0,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="Product",
        )
    )
    table_cells.append(
        TableCell(
            col_span=2,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=1,
            end_col_offset_idx=3,
            text="Years",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            text="2016",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            text="2017",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            text="Apple",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            text="49823",
        )
    )
    table_cells.append(
        TableCell(
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            text="695944",
        )
    )
    table_data = TableData(num_rows=3, num_cols=3, table_cells=table_cells)
    doc.add_table(data=table_data, caption=tab_caption)

    fig_caption_1 = doc.add_text(
        label=DocItemLabel.CAPTION, text="This is the caption of figure 1."
    )
    fig_item = doc.add_picture(caption=fig_caption_1)

    size = (64, 64)
    fig2_image = PILImage.new("RGB", size, "black")

    # Draw a red disk touching the borders
    draw = ImageDraw.Draw(fig2_image)
    draw.ellipse((0, 0, size[0] - 1, size[1] - 1), fill="red")

    fig_caption_2 = doc.add_text(
        label=DocItemLabel.CAPTION, text="This is the caption of figure 2."
    )
    fig2_item = doc.add_picture(
        image=ImageRef.from_pil(image=fig2_image, dpi=72), caption=fig_caption_2
    )
    return doc


def test_pil_image():
    doc = DoclingDocument(name="Untitled 1")

    fig_image = PILImage.new(mode="RGB", size=(2, 2), color=(0, 0, 0))
    fig_item = doc.add_picture(image=ImageRef.from_pil(image=fig_image, dpi=72))

    ### Serialize and deserialize the document
    yaml_dump = yaml.safe_dump(doc.model_dump(mode="json", by_alias=True))
    doc_reload = DoclingDocument.model_validate(yaml.safe_load(yaml_dump))
    reloaded_fig = doc_reload.pictures[0]
    reloaded_image = reloaded_fig.image.pil_image

    assert isinstance(reloaded_image, PILImage.Image)
    assert reloaded_image.size == fig_image.size
    assert reloaded_image.mode == fig_image.mode
    assert reloaded_image.tobytes() == fig_image.tobytes()


def test_image_ref():

    data_uri = {
        "dpi": 72,
        "mimetype": "image/png",
        "size": {"width": 10, "height": 11},
        "uri": "file:///tests/data/image.png",
    }
    image = ImageRef.model_validate(data_uri)
    assert isinstance(image.uri, AnyUrl)
    assert image.uri.scheme == "file"
    assert image.uri.path == "/tests/data/image.png"

    data_path = {
        "dpi": 72,
        "mimetype": "image/png",
        "size": {"width": 10, "height": 11},
        "uri": "./tests/data/image.png",
    }
    image = ImageRef.model_validate(data_path)
    assert isinstance(image.uri, Path)
    assert image.uri.name == "image.png"


def test_version_doc():

    # default version
    doc = DoclingDocument(name="Untitled 1")
    assert doc.version == CURRENT_VERSION

    with open("test/data/doc/dummy_doc.yaml") as fp:
        dict_from_yaml = yaml.safe_load(fp)
    doc = DoclingDocument.model_validate(dict_from_yaml)
    assert doc.version == CURRENT_VERSION

    # invalid version
    with pytest.raises(ValidationError, match="NoneType"):
        DoclingDocument(name="Untitled 1", version=None)
    with pytest.raises(ValidationError, match="pattern"):
        DoclingDocument(name="Untitled 1", version="abc")

    # incompatible version (major)
    major_split = CURRENT_VERSION.split(".", 1)
    new_version = f"{int(major_split[0]) + 1}.{major_split[1]}"
    with pytest.raises(ValidationError, match="incompatible"):
        DoclingDocument(name="Untitled 1", version=new_version)

    # incompatible version (minor)
    minor_split = major_split[1].split(".", 1)
    new_version = f"{major_split[0]}.{int(minor_split[0]) + 1}.{minor_split[1]}"
    with pytest.raises(ValidationError, match="incompatible"):
        DoclingDocument(name="Untitled 1", version=new_version)

    # compatible version (equal or lower minor)
    patch_split = minor_split[1].split(".", 1)
    comp_version = f"{major_split[0]}.{minor_split[0]}.{int(patch_split[0]) + 1}"
    doc = DoclingDocument(name="Untitled 1", version=comp_version)
    assert doc.version == CURRENT_VERSION


def test_docitem_get_image():
    # Prepare the document
    doc = DoclingDocument(name="Dummy")

    page1_image = PILImage.new(mode="RGB", size=(200, 400), color=(0, 0, 0))
    doc_item_image = PILImage.new(mode="RGB", size=(20, 40), color=(255, 0, 0))
    page1_image.paste(doc_item_image, box=(20, 40))

    doc.add_page(  # With image
        page_no=1,
        size=Size(width=20, height=40),
        image=ImageRef.from_pil(page1_image, dpi=72),
    )
    doc.add_page(page_no=2, size=Size(width=20, height=40), image=None)  # Without image

    # DocItem with no provenance
    doc_item = DocItem(self_ref="#", label=DocItemLabel.TEXT, prov=[])
    assert doc_item.get_image(doc=doc) is None

    # DocItem on an invalid page
    doc_item = DocItem(
        self_ref="#",
        label=DocItemLabel.TEXT,
        prov=[ProvenanceItem(page_no=3, bbox=Mock(spec=BoundingBox), charspan=(1, 2))],
    )
    assert doc_item.get_image(doc=doc) is None

    # DocItem on a page without page image
    doc_item = DocItem(
        self_ref="#",
        label=DocItemLabel.TEXT,
        prov=[ProvenanceItem(page_no=2, bbox=Mock(spec=BoundingBox), charspan=(1, 2))],
    )
    assert doc_item.get_image(doc=doc) is None

    # DocItem on a page with valid page image
    doc_item = DocItem(
        self_ref="#",
        label=DocItemLabel.TEXT,
        prov=[
            ProvenanceItem(
                page_no=1, bbox=BoundingBox(l=2, t=4, r=4, b=8), charspan=(1, 2)
            )
        ],
    )
    returned_doc_item_image = doc_item.get_image(doc=doc)
    assert (
        returned_doc_item_image is not None
        and returned_doc_item_image.tobytes() == doc_item_image.tobytes()
    )


def test_floatingitem_get_image():
    # Prepare the document
    doc = DoclingDocument(name="Dummy")

    page1_image = PILImage.new(mode="RGB", size=(200, 400), color=(0, 0, 0))
    floating_item_image = PILImage.new(mode="RGB", size=(20, 40), color=(255, 0, 0))
    page1_image.paste(floating_item_image, box=(20, 40))

    doc.add_page(  # With image
        page_no=1,
        size=Size(width=20, height=40),
        image=ImageRef.from_pil(page1_image, dpi=72),
    )
    doc.add_page(page_no=2, size=Size(width=20, height=40), image=None)  # Without image

    # FloatingItem with explicit image different from image based on provenance
    new_image = PILImage.new(mode="RGB", size=(40, 80), color=(0, 255, 0))
    floating_item = FloatingItem(
        self_ref="#",
        label=DocItemLabel.PICTURE,
        prov=[
            ProvenanceItem(
                page_no=1, bbox=BoundingBox(l=2, t=4, r=6, b=12), charspan=(1, 2)
            )
        ],
        image=ImageRef.from_pil(image=new_image, dpi=72),
    )
    retured_image = floating_item.get_image(doc=doc)
    assert retured_image is not None and retured_image.tobytes() == new_image.tobytes()

    # FloatingItem without explicit image and no provenance
    floating_item = FloatingItem(
        self_ref="#", label=DocItemLabel.PICTURE, prov=[], image=None
    )
    assert floating_item.get_image(doc=doc) is None

    # FloatingItem without explicit image on invalid page
    floating_item = FloatingItem(
        self_ref="#",
        label=DocItemLabel.PICTURE,
        prov=[ProvenanceItem(page_no=3, bbox=Mock(spec=BoundingBox), charspan=(1, 2))],
        image=None,
    )
    assert floating_item.get_image(doc=doc) is None

    # FloatingItem without explicit image on a page without page image
    floating_item = FloatingItem(
        self_ref="#",
        label=DocItemLabel.PICTURE,
        prov=[ProvenanceItem(page_no=2, bbox=Mock(spec=BoundingBox), charspan=(1, 2))],
        image=None,
    )
    assert floating_item.get_image(doc=doc) is None

    # FloatingItem without explicit image on a page with page image
    floating_item = FloatingItem(
        self_ref="#",
        label=DocItemLabel.PICTURE,
        prov=[
            ProvenanceItem(
                page_no=1, bbox=BoundingBox(l=2, t=4, r=4, b=8), charspan=(1, 2)
            )
        ],
        image=None,
    )
    retured_image = floating_item.get_image(doc=doc)
    assert (
        retured_image is not None
        and retured_image.tobytes() == floating_item_image.tobytes()
    )


def test_save_pictures():

    doc: DoclingDocument = _construct_doc()

    new_doc = doc._with_pictures_refs(image_dir=Path("./test/data/constructed_images/"))

    img_paths = new_doc._list_images_on_disk()
    assert len(img_paths) == 1, "len(img_paths)!=1"


def _normalise_string_wrt_filepaths(instr: str, paths: List[Path]):

    for p in paths:
        instr = instr.replace(str(p), str(p.name))
        instr = instr.replace(str(p).replace("_", "\\_"), str(p.name))

    return instr


def _verify_saved_output(filename: str, paths: List[Path]):

    pred = ""
    with open(filename, "r") as fr:
        pred = fr.read()

    pred = _normalise_string_wrt_filepaths(pred, paths=paths)

    if GENERATE:
        with open(str(filename) + ".gt", "w") as fw:
            fw.write(pred)
    else:
        gt = ""
        with open(str(filename) + ".gt", "r") as fr:
            gt = fr.read()

        assert pred == gt, f"pred!=gt for {filename}"


def test_save_to_disk():

    doc: DoclingDocument = _construct_doc()

    image_dir = Path("./test/data/doc/constructed_images/")

    doc_with_references = doc._with_pictures_refs(
        image_dir=image_dir  # Path("./test/data/constructed_images/")
    )

    # paths will be different on different machines, so needs to be kept!
    paths = doc_with_references._list_images_on_disk()
    assert len(paths) == 1, "len(paths)!=1"

    ### MarkDown

    filename = Path("test/data/doc/constructed_doc.placeholder.md")
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.PLACEHOLDER
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.embedded.md")
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.referenced.md")
    doc.save_as_markdown(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    ### HTML

    filename = Path("test/data/doc/constructed_doc.placeholder.html")
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.PLACEHOLDER
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.embedded.html")
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.referenced.html")
    doc.save_as_html(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    ### Document Tokens

    filename = Path("test/data/doc/constructed_doc.dt")
    doc.save_as_document_tokens(filename=filename)
    _verify_saved_output(filename=filename, paths=paths)

    ### JSON

    filename = Path("test/data/doc/constructed_doc.embedded.json")
    doc.save_as_json(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.referenced.json")
    doc.save_as_json(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    ### YAML

    filename = Path("test/data/doc/constructed_doc.embedded.yaml")
    doc.save_as_yaml(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.EMBEDDED
    )
    _verify_saved_output(filename=filename, paths=paths)

    filename = Path("test/data/doc/constructed_doc.referenced.yaml")
    doc.save_as_yaml(
        filename=filename, artifacts_dir=image_dir, image_mode=ImageRefMode.REFERENCED
    )
    _verify_saved_output(filename=filename, paths=paths)

    assert True
