"""Test DocTags serialization filtering functionality."""

from docling_core.transforms.serializer.doctags import (
    DocTagsDocSerializer,
    DocTagsParams,
    create_task_filtered_params,
)


def test_create_task_filtered_params_defaults():
    """Test default behavior."""
    params = create_task_filtered_params([])
    assert params.include_ocr is True
    assert params.include_layout is True
    assert params.include_otsl is True
    assert params.include_code is True
    assert params.include_picture is True
    assert params.include_chart is True
    assert params.include_formula is True


def test_create_task_filtered_params_specific_tasks():
    """Test with specific task list."""
    params = create_task_filtered_params(["ocr", "layout"])
    assert params.include_ocr is True
    assert params.include_layout is True
    assert params.include_otsl is False
    assert params.include_code is False
    assert params.include_picture is False
    assert params.include_chart is False
    assert params.include_formula is False


def test_create_task_filtered_params_with_layout():
    """Test with layout in task list."""
    params = create_task_filtered_params(["layout"])
    assert params.include_layout is True
    assert params.add_location is True


def test_create_task_filtered_params_without_layout():
    """Test without layout in task list."""
    params = create_task_filtered_params(["ocr"])
    assert params.include_layout is False
    assert params.add_location is False


def test_create_task_filtered_params_with_kwargs():
    """Test with additional kwargs."""
    params = create_task_filtered_params(["ocr"], xsize=1000, ysize=1000)
    assert params.xsize == 1000
    assert params.ysize == 1000
    assert params.include_ocr is True


def test_doctags_exclude_ocr(sample_doc):
    """Test excluding OCR."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_ocr=False).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_exclude_otsl(sample_doc):
    """Test excluding OTSL."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_otsl=False, include_layout=True).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_exclude_picture(sample_doc):
    """Test excluding pictures."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_picture=False).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_exclude_chart(sample_doc):
    """Test excluding charts."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_chart=False).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_exclude_code(sample_doc):
    """Test excluding code."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_code=False).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_exclude_formula(sample_doc):
    """Test excluding formulas."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_formula=False).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_no_layout_no_locations(sample_doc):
    """Test no locations when layout is disabled."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_layout=False, add_location=True).model_dump()
    )
    result = serializer.serialize()
    assert "<loc" not in result.text or result.text == ""


def test_doctags_layout_with_locations(sample_doc):
    """Test locations when layout is enabled."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_layout=True, add_location=True).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_table_location_without_otsl(sample_doc):
    """Test table locations without OTSL."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(
            include_otsl=False, include_layout=True, add_location=True
        ).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_table_caption_without_otsl(sample_doc):
    """Test table captions without OTSL."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(
            include_otsl=False, include_layout=True, add_caption=True
        ).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_multiple_filters(sample_doc):
    """Test multiple filters."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(
            include_ocr=True,
            include_otsl=False,
            include_picture=False,
            include_chart=False,
            include_code=False,
            include_formula=False,
        ).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_layout_mode_only(sample_doc):
    """Test layout mode only."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(
            include_layout=True,
            layout_mode_only=True,
            add_location=True,
            add_content=False,
        ).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_params_mode_minified(sample_doc):
    """Test minified mode."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(mode=DocTagsParams.Mode.MINIFIED).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None
    assert serializer.params.mode == DocTagsParams.Mode.MINIFIED


def test_doctags_params_mode_human_friendly(sample_doc):
    """Test human-friendly mode."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(mode=DocTagsParams.Mode.HUMAN_FRIENDLY).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None


def test_doctags_picture_serializer_helper_methods(sample_doc):
    """Test picture serializer helper methods."""
    serializer = DocTagsDocSerializer(doc=sample_doc)
    picture_serializer = serializer.picture_serializer

    assert hasattr(picture_serializer, "_get_predicted_class")
    assert hasattr(picture_serializer, "_is_chart_type")
    assert hasattr(picture_serializer, "_get_molecule_smi")
    assert hasattr(picture_serializer, "_get_tabular_chart_data")
    assert hasattr(picture_serializer, "_build_body_content")


def test_doctags_should_process_item_logic(sample_doc):
    """Test item processing logic."""
    serializer = DocTagsDocSerializer(doc=sample_doc)

    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_layout=True).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None

    serializer.params = serializer.params.merge_with_patch(
        DocTagsParams(include_layout=True, include_otsl=False).model_dump()
    )
    result = serializer.serialize()
    assert result.text is not None
