import json
import os
import pathlib

import pytest

from docling_core.transforms.chunker.base_code_chunker import CodeChunk
from docling_core.transforms.chunker.language_code_chunkers import (
    CFunctionChunker,
    JavaFunctionChunker,
    JavaScriptFunctionChunker,
    PythonFunctionChunker,
    TypeScriptFunctionChunker,
)
from docling_core.types.doc.labels import DocItemLabel

from .test_data_gen_flag import GEN_TEST_DATA
from .test_utils_repo_ds import create_ds, language_to_extension

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "chunker_repo"
DATA.mkdir(parents=True, exist_ok=True)

REPO_SPECS = [
    (
        "Java",
        "/test/data/chunker_repo/repos/acmeair",
        "https://github.com/acmeair/acmeair",
        lambda: JavaFunctionChunker(max_tokens=5000),
    ),
    (
        "TypeScript",
        "/test/data/chunker_repo/repos/outline",
        "https://github.com/outline/outline",
        lambda: TypeScriptFunctionChunker(max_tokens=5000),
    ),
    (
        "JavaScript",
        "/test/data/chunker_repo/repos/jquery",
        "https://github.com/jquery/jquery",
        lambda: JavaScriptFunctionChunker(max_tokens=5000),
    ),
    (
        "Python",
        "/test/data/chunker_repo/repos/docling",
        "https://github.com/docling-project/docling",
        lambda: PythonFunctionChunker(max_tokens=5000),
    ),
    (
        "C",
        "/test/data/chunker_repo/repos/json-c",
        "https://github.com/json-c/json-c",
        lambda: CFunctionChunker(max_tokens=5000),
    ),
]


def _dump_or_assert(act_data: dict, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if GEN_TEST_DATA:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(act_data, fp=f, indent=4)
            f.write("\n")
    else:
        with out_path.open(encoding="utf-8") as f:
            exp_data = json.load(fp=f)
        assert exp_data == act_data


@pytest.mark.parametrize("name,local_path,repo_url,chunker_factory", REPO_SPECS)
def test_function_chunkers_repo(name, local_path, repo_url, chunker_factory):

    local_path_full = os.getcwd() + local_path

    if not os.path.isdir(local_path_full):
        pytest.skip(f"Missing repo at {local_path_full}; skipping {name} test.")

    docs = create_ds(local_path_full, repo_url, commit_id="abc123def456")
    docs = [
        doc
        for doc in docs
        if any(text.label == DocItemLabel.CODE and text.text for text in doc.texts)
    ]
    docs = [doc for doc in docs if doc.name.endswith(language_to_extension[name])]
    if not docs:
        pytest.skip(f"No documents found in {local_path_full} for {name}.")

    sample = docs[:3]

    chunker = chunker_factory()
    all_chunks = []
    for doc in sample:
        chunks_iter = chunker.chunk(dl_doc=doc)

        chunks = [CodeChunk.model_validate(n) for n in chunks_iter]
        all_chunks.extend(chunks)
        assert chunks, f"Expected chunks for {doc.name}"
        for c in chunks:
            assert c.text and isinstance(c.text, str)

    act_data = {"root": [c.export_json_dict() for c in all_chunks]}
    out_path = DATA / name / "repo_out_chunks.json"
    _dump_or_assert(act_data, out_path)
