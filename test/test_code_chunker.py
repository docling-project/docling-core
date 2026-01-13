import glob
import json
import os
import pathlib
import sys
from typing import Optional

import git
import pytest

from docling_core.transforms.chunker.code_chunking._utils import _get_file_extensions
from docling_core.transforms.chunker.code_chunking.code_chunk import CodeChunk
from docling_core.transforms.chunker.code_chunking.standard_code_chunking_strategy import (
    StandardCodeChunkingStrategy,
)
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc import DoclingDocument, DocumentOrigin
from docling_core.types.doc.labels import CodeLanguageLabel, DocItemLabel
from docling_core.utils.legacy import _create_hash

from .test_data_gen_flag import GEN_TEST_DATA


def get_latest_commit_id(file_dir: str) -> str:
    """Returns the latest commit ID in the given Git repository directory."""
    try:
        repo = git.Repo(file_dir, search_parent_directories=True)
        return repo.head.commit.hexsha
    except Exception:
        return ""


def create_documents_from_repository(
    file_dir: str,
    repo_url: str,
    language: CodeLanguageLabel,
    commit_id: Optional[str] = None,
) -> list[DoclingDocument]:
    """Build DoclingDocument objects from a local checkout, one per code file."""

    documents: list[DoclingDocument] = []
    if commit_id is None:
        commit_id = get_latest_commit_id(file_dir)

    all_extensions = set()
    for lang in [
        CodeLanguageLabel.PYTHON,
        CodeLanguageLabel.TYPESCRIPT,
        CodeLanguageLabel.JAVASCRIPT,
        CodeLanguageLabel.C,
        CodeLanguageLabel.JAVA,
    ]:
        all_extensions.update(_get_file_extensions(lang))

    all_files = []
    for extension in all_extensions:
        all_files.extend([f for f in sorted(glob.glob(f"{file_dir}/**/*{extension}", recursive=True))])

    all_files = sorted(set(all_files))

    for file_path in all_files:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        file_relative = os.path.relpath(file_path, start=file_dir).replace("\\", "/")

        origin = DocumentOrigin(
            filename=file_relative,
            uri=(f"{repo_url}/blob/{commit_id}/{file_relative}" if commit_id else f"{repo_url}/{file_relative}"),
            mimetype="text/plain",
            binary_hash=_create_hash(file_content),
        )

        doc = DoclingDocument(name=file_relative, origin=origin)
        doc.add_code(text=file_content, code_language=language)
        documents.append(doc)

    return documents


HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "chunker_repo"
DATA.mkdir(parents=True, exist_ok=True)

REPO_SPECS = [
    (
        "Java",
        "/test/data/chunker_repo/repos/acmeair",
        "https://github.com/acmeair/acmeair",
        lambda: HierarchicalChunker(code_chunking_strategy=StandardCodeChunkingStrategy(max_tokens=5000)),
    ),
    (
        "TypeScript",
        "/test/data/chunker_repo/repos/outline",
        "https://github.com/outline/outline",
        lambda: HierarchicalChunker(code_chunking_strategy=StandardCodeChunkingStrategy(max_tokens=5000)),
    ),
    (
        "JavaScript",
        "/test/data/chunker_repo/repos/jquery",
        "https://github.com/jquery/jquery",
        lambda: HierarchicalChunker(code_chunking_strategy=StandardCodeChunkingStrategy(max_tokens=5000)),
    ),
    (
        "Python",
        "/test/data/chunker_repo/repos/docling",
        "https://github.com/docling-project/docling",
        lambda: HierarchicalChunker(code_chunking_strategy=StandardCodeChunkingStrategy(max_tokens=5000)),
    ),
    (
        "C",
        "/test/data/chunker_repo/repos/json-c",
        "https://github.com/json-c/json-c",
        lambda: HierarchicalChunker(code_chunking_strategy=StandardCodeChunkingStrategy(max_tokens=5000)),
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

    if name == "Java" and sys.version_info < (3, 10):
        pytest.skip("Skipping Java tests on python < 3.10.")

    local_path_full = os.getcwd() + local_path

    if not os.path.isdir(local_path_full):
        pytest.skip(f"Missing repo at {local_path_full}; skipping {name} test.")

    docs = create_documents_from_repository(
        local_path_full,
        repo_url,
        language=CodeLanguageLabel(name),
        commit_id="abc123def456",
    )
    docs = [doc for doc in docs if any(text.label == DocItemLabel.CODE and text.text for text in doc.texts)]
    if not docs:
        pytest.skip(f"No documents found in {local_path_full} for {name}.")

    sample = docs[:3]

    chunker = chunker_factory()
    all_chunks = []
    for doc in sample:
        chunks_iter = chunker.chunk(dl_doc=doc)
        chs = list(chunks_iter)

        chunks = [CodeChunk.model_validate(n) for n in chs]
        all_chunks.extend(chunks)
        assert chunks, f"Expected chunks for {doc.name}"
        for c in chunks:
            assert c.text and isinstance(c.text, str)

    act_data = {"root": [c.export_json_dict() for c in all_chunks]}
    out_path = DATA / name / "repo_out_chunks.json"
    _dump_or_assert(act_data, out_path)
