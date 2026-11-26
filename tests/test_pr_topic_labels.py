# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pr_topic_labels",
    Path(__file__).resolve().parents[1] / ".github/scripts/pr_topic_labels.py",
)
assert SPEC is not None and SPEC.loader is not None
topics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = topics
SPEC.loader.exec_module(topics)


def test_title_type_scope_and_paths_give_labels() -> None:
    labels = topics.deterministic_topics(
        "fix(Markdown, tables): keep pipe rows",
        [
            "docling_core/transforms/serializer/markdown.py",
            "docling_core/types/doc/items/table/data.py",
            "tests/test_serialization.py",
        ],
    )
    assert labels == ["bug", "serialization", "tables", "document-model"]


def test_doclang_and_doctags_serializers_are_not_serialization() -> None:
    for path, label in (
        ("docling_core/transforms/serializer/doclang.py", "doclang"),
        ("docling_core/transforms/serializer/doctags.py", "doctags"),
    ):
        assert topics.deterministic_topics("refactor: x", [path]) == [label]


def test_documentation_label_only_when_all_source_changes_are_docs() -> None:
    assert topics.deterministic_topics("chore: update guide", ["docs/usage.md", "README.md"]) == ["documentation"]
    assert "documentation" not in topics.deterministic_topics(
        "chore: update", ["docs/usage.md", "docling_core/cli/view.py"]
    )


def test_nested_markdown_files_are_not_documentation_only_changes() -> None:
    assert "documentation" not in topics.deterministic_topics("chore: update readme", ["packages/dclq/README.md"])


def test_unknown_title_types_and_scopes_add_nothing() -> None:
    assert topics.deterministic_topics("chore(release): 2.0", []) == []
    assert topics.deterministic_topics("Not a conventional title", []) == []


def test_model_topics_are_validated_and_limited() -> None:
    with pytest.raises(ValueError, match="unknown topic"):
        topics.parse_model_topics(["wontfix"])
    assert topics.parse_model_topics(None) == []
    assert topics.parse_model_topics(["tables", "tables", "media", "cli", "ci"]) == [
        "tables",
        "media",
        "cli",
    ]


def test_only_existing_repository_labels_are_selected() -> None:
    selected = topics.select_topics(["bug", "doclang"], ["doclang", "tables"], {"bug", "tables"})
    assert selected == ["bug", "tables"]


def test_every_title_and_path_label_is_in_the_allowlist() -> None:
    mapped = {
        *topics.TYPE_LABELS.values(),
        *topics.SCOPE_LABELS.values(),
        *(label for _, label in topics.PATH_LABELS),
    }
    assert mapped <= set(topics.TOPIC_LABELS)


def test_every_path_rule_matches_a_file_in_the_repository() -> None:
    # A renamed module would silently stop adding its label.
    root = Path(__file__).resolve().parents[1]
    for pattern, label in topics.PATH_LABELS:
        assert any(root.glob(pattern)), (pattern, label)


def test_triage_prompt_lists_exactly_the_allowed_labels() -> None:
    # The model answer is validated against TOPIC_LABELS: a label missing from
    # the prompt is never proposed, an extra one makes the answer invalid.
    prompt = (Path(__file__).resolve().parents[1] / ".github/ai-triage/triage-prompt.md").read_text("utf-8")
    listed = prompt.split("Use only these names:", 1)[1].split(".\n", 1)[0]
    assert set(re.findall(r"`([^`]+)`", listed)) == set(topics.TOPIC_LABELS)
