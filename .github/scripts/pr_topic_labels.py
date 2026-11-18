# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Select topic labels for a pull request from the repository label set.

Deterministic rules map the conventional-commit title and the changed source
paths to labels. The AI triage can add a few more labels, but only names from
``TOPIC_LABELS``. The workflow only adds topic labels and never removes them,
because maintainers own these labels.
"""

from __future__ import annotations

import fnmatch
import re

# Existing repository labels that describe what a PR changes. Process labels
# (priority, triage, duplicate, wontfix, ...) are intentionally not here.
TOPIC_LABELS: tuple[str, ...] = (
    "bug",
    "enhancement",
    "documentation",
    "performance",
    "dependencies",
    "security",
    "ci",
    "doclang",
    "doctags",
    "document-model",
    "serialization",
    "tables",
    "chunking",
    "media",
    "cli",
)
MAX_MODEL_TOPICS = 3
MAX_TOPICS = 6

TITLE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?!?:")
TYPE_LABELS = {
    "fix": "bug",
    "feat": "enhancement",
    "docs": "documentation",
    "perf": "performance",
    "ci": "ci",
    "build": "ci",
}
# Keys are lowercase: the scope is lowercased before the lookup.
SCOPE_LABELS = {
    "doclang": "doclang",
    "dclx": "doclang",
    "doctags": "doctags",
    "idoctags": "doctags",
    "document": "document-model",
    "doc": "document-model",
    "docitem": "document-model",
    "types_doc": "document-model",
    "meta": "document-model",
    "origin": "document-model",
    "serializer": "serialization",
    "serialization": "serialization",
    "markdown": "serialization",
    "md": "serialization",
    "html": "serialization",
    "html-serializer": "serialization",
    "latex": "serialization",
    "outline": "serialization",
    "table": "tables",
    "tables": "tables",
    "otsl": "tables",
    "chunker": "chunking",
    "chunking": "chunking",
    "webvtt": "media",
    "vtt": "media",
    "media": "media",
    "cli": "cli",
    "dclq": "cli",
    "viewer": "cli",
    "deps": "dependencies",
    "security": "security",
    "ci": "ci",
    "mergify": "ci",
}
# fnmatch: `*` also matches `/`. The serializer files are listed one by one,
# because the `serialization` label does not cover DocLang and DocTags.
PATH_LABELS: tuple[tuple[str, str], ...] = (
    ("docling_core/transforms/serializer/doclang.py", "doclang"),
    ("docling_core/transforms/serializer/_doclang_utils.py", "doclang"),
    ("docling_core/transforms/deserializer/doclang*", "doclang"),
    ("docling_core/transforms/serializer/doctags.py", "doctags"),
    ("docling_core/types/doc/doctags.py", "doctags"),
    ("docling_core/types/doc/tokens.py", "doctags"),
    ("docling_core/transforms/serializer/markdown*", "serialization"),
    ("docling_core/transforms/serializer/html*", "serialization"),
    ("docling_core/transforms/serializer/latex.py", "serialization"),
    ("docling_core/transforms/serializer/plain_text.py", "serialization"),
    ("docling_core/transforms/serializer/azure.py", "serialization"),
    ("docling_core/transforms/serializer/base.py", "serialization"),
    ("docling_core/transforms/serializer/common.py", "serialization"),
    ("docling_core/experimental/serializer/*", "serialization"),
    ("docling_core/transforms/serializer/webvtt.py", "media"),
    ("docling_core/types/doc/webvtt.py", "media"),
    ("docling_core/types/doc/items/picture/*", "media"),
    ("docling_core/types/doc/items/table/*", "tables"),
    ("docling_core/types/doc/document.py", "document-model"),
    ("docling_core/types/doc/base.py", "document-model"),
    ("docling_core/types/doc/labels.py", "document-model"),
    ("docling_core/types/doc/page.py", "document-model"),
    ("docling_core/types/doc/items/*", "document-model"),
    ("docling_core/types/doc/common/*", "document-model"),
    ("docling_core/transforms/chunker/*", "chunking"),
    ("docling_core/cli/*", "cli"),
    ("packages/dclq/dclq/*", "cli"),
    ("uv.lock", "dependencies"),
    (".github/*", "ci"),
    (".pre-commit-config.yaml", "ci"),
)
DOC_PATHS = ("docs/*",)


def _is_doc_path(path: str) -> bool:
    """True for files under docs/ and for Markdown files in the repository root.

    `fnmatch` lets `*` match `/`, so `*.md` alone would also match a README
    inside `packages/` or agent instruction files, which is not a
    documentation-only change.
    """
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in DOC_PATHS) or (
        "/" not in path and path.endswith(".md")
    )


def deterministic_topics(title: str, paths: list[str]) -> list[str]:
    """Return topic labels from the PR title and the changed paths."""
    labels: list[str] = []

    def add(label: str) -> None:
        if label not in labels:
            labels.append(label)

    match = TITLE.match(title.strip())
    if match:
        if match.group("type") in TYPE_LABELS:
            add(TYPE_LABELS[match.group("type")])
        for scope in re.split(r"[,/ ]+", match.group("scope") or ""):
            if scope.lower() in SCOPE_LABELS:
                add(SCOPE_LABELS[scope.lower()])
    for path in paths:
        for pattern, label in PATH_LABELS:
            if fnmatch.fnmatchcase(path, pattern):
                add(label)
    source = [path for path in paths if not path.startswith("tests/data/")]
    if source and all(_is_doc_path(path) for path in source):
        add("documentation")
    return labels


def parse_model_topics(value: object) -> list[str]:
    """Validate the topics of the model answer against the allowlist."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("topics must be a list")
    topics: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in TOPIC_LABELS:
            raise ValueError(f"unknown topic label: {item!r}")
        if item not in topics:
            topics.append(item)
    return topics[:MAX_MODEL_TOPICS]


def select_topics(deterministic: list[str], model: list[str], existing: set[str]) -> list[str]:
    """Merge the topics. Only labels that exist in the repository are kept."""
    merged: list[str] = []
    for label in [*deterministic, *model]:
        if label in existing and label not in merged:
            merged.append(label)
    return merged[:MAX_TOPICS]
