# dlgrep JSON Output Handoff

Date: 2026-07-24

## Goal

Replace the current chunk-shaped JSON output for `dlgrep search` and `dlgrep show`
with a dlgrep-native record shape. Use the same basic semantic record for
`dlgrep outline`, while keeping raw XPath results and document inventories
explicitly distinct.

Do not reuse docling-core chunk schemas here. Do not force dlgrep records into
chunk terminology such as `chunk_index`, `num_tokens`, `raw_text`, or a generic
top-level `metadata` bag.

## Core rules

- `search`, `show`, and `outline` share one basic semantic record schema.
- Omit absent fields entirely. Do not serialize `null`.
- Keep source identity first-class, not buried in `metadata`.
- `context.before[]` and `context.after[]` use the same basic record shape as the
  hit itself, not a second ad hoc shape.
- `text` is always a string.
- Structured table data, when present, is additive and never replaces `text`.

## Record shape

```json
{
  "document": "/abs/path/doc.dclg",
  "xpaths": [
    "/d:doclang/d:field_region[3]/d:field_heading[1]"
  ],
  "logical_type": "field_heading",
  "text": "Customer Information",
  "pages": [1],
  "doc_items": ["#/texts/12"],
  "matches": [
    {
      "start": 9,
      "end": 20,
      "text": "Information"
    }
  ],
  "context": {
    "before": [
      {
        "document": "/abs/path/doc.dclg",
        "xpaths": [
          "/d:doclang/d:field_region[3]/d:field_heading[0]"
        ],
        "logical_type": "field_heading",
        "text": "Previous heading",
        "pages": [1],
        "doc_items": ["#/texts/11"]
      }
    ],
    "after": []
  },
  "truncated": true
}
```

## Field semantics

- `document`: input source path or `-` for stdin.
- `xpaths`: all source XPaths contributing to the result.
  - Single-element list for the normal case.
  - Multi-element list for threaded or other aggregate results.
  - Do not also emit a separate `xpath` field.
- `logical_type`: dlgrep semantic type for the hit.
- `text`: canonical plain-text rendering of the hit.
- `pages`: sorted unique pages contributing to the hit.
- `doc_items`: contributing Docling item refs.
- `matches`: present only for `search`.
- `context`: present only when context was requested and there is at least one
  before/after record.
- `truncated`: present only when text was truncated.

`show --section` is a selection mode: it emits one normal record per canonical
semantic element in the selected heading subtree. It does not create a
multi-XPath aggregate record or a synthetic `section` logical type.

## Context rules

- `context.before[]` and `context.after[]` should reuse the same basic record
  schema as the hit:
  - `document`
  - `xpaths`
  - `logical_type`
  - `text`
  - `pages`
  - `doc_items`
  - optional `truncated`
- Context records should not themselves nest `context`.
- Context records should normally not include `matches`.

## Other command semantics

### Outline

- Emit a JSON array of semantic records.
- Add `depth` to each heading record.
- Do not wrap records in a `headings` object or duplicate `xpaths` with `xpath`.

### Select

`select` is a raw XPath query, not necessarily a semantic-unit query:

- Keep its document/result envelope and scalar `value` responses.
- Element results use `xpaths` and `xml`, not the semantic `text` contract.
- Omit `truncated` unless truncation occurred.
- Omit absent fields from optional semantic source bindings.

### Inspect

`inspect` remains a document inventory rather than a semantic record:

- Use `page_count` for the document-level count.
- Use `metadata_elements` for metadata element counts.
- Do not invent `text`, `xpaths`, or `logical_type` for aggregate inventory data.

## Table semantics

There are two separate cases:

1. Whole-table hit
2. Table-cell hit

Do not use a vague field named `table` for both cases unless the meaning is kept
strictly additive and explicit.

### Whole-table hit

When `logical_type == "table"`:

- `text` stays a string and contains the canonical plain-text table rendering.
- A structured `table` field may be added.

Example:

```json
{
  "document": "/abs/path/doc.dclg",
  "xpaths": ["/d:doclang/d:table[1]"],
  "logical_type": "table",
  "text": "| Metric | Value |\n| --- | --- |\n| Gross margin | 48.2% |",
  "pages": [1],
  "doc_items": ["#/tables/0"],
  "table": {
    "caption": "Quarterly results",
    "cells": [
      {"row": 0, "column": 0, "text": "Metric", "role": "column_header"},
      {"row": 0, "column": 1, "text": "Value", "role": "column_header"},
      {"row": 1, "column": 0, "text": "Gross margin", "role": "row_header"},
      {"row": 1, "column": 1, "text": "48.2%"}
    ]
  }
}
```

### Table-cell hit

When the hit is a table-cell-like semantic unit:

- Do not emit the full `table` structure by default.
- Optionally emit `cell_context` with local table semantics only.

Example:

```json
{
  "document": "/abs/path/doc.dclg",
  "xpaths": ["/d:doclang/d:table[1]/d:fcel[3]"],
  "logical_type": "table_cell",
  "text": "48.2%",
  "pages": [1],
  "doc_items": ["#/tables/0"],
  "cell_context": {
    "row": 1,
    "column": 1,
    "row_headers": ["Gross margin"],
    "column_headers": ["Value"],
    "caption": "Quarterly results"
  }
}
```

### Non-table hit

- No `table`
- No `cell_context`

## Explicit non-goals

- No chunk schema compatibility layer in dlgrep JSON.
- No `chunk_index`.
- No `num_tokens`.
- No top-level `headings`.
- No `view`.
- No free-form `metadata` bag for core identity fields such as XPath or type.
