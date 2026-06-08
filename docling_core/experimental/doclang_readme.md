# DocLang implementation notes

This document tracks known discrepancies between the DocLang serializer/deserializer in
[`doclang.py`](doclang.py) and the [DocLang specification](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md).

**Tracked spec version:** `0.5` (see `_DOCLANG_VERSION` in `doclang.py`).

When `_DOCLANG_VERSION` changes, update this file against the matching spec tag:
`https://github.com/doclang-project/doclang/blob/v{_DOCLANG_VERSION}.0/spec.md`.

## How to maintain

Add or remove entries when implementation or spec changes. For each entry:

- Cite the relevant spec section (anchor in `spec.md` when possible).
- Point to code (`doclang.py`) and/or tests under `test/test_serialization_doclang.py` or
  `test/test_deserializer_doclang.py`.
- Mark status as **open**, **partial**, or **by design** (intentional, often via `DoclangParams`).

Re-validate against the reference validator when installed (`uv sync --extra doclang-validation`).

---

## Serialization gaps

Docling document constructs that are not emitted as spec-conformant DocLang (or are omitted).

| Topic | Spec reference | Status | Notes |
| --- | --- | --- | --- |
| `FormItem` | [Fields](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#fields) | **open** | `DoclangFormSerializer` is a no-op. Legacy graph-based forms are expected to use `_migrate_to_field_regions()` before export (`test_kv_migration_*`). |
| `KeyValueItem` (pre-migration) | [Fields](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#fields) | **open** | `DoclangKeyValueSerializer` is a no-op. Migrate to field regions first. |
| `DocItemLabel.EMPTY_VALUE` | [Fields](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#fields) | **open** | Currently mapped to `<text>` with a `FIXME` to emit proper field markup (`DoclangTextSerializer._serialize_single_item`). |
| Temporal tokens | Appendix A | **open** | Not emitted despite vocabulary support. |
| `<xref>` | [Cross-references](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#cross-references) | **open** | Not emitted. |
| List-item split across pages | [Lists — single list-item broken by a page break](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#lists) | **open** | See deserialization gaps; serializer does not emit per-item `<thread>` + reopened `<list>` for a single `ListItem` spanning pages. Cross-page lists with whole items per page are covered (`test_cross_page_list_*`). |
| Document metadata in `<head>` | [Appendix C — Metadata](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#metadata) | **open** | Only `<default_resolution>` is emitted when `xsize`/`ysize` differ from 512. Future `<head>` elements (`title`, `author`, `date`, …) are not serialized. |
| OTSL structural validation | [Tables / OTSL](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#tables) | **partial** | `xcel` / `lcel` / `ucel` placement is produced for common cases (`test_table_xcel_span`, `test_table_corn_header`) but not validated against the full OTSL grammar (`# TODO` in `test_serialization_doclang.py`). |

---

## Deserialization gaps

Features emitted (or accepted) by conforming DocLang XML that are not reconstructed into
`DoclingDocument` today.

| Topic | Spec reference | Status | Notes |
| --- | --- | --- | --- |
| Fields (`<field_region>`, `<field_item>`, `<key>`, `<value>`, `<hint>`) | [Fields](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#fields) | **open** | Serialization is covered (`DoclangFallbackSerializer`, `test_kv_*`). `_dispatch_element` has no field handlers; unknown tags fall through to `_walk_children`, so field markup is not parsed into `FieldRegionItem` / `FieldItem`. |
| Hyperlinks (`<href uri="…"/>`) | [References — hyperlinks](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#hyperlinks) | **open** | Serialized via `_create_href_token` (`test_text_with_hyperlink`). Not read back into `TextItem.hyperlink`. |
| Cross-references (`<xref thread_id="…"/>`) | [References — cross-references](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#cross-references) | **open** | Token is in `DoclangVocabulary`; neither serialization nor deserialization implemented. |
| Custom vocabularies (`<custom>…</custom>`) | [Custom vocabularies](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#custom-vocabularies) | **open** | Serialized from item meta (`_serialize_item_custom_head`, `DoclangMetaSerializer`). `_walk_children` skips `<custom>` at pass-through level; content is not restored to meta. |
| Document head (`<head>`, `<default_resolution>`) | [Document head](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#document-head), [`default_resolution`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#default_resolution) | **open** | Non-default resolution is serialized in `_create_head` (`test_include_namespace_and_version`). `<head>` children are ignored during `_parse_document_root` / `_walk_children`. Deserializer always uses `DOCLANG_DFLT_RESOLUTION` (512). |
| Picture `src` (`<src uri="…"/>`) | [`<src>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#src) | **open** | Serialized by `DoclangPictureSerializer` (`_create_src_token`). `_parse_picture` does not load image URI into `PictureItem`. |
| Picture `class="chart"` | [`<picture>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#picture) | **partial** | Chart table body under `<picture>` is deserialized into `meta.tabular_chart`. The `class="chart"` attribute itself is not read; chart vs non-chart is inferred only from embedded OTSL. |
| Temporal tokens (`<hour>`, `<minute>`, `<second>`, `<centisecond>`) | Appendix A — temporal property elements | **open** | Listed in `DoclangToken` / `_ELEMENT_HEAD_TAGS` but never produced or consumed. |
| Whitespace-only text | [Content encoding and whitespace](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#content-encoding-and-whitespace-handling) | **open** | `_get_children_simple_text_block` ignores whitespace-only text nodes (`# TODO should still support whitespace-only`). |
| List-item threading across pages | [Lists — page breaks and continuation](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#lists) | **open** | Spec shows a single list item split across pages with nested `<thread>` on both list and `<text>`. `DoclangTextSerializer` explicitly skips multi-prov splitting for `ListItem`; only whole-item / list-group page splits are handled (`DoclangListSerializer`). |
| Referenced captions in element head | [Element head — `<caption>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#caption) | **partial** | Host-element `<caption>` in the head is serialized (`_serialize_floating_caption_head`). `_extract_caption` only handles element-head / group-wrapper captions, not full round-trip fidelity for all referenced-caption layouts. |

---

## Intentional or parameterized differences

Behaviour that diverges from the spec default but is deliberate or configurable via `DoclangParams`.

| Topic | Spec reference | Status | Notes |
| --- | --- | --- | --- |
| Virtual `<text>` in lists | [Lists — virtual text](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#lists) | **by design** | `use_virtual_text=True` (default) omits `<text>` when allowed. Set `use_virtual_text=False` for explicit wrappers (`test_virtual_text_*`). |
| Root `xmlns` / `version` | [`<doclang>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#doclang) | **by design** | `include_namespace=False`, `include_version=False` by default. Enable with `include_namespace` / `include_version` (`test_include_namespace_and_version`). |
| Unordered list `class` | [`<list>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#list) | **by design** | Unordered lists emit bare `<list>`; `class="ordered"` only when ordered. Equivalent to default unordered per spec. |
| Heading level 1 omission | [`<heading>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#heading) | **by design** | Level 1 emitted as `<heading>` without `level` attribute (`_create_level_open_token`). |
| `TitleItem` ↔ `<heading>` | [`<heading>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#heading) | **by design** | `TitleItem` serializes as `<heading>` (level 1). Deserializer maps level 1 back to `TitleItem`, level ≥ 2 to `SectionHeaderItem` (`_parse_heading`). |
| Hyperlinks in inline scope | [Element head — `<href>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#href) | **by design** | `include_href=False` inside inline groups (`is_inline_scope`); `serialize_hyperlink` returns plain text. |
| Whitespace preservation | [`<content>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#content) | **by design** | Leading/trailing whitespace and newlines wrapped in `<content>` (`WrapMode`) instead of `xml:space="preserve"`. |
| Suppress empty elements | — | **by design** | `suppress_empty_elements` omits items with no text and no location (`test_suppress_empty_elements`). |
| Content filtering | — | **by design** | `add_content`, `content_types`, `layers`, `add_location`, `add_page_break`, `image_mode`, etc. control output subset. |
| Code `UNKNOWN` label | [Appendix B — code labels](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#code) | **by design** | Maps to `undefined` by default; `interpret_code_unknown_as_other=True` maps to `other`. |
| Table cell locations | [`<location>`](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#location) | **by design** | `add_table_cell_location=False` by default. |
| Picture classification labels | [Appendix B — pictures](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#pictures) | **by design** | Docling `OTHER` → DocLang `other`; unmapped Linguist keys → `other` on export, `UNKNOWN` on import when unrecognized. |

---

## Spec future extensions (not in v0.5 scope)

Listed in [Appendix C](https://github.com/doclang-project/doclang/blob/v0.5.0/spec.md#appendix-c-future-extensions) or commented out in the spec; not implemented.

| Topic | Status | Notes |
| --- | --- | --- |
| Horizontal threading (`<h_thread>`) | **not planned for v0.5** | Commented split-table example in spec; no token in `DoclangToken`. |
| Rich document `<head>` metadata (`title`, `author`, `date`, `language`, …) | **not planned for v0.5** | Appendix C; only `default_resolution` partially supported today. |

---

## Covered continuation / threading scenarios

These areas are implemented and tested (reference validator when installed). Listed here
because they are easy to confuse with gaps above.

| Scenario | Tests |
| --- | --- |
| Multi-prov text, same page (cross-column paragraph) | `test_multi_prov_text_emits_thread`, `test_cross_column_paragraph_roundtrip` |
| Cross-page paragraph | `test_cross_page_paragraph_*` |
| Cross-page list (whole items per page) | `test_cross_page_list_*` |
| Cross-page table | `test_cross_page_table_*` |
| Cross-column list (separate items, same page) | `test_cross_column_list_*` — no `<thread>` (each item is its own component, not a split fragment) |
