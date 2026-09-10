# dclq implementation plan

Status: proposed

Distribution, import package, and executable: `dclq`

## 1. Summary

`dclq` is a deterministic, read-only CLI for querying DocLang documents. It
combines:

- the original DocLang XML tree for XPath evaluation and durable source
  addresses;
- a `docling-core` `DoclingDocument` for hierarchy, reading order, semantic
  items, tables, lists, fields, pages, layers, provenance, and serialization.

Every public source address is XPath. Docling JSON pointers remain internal.
The core workflow is:

```text
inspect or outline a document
    -> grep or list semantic units
    -> receive bounded results with XPath addresses
    -> retrieve a section or exact source selection
```

`dclq` retrieves evidence. It does not convert documents, mutate content,
answer questions, or summarize whole documents.

## 2. Goals

The implementation must:

1. query large DocLang documents without returning the whole document;
2. operate on semantic units rather than flattened XML text;
3. return reusable XPath addresses for every result;
4. expose heading, list, table, field, picture, page, and layer structure;
5. provide deterministic structural context around matches;
6. support exact XPath selection when semantic normalization is undesirable;
7. keep human and machine-readable output bounded and pipeline-safe;
8. reuse `docling-core` instead of creating another document model;
9. remain local, read-only, and free of model calls.

## 3. Non-goals

`dclq` does not:

- convert PDF, DOCX, HTML, Markdown, images, or other source formats;
- mutate DocLang XML or `DoclingDocument` content;
- answer questions, generate summaries, or create embeddings;
- expose Docling JSON pointers as public addresses;
- fetch remote `href` or `src` resources;
- provide a predicate DSL in addition to XPath;
- retrieve page images or provenance crops;
- return unlimited output by default.

## 4. Package and dependency boundary

The package lives at `packages/dclq` in the `docling-core` workspace and is
released in lockstep with `docling-core`.

- Distribution: `dclq`
- Import package: `dclq`
- Console script: `dclq = "dclq.cli:main"`
- Runtime dependency: `docling-core[dclq]==<same version>`
- Build output: `dist/dclq/`

The `dclq` extra owns Typer, Click, DocLang, and lxml. The CLI may use the
standard library and dependencies already provided by that extra, but it must
not pull in agent, MCP, server, or model dependencies.

Implement one Typer application with explicit subcommands. Do not add an
`argparse` parser, manual compatibility dispatcher, or alternate executable.

## 5. Document model

One loaded input contains:

```text
raw document.xml bytes
    -> safe lxml tree
    -> DoclingDocument
    -> bidirectional DocLang source map
    -> ordered semantic units
```

### 5.1 XML authority

The XML view is authoritative for:

- the SHA-256 document identity;
- XPath evaluation and generated absolute XPath addresses;
- raw XML output;
- physical thread fragments;
- archive members and source element types.

XPath is evaluated with lxml over the same bytes passed to the semantic
deserializer. This is one source parsed for two purposes, not two competing
document representations.

### 5.2 Semantic authority

The `DoclingDocument` is authoritative for:

- hierarchy and reading order;
- semantic item types and text projection;
- list, table, field, picture, caption, and footnote relationships;
- table grids, header roles, and spanning cells;
- page and content-layer filtering;
- logical thread reconstruction;
- serializers and provenance.

Call package-local `DoclingDocument._hierarchize()` after deserialization.
Hierarchy normalization must preserve item refs so source bindings remain
valid.

### 5.3 Source map

Consume the optional `DocLangSourceMap` produced during DocLang
deserialization. Do not reconstruct bindings by comparing text: repeated text,
virtual list and table text, whitespace normalization, and threads make that
ambiguous.

Maintain both directions:

```text
XPath -> semantic target
semantic target -> one or more XPath addresses
```

Required bindings:

| DocLang construct | Semantic target | Public address |
| --- | --- | --- |
| Ordinary element | `DocItem` | Element XPath |
| Heading | title or section-header item | Heading XPath |
| List | list group | `list` XPath |
| List item | list item | Starting `ldiv` XPath |
| Table or index | table item | Container XPath |
| Table cell | table origin coordinates | Origin OTSL marker XPath |
| Field element | field item | Element XPath |
| Picture | picture item | Picture XPath |
| Thread fragments | one logical item | Ordered physical XPaths |
| Page break | page number | Page boundary |

Every searchable semantic item must have a source binding or a documented
derived anchor. Unbound selected XML nodes fall back to raw XML in `show`.

## 6. Inputs and safety

Supported inputs:

- `.dclg` and `.xml` DocLang XML;
- `.dclx` archives containing exactly one `document.xml`;
- `-` for DocLang XML on standard input;
- multiple inputs for `grep`, `list`, and `inspect`.

For `.dclx`, read members directly or through a private temporary directory.
Never extract to a user-controlled path.

The loader must:

- disable external XML entities and network resolution;
- require a DocLang root and fail on malformed XML;
- reject archive path traversal and ambiguous main documents;
- enforce XML, ZIP member, total archive, result, and output size limits;
- never fetch remote resources;
- allow `--validate` to run DocLang XSD and Schematron validation.

Predefine `d = https://www.doclang.ai/ns/v0` for XPath. Accept shorthand XPath
without the namespace prefix or `/d:doclang` root, but always emit canonical
namespace-qualified addresses.

## 7. Command surface

```text
dclq grep PATTERN INPUT... [OPTIONS]
dclq grep {-e PATTERN | -f FILE}... INPUT... [OPTIONS]
dclq list INPUT... [OPTIONS]
dclq fields INPUT [OPTIONS]
dclq inspect INPUT... [OPTIONS]
dclq outline INPUT [OPTIONS]
dclq show INPUT XPATH [OPTIONS]
dclq select INPUT XPATH [OPTIONS]
```

Use typed `Annotated` Typer arguments and `Literal` choices so invalid formats,
scopes, layers, ranges, and missing arguments fail before document loading.
Support `-h` and `--help`, disable completion, keep exception output concise,
and expose `main(argv=None) -> int` for the console entry point and tests.

The top-level help and README must identify `dclq` as experimental and state
that functionality may change incompatibly without prior warning.

## 8. Semantic units and projection

The ordered unit sequence contains the smallest useful semantic units:

| Content | Unit and address |
| --- | --- |
| Heading | Heading item and heading XPath |
| Paragraph, code, formula, caption, footnote | Corresponding item XPath |
| List content | Logical list item and starting `ldiv` XPath |
| Table content | Origin cell and OTSL marker XPath |
| Field content | Key, value, hint, or checkbox XPath |
| Picture content | Nested semantic item or picture XPath |
| Threaded content | Joined logical item with all physical XPaths |

Do not search containers in addition to their children by default. Explicit
`--type` selection may return containers. Collapse nested inline matches into
their rendered parent for `grep`; keep the exact selected unit for `show`.

The default `visible` projection includes document content and excludes
descriptions, summaries, custom metadata, links, sources, furniture, and
background layers. `metadata` searches descriptions, summaries, and textual
metadata. `all` combines both.

Use Docling serializers and table APIs. Do not flatten XML with `string(.)`.

## 9. `grep`

`grep` searches projected semantic text with regular expressions by default.

```bash
dclq grep 'termination' contract.dclx
dclq grep -i -e 'termination' -e 'cancellation' contract.dclx
dclq grep -F 'gross margin' report.dclg
dclq grep -C 2 --context-scope section 'liability' contract.dclx
```

Core options:

```text
-e, --regexp PATTERN
-f, --file FILE
-F, --fixed-strings
-i, --ignore-case
-w, --word-regexp
-A, --after-context N
-B, --before-context N
-C, --context N
--context-scope auto|container|section|document
--type TYPE
--layer body|furniture|background|all
--page LIST
--within-xpath XPATH
--section
--view visible|metadata|all
--offset N
--limit N
--max-chars N
--max-output-chars N
--all
-q, --quiet
-c, --count
-l, --files-with-matches
-n, --with-xpath
--format text|json|jsonl
--validate
```

Repeated `-e` and patterns read through `-f` are ORed. `-F`, `-i`, and `-w`
apply to every pattern. `-f -` conflicts with a document read from standard
input. `--page` accepts one-based numbers and inclusive ranges such as
`2-4,7`; selection remains in document order.

Multiple matches in one unit produce one result with multiple zero-based,
end-exclusive spans. Deduplicate units that resolve to the same logical item.
Apply `--offset` and `--limit` after filtering and deduplication.

`-q`, `-c`, and `-l` are mutually exclusive. Summary modes apply all search
filters but do not render result text or context. Quiet mode still reports
input errors through exit status `2`.

## 10. `list`

`list` enumerates semantic units without a pattern and reuses the same ordered
unit pipeline as `grep`.

```bash
dclq list report.dclx --type table_cell --page 3
dclq list report.dclg --offset 40 --limit 10
```

Support the non-pattern options from `grep`: semantic filters, bounds, summary
modes, XPath prefixes, output format, and validation. Reject pattern and
context options.

Exit `0` when at least one unit is selected, `1` for an empty selection, and
`2` on error.

## 11. Structural context

`-A`, `-B`, and `-C` count semantic units rather than physical lines. The hit
does not consume a context position. Preserve command-line option ordering:
a later `-A` or `-B` overrides its side, while a later `-C` resets both.

Scopes:

- `document`: global semantic reading order;
- `section`: reading order inside the nearest heading subtree;
- `container`: siblings under the immediate semantic parent, table cells in
  the same table, or list items in the same list;
- `auto`: container scope for table cells and list items, section scope for
  ordinary content, then document fallback.

Page targeting belongs to `--page`; page boundaries are not context scopes.

Search filters determine which units may match, not which neighbouring unit
types may appear. `--within-xpath`, `--section`, `--page`, and `--layer` still
bound the context sequence.

Text output merges overlapping context windows and emits each unit once. A
standalone `--` separates only disjoint groups. JSON and JSONL retain ordered
`before` and `after` arrays per result.

Attach compact structural metadata without consuming context positions:

- heading ancestry and semantic parent for ordinary items;
- list ancestry for list items;
- row and column headers, spans, and caption for table cells;
- key, value, hint, region, and heading ancestry for fields;
- captions, footnotes, and section ancestry for floating items;
- physical fragments and pages for threaded items.

## 12. `fields`

`fields` lists field slots as paired, addressable records:

```text
dclq fields INPUT [--fillable] [--empty]
                  [--checkbox selected|unselected|all]
                  [-n] [--format text|json|jsonl] [--validate]
```

Walk semantic field items in document order. Ownership of keys, values, hints,
and checkboxes is determined by the nearest field-item ancestor so nested
field items do not leak into their parents.

A field item's slots, in document order, are:

1. its owned values;
2. owned checkboxes that are not already inside an owned value.

Emit one row per slot and repeat the key for multi-value fields. A field item
with no value emits a row addressed at its key. A keyless field emits an empty
key. A value that contains a nested field region or field item is a container,
so only its nested fields emit rows at `depth + 1`.

Associate each hint with the preceding owned key or value in document order.
Render checkboxes as `[ ] label` or `[x] label`. Ignore field headings and
markers in the row stream.

Every machine record contains:

```json
{
  "document": "form.dclg",
  "key": "name",
  "value": "",
  "kind": "fillable",
  "checkbox": null,
  "empty": true,
  "hints": ["Enter your first and last name"],
  "depth": 1,
  "key_xpath": "/d:doclang/d:field_region[1]/d:field_item[2]/d:key[1]",
  "value_xpath": "/d:doclang/d:field_region[1]/d:field_item[2]/d:value[1]",
  "pages": [1]
}
```

The filter facts are independent:

| Fact | Values | Meaning |
| --- | --- | --- |
| `kind` | `fillable`, `read_only`, or `null` | Value class; bare checkboxes use `null` |
| `checkbox` | `selected`, `unselected`, or `null` | Checkbox state |
| `empty` | boolean | Slot has no text and no content |

A checkbox is content and therefore is never empty. `--fillable` selects
`kind == "fillable"` or a checkbox slot. `--empty` selects structurally empty
slots regardless of kind. `--checkbox` selects checkbox state. Combine filters
with AND.

Implement `fields` directly over field units rather than the general display
unit collapse, which may render an inline field as its enclosing paragraph.

## 13. Retrieval commands

### 13.1 `show`

`show` evaluates XPath, resolves selected nodes through the source map,
deduplicates logical targets, and serializes each selected unit independently.

```bash
dclq show report.dclx '/heading[4]'
dclq show report.dclx '/heading[4]' --section
dclq show report.dclx '/table[3]/fcel[7]' -C 1 -n
```

`--section` selects the heading and all canonical semantic units in its
subtree, preserving each unit's own XPath. `--raw` returns selected XML.
Context options, result bounds, XPath prefixes, formats, and validation match
the query commands where applicable.

### 13.2 `select`

`select` evaluates XPath against the original XML. Node selections remain in
document order; string, number, and boolean expressions return scalars.

```bash
dclq select report.dclx '//picture[@class="chart"]'
dclq select report.dclx 'count(//page_break) + 1' --format json
```

Support semantic binding details, result bounds, `xml`, `text`, `json`, and
`jsonl` formats, and validation.

### 13.3 `outline`

`outline` emits titles and section headings with depth and XPath. Use
`OutlineDocSerializer`, translate Docling refs through the source map, and
support bounded depth plus text and JSON output.

### 13.4 `inspect`

`inspect` emits a bounded inventory: input type, page count, semantic-unit
count, element counts, metadata counts, archive assets, and source-map
coverage. It reports structure, never complete document content.

## 14. XPath and document identity

Generated XPath addresses are absolute, namespace-qualified with `d`, based
on element siblings only, and positional among same-name siblings:

```xpath
/d:doclang/d:heading[1]
/d:doclang/d:list[2]/d:ldiv[3]
/d:doclang/d:table[3]/d:fcel[7]
/d:doclang/d:field_region[2]/d:field_item[4]/d:value[1]
```

Logical units with several source fragments carry ordered `xpaths` and a
cardinality. Section expansion never creates a synthetic section address; its
members keep their own XPath.

Machine-readable results include SHA-256 of the raw `document.xml` bytes so a
positional XPath cannot silently be reused against another document version.
The identity is the same for identical XML inside or outside a `.dclx` archive.

## 15. Output and exit codes

Default bounds:

```text
maximum results:              20
maximum text per result:      2,000 characters
maximum total textual output: 20,000 characters
default layer:                body
default view:                 visible
```

`--all` removes convenience defaults but never hard parser, archive, memory,
or output caps. Mark truncation in machine output and with an ellipsis in text.

Normal single-input text output contains semantic content only. It does not
print standalone metadata headers. With `-n`, shorten XPath prefixes by
removing `/d:doclang` and `d:`. Use `:` for a match or selected unit and `-`
for context. Prefix multi-input results with their source filename.

Highlight only matched substrings on a capable TTY. Never emit ANSI escapes
when redirected, in machine output, or when `NO_COLOR` is set.

JSONL emits one complete record per line. Diagnostics go to standard error.

Exit codes:

- `0`: at least one result or successful scalar operation;
- `1`: no query or listing results;
- `2`: usage, input, parse, XPath, validation, or runtime error.

Errors remain strict in quiet mode.

## 16. Implementation sequence

### Phase 1: package and loader

- Add the `dclq` workspace package, extra, console script, build, release, and
  pre-commit wiring.
- Implement safe `.dclg`, `.xml`, `.dclx`, and standard-input loading.
- Parse one byte source into the XML and semantic views.
- Consume `DocLangSourceMap`, hierarchize, and build ordered semantic units.

Exit criterion: representative headings, lists, cells, fields, pictures, and
threads round-trip from XPath to semantic targets and back without text
matching.

### Phase 2: core queries

- Implement `grep` and `list` on the shared unit pipeline.
- Add filters, matching modes, bounds, context, summary modes, and output
  rendering.
- Implement `show`, `select`, `outline`, and `inspect` using the same loader and
  canonical addresses.

Exit criterion: an agent can inspect, locate, retrieve, and expand bounded
evidence without receiving the full document.

### Phase 3: structural records

- Add table header, span, caption, list, field, picture, and thread context.
- Implement `fields` with nearest-owner slot pairing and independent filters.
- Report source-map coverage in `inspect`.

Exit criterion: table, list, field, picture, and threaded results retain useful
type-specific context and reusable XPath addresses.

### Phase 4: release validation

- Run focused package and core tests.
- Build both wheels and install them together in a clean environment.
- Verify imports, CLI help/version, lockstep dependency metadata, and artifacts.

Exit criterion: the locally built `dclq` wheel installs only with the matching
`docling-core` wheel and runs successfully.

## 17. Test plan

### 17.1 Source-map invariants

Test that:

- every emitted XPath selects its intended source node;
- reverse bindings survive hierarchization;
- repeated text never affects identity;
- namespace-prefix differences produce equivalent canonical XPath;
- virtual list and cell text uses marker anchors;
- threaded fragments map many-to-one in source order;
- rich and spanning cells map to origin coordinates.

### 17.2 Query behavior

Test:

- command help, version, missing arguments, and invalid choices;
- regex, fixed, case-insensitive, whole-word, `-e`, and `-f` matching;
- standard input and multiple inputs;
- type, layer, page, XPath, section, and view filters;
- deterministic ordering, deduplication, offset, and limit;
- quiet, count, filename-only, and grep-compatible exit statuses;
- exact context counts, flag precedence, scope boundaries, and merged windows;
- text, JSON, and JSONL bounds and schemas;
- XPath prefixes, multi-input filenames, TTY color, and `NO_COLOR`;
- invalid XML, XPath, archives, and validation failures.

### 17.3 Retrieval and structure

Test:

- XPath node and scalar selection;
- exact `show`, raw XML, sections, inline units, and table containers;
- heading outline depth and source addresses;
- structural inventory and source-map coverage;
- table row and column headers, captions, and merged cells;
- nested lists, fields, pictures, captions, and threaded content.

### 17.4 Fields

Test:

- one row per value with repeated keys;
- keyless, valueless, picture-key, inline, and nested fields;
- nearest-owner scoping for nested field items;
- hint association in document order;
- all supported checkbox placements with the same record shape;
- checkbox state independent from structural emptiness;
- `--fillable`, `--empty`, and `--checkbox` alone and combined;
- value and key XPath round-trips through `show`;
- exit `1` when no field slots exist.

Prefer existing DocLang and `docling-core` fixtures. Add one compact fixture
only when existing data cannot cover an end-to-end invariant.

## 18. Verification

```bash
uv sync --all-extras --all-packages
uv run pytest packages/dclq/tests test/
uv run pre-commit run --all-files
uv build --package docling-core --out-dir dist/docling-core
uv build --package dclq --out-dir dist/dclq
uv pip install dist/docling-core/*.whl dist/dclq/*.whl
python -m dclq --help
```

The release workflow must publish `docling-core` first and `dclq` second
because `dclq` pins the matching core version exactly.

## 19. Acceptance workflow

```bash
dclq inspect contract.dclx --format json

dclq outline contract.dclx --depth 3 --format json

dclq grep -i 'termination|cancellation' contract.dclx \
  --type heading,text,list_item,table_cell,footnote \
  -C 2 --limit 12 --format json

dclq show contract.dclx '/heading[2]' --section --max-chars 5000

dclq list contract.dclx --within-xpath '/heading[2]' --section -n

dclq fields form.dclx --empty --format jsonl
```

Every returned evidence record must be bounded, deterministic, and reusable
through its XPath address without a model call or complete-document prompt.
