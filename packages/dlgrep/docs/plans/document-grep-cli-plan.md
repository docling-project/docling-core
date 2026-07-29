# Document Grep CLI

Status: Implementation specification
Executable and package name: `dlgrep`
Target users: agents, developers, and document-processing pipelines

## 1. Summary

`dlgrep` is a deterministic, read-only CLI for locating and retrieving small,
structurally meaningful regions from DocLang documents.

It combines two complementary views of one document:

- the original DocLang XML tree, used for XPath selection and stable source
  addresses;
- a `docling-core` `DoclingDocument`, used for document hierarchy, semantic
  items, tables, lists, fields, pages, layers, provenance, serializers,
  chunking, and images.

Every public source address is XPath. Docling JSON pointers such as
`#/texts/7` are internal implementation keys and are not the CLI's durable
address format.

The primary workflow is:

```text
inspect structure
    -> grep semantic units
    -> receive bounded matches with XPath addresses
    -> retrieve structural context
    -> optionally retrieve the original XML or page region
```

The CLI retrieves evidence. It does not answer questions or summarize whole
documents.

## 2. Goals

The CLI must:

1. search large DocLang documents without returning the whole document;
2. search semantic units instead of flattened lines;
3. return reusable XPath addresses for every result;
4. expose heading, list, table, form, picture, page, and layer structure;
5. provide type-aware structural context around a match;
6. use existing `docling-core` functionality instead of reimplementing its
   document model;
7. support exact XML selection when semantic normalization is undesirable;
8. produce deterministic, bounded human and machine-readable output;
9. remain read-only and composable in shell and agent workflows.

## 3. Non-goals

The CLI does not:

- convert PDF, DOCX, HTML, Markdown, images, or other source formats;
- mutate DocLang XML or `DoclingDocument` content;
- answer questions or generate summaries;
- create or query a vector database;
- expose Docling JSON pointers as public source addresses;
- fetch remote `href` or `src` URIs;
- return unlimited output by default;
- replace general XML tools for arbitrary document mutation or transformation.

## 4. Existing foundations

The implementation should reuse the following behavior rather than copy it.

### 4.1 DocLang

DocLang provides:

- the XML vocabulary and namespace;
- headings and heading levels;
- semantic and structural elements;
- page boundaries through top-level `page_break` elements;
- page-relative geometry through `location` elements;
- body, furniture, and background layers;
- threads and cross-references;
- virtual text in list items and table cells;
- OTSL tables and indexes;
- `.dclx` archives containing `document.xml`, page images, and assets;
- XSD and Schematron validation;
- XML tooling dependencies, including `lxml` and Saxon/C.

### 4.2 docling-core

`docling-core` provides:

- DocLang deserialization into `DoclingDocument`;
- typed document items and parent/child relationships;
- heading hierarchization without replacing item `self_ref` values;
- semantic traversal with page and content-layer filters;
- serializers for individual items, subtrees, pages, tables, and documents;
- serialization span metadata identifying contributing document items;
- table grids, row and column spans, header roles, and rich cells;
- list, field, picture, caption, footnote, and nested-content models;
- thread reconstruction for text, lists, and tables;
- hierarchical chunks containing heading chains and contributing items;
- provenance and page-image cropping.

### 4.3 docling-agent

`docling-agent` demonstrates higher-level workflows that should be available
without requiring an LLM:

- creating JSON, Markdown, and indented document outlines;
- converting heading levels into an explicit document hierarchy;
- retrieving a heading subtree as a section;
- falling back to heading-boundary scanning for flat documents;
- serializing one page with rich table content;
- collecting subtree text;
- selecting bounded document regions before agent reasoning.

### 4.4 docling-mcp

`docling-mcp` demonstrates the agent-facing interaction model:

- inspect document structure;
- locate matching document anchors;
- retrieve one selected item;
- retain converted documents in a local cache;
- retrieve page thumbnails.

`dlgrep` should provide these retrieval primitives locally and deterministically,
without depending on the MCP server, cache, or tool protocol.

### 4.5 Runtime dependency boundary

The CLI has three direct runtime dependencies:

- `typer>=0.15.1,<0.25.0`, for the command surface, help, typed options, and
  validation;
- `doclang>=0.7,<0.8`, for validation and archive behavior;
- `docling-core>=2.87.1`, for semantic document behavior and source bindings.

It may reuse code patterns from `docling-agent` and `docling-mcp`, but it must
not pull their LLM, server, or integration dependencies into the runtime. Use
transitively available `lxml`, the standard library, and Pillow rather than
adding another CLI or document-model dependency.

Typer is part of the architecture, not an optional presentation layer. Do not
build a parallel `argparse` parser or manually dispatch every command.

### 4.6 Required docling-core extensions

Everything in §4.2 is consumed as-is and read-only. The design also needs a
small, clearly bounded set of additions inside `docling-core` itself, so the
CLI never reaches into deserializer internals:

1. Source-binding recorder at the deserialization boundary. An opt-in sidecar
   records, at each item and structural-unit creation point, the
   mapping from source element to the resulting `self_ref` (§18.1). When no
   recorder is supplied, deserializer behavior and output are unchanged. This
   boundary is the authoritative place to build the source map; the CLI only
   consumes it.

2. Origin identity for table cells. The OTSL table path re-serializes cell
   nodes and re-parses them from a string, so a resulting `TableCell` carries
   no reference to its source marker. Binding a cell to a reusable XPath
   without post-hoc text matching requires the deserializer to carry a stable
   origin identity (source element or origin index) through OTSL parsing into
   each `TableCell`. This is the one binding a CLI-local shim cannot
   reconstruct cleanly.

3. Package-local hierarchy reuse. Because dlgrep ships with docling-core, it
   calls `DoclingDocument._hierarchize()` directly rather than adding a public
   API solely for this package-local consumer.

The sidecar recorder is additive and does not affect ordinary deserialization.
Origin identity remains the gating dependency for table-cell addressing.

## 5. System model

One loaded input is represented by a `LoadedDocument` containing:

```text
raw bytes
    -> XML tree
    -> DoclingDocument
    -> bidirectional source map
    -> semantic indexes
```

### 5.1 XML view

The XML tree is the authority for:

- document hashing;
- XPath evaluation;
- absolute XPath generation;
- raw XML output;
- physical thread fragments;
- archive-relative `src` assets;
- structures not represented by `DoclingDocument`.

XPath is evaluated with `lxml` on a parallel parse of the same bytes, because
the semantic deserializer's parser does not provide XPath. This is one byte
source parsed for two purposes, not two competing authorities.

### 5.2 Semantic view

The `DoclingDocument` is the authority for:

- logical document items;
- hierarchy and section ancestry;
- reading-order traversal;
- page and layer filtering;
- table grids and cell roles;
- list and field relationships;
- logical thread reconstruction;
- semantic serialization;
- chunks and heading context;
- provenance and image crops.

### 5.3 Source map

The source map connects XPath addresses to Docling semantic targets. It must be
created during DocLang deserialization, when both the source node and resulting
semantic object are known.

Reconstructing this map later by matching text is not acceptable: repeated
text, whitespace normalization, virtual text, rich cells, and threads make
post-hoc matching ambiguous.

The internal target forms are:

```text
item target      = DocItem self_ref
group target     = GroupItem self_ref
cell target      = table self_ref + origin row + origin column
page target      = page number
fragment target  = item self_ref + physical fragment index
```

The implementation maintains both directions:

```text
XPath -> semantic target
semantic target -> one or more XPath addresses
```

Docling refs are not emitted in normal CLI output. A debug option may expose
them for diagnostics.

### 5.4 Required binding rules

| DocLang construct | Docling representation | Public address |
| --- | --- | --- |
| Ordinary semantic element | One `DocItem` | Element XPath |
| Heading | `TitleItem` or `SectionHeaderItem` | Heading XPath |
| `list` | `ListGroup` | `list` XPath |
| List item | `ListItem` | Starting `ldiv` XPath |
| Virtual list text | Text on `ListItem` | `ldiv` XPath plus match offsets |
| `table` or `index` | `TableItem` | Container XPath |
| Table cell | `TableCell` or `RichTableCell` | Origin OTSL marker XPath |
| Virtual cell text | Cell text | Origin marker XPath plus match offsets |
| Field region/item/key/value/hint | Corresponding field item | Element XPath |
| Picture | `PictureItem` | Picture XPath |
| Multiple thread fragments | One logical item | All physical XPaths and a logical selecting XPath |
| Page | `PageItem` and page-filtered items | Derived page-range XPath |
| Section | Heading subtree | Heading XPath as section anchor |
| Chunk | Multiple contributing items | Source XPaths of those items |

### 5.5 Hierarchy normalization

After source bindings are created, the CLI may hierarchize the
`DoclingDocument` using heading levels. Hierarchization changes semantic
parent/child relationships but preserves item refs, so the source map remains
valid.

The original XML tree is never rearranged.

## 6. Inputs

Supported inputs:

- `.dclg` DocLang XML;
- `.xml` containing DocLang XML;
- `.dclx` DocLang archive;
- `-` for DocLang XML on standard input;
- multiple input paths for search and inspect operations.

For `.dclx`, the loader reads:

- `document.xml` as the document;
- `pages/{N}.{png|jpg|jpeg|webp}` as optional page images;
- archive-relative files under `assets/` for local picture assets.

The loader must not extract an archive to a user-controlled filesystem path.
It should read required members directly or use a private temporary directory.

### 6.1 Namespace handling

The CLI predefines:

```text
d = https://www.doclang.ai/ns/v0
```

Returned XPath expressions always use `d`, regardless of the prefix used in
the source document.

An unnamespaced DocLang document may be normalized in an internal XML copy.
The source hash and raw output still refer to the original `document.xml`
bytes.

### 6.2 Validation

Default behavior:

- parse safely;
- verify that the root is DocLang;
- fail on malformed XML or malformed `.dclx` structure;
- do not run full XSD and Schematron validation on every query.

`--validate` runs DocLang XSD and Schematron validation before the requested
operation.

### 6.3 Security limits

The loader must:

- disable external XML entity and network resolution;
- reject archive path traversal;
- enforce configurable XML and ZIP member size limits;
- reject ambiguous archives with multiple main document parts;
- never fetch remote `src` or `href` URIs;
- keep output bounds in force even for broad XPath expressions.

## 7. Command surface

```text
dlgrep [OPTIONS] PATTERN INPUT...
dlgrep [OPTIONS] {-e PATTERN | -f FILE}... INPUT...
dlgrep search [OPTIONS] PATTERN INPUT...
dlgrep show INPUT XPATH
dlgrep select INPUT XPATH
dlgrep outline INPUT
dlgrep inspect INPUT...
dlgrep image INPUT XPATH
```

Implement this surface as one Typer application:

- create `app = typer.Typer(...)` with completion disabled, predictable
  exception rendering, and both `-h` and `--help`;
- register `search`, `show`, `select`, `outline`, and `inspect` with
  `@app.command`; reserve `image` for the deferred visual iteration;
- use `typing.Annotated` Typer arguments/options and `Literal` choices so
  invalid formats, scopes, layers, ranges, and missing arguments fail before
  document loading;
- use a small `typer.core.TyperGroup` subclass, following Docling's CLI
  pattern, that inserts `search` when the first token is not a known
  subcommand or a top-level help flag;
- keep `search` available explicitly even though its command name may be
  omitted;
- preserve command-line ordering of `-A`, `-B`, and `-C` with Typer option
  callbacks because later context flags override earlier ones;
- expose `main(argv=None) -> int` for the console entry point and tests, and
  use `typer.Exit` to retain grep-compatible process statuses.

`search`, `show`, `select`, `outline`, and `inspect` are reserved first tokens;
`image` becomes reserved when that deferred command is registered. Use `-e`
for a literal search pattern equal to a reserved subcommand name. Do not add a
second parser or a separate compatibility CLI for bare search syntax.

## 8. Search

### 8.1 Basic usage

```bash
dlgrep 'termination' contract.dclx
dlgrep -i 'terminate|cancellation' contract.dclx
dlgrep -F 'gross margin' report.dclg
dlgrep -i -e 'termination' -e 'cancellation' contract.dclx
dlgrep -w 'margin' report.dclg --page 2-4,7
dlgrep -C 30 'force majeure' contract.dclx
dlgrep -B 5 -A 10 'net income' report.dclx
dlgrep -l 'termination' contracts/*.dclx
```

Search uses regular expressions by default. Supply either one positional
pattern or one or more `-e` and `-f` options.

Core options:

```text
-e, --regexp PATTERN      add a search pattern; repeatable
-f, --file FILE           read search patterns, one per line
-F, --fixed-strings       fixed-string search
-i, --ignore-case         case-insensitive search
-w, --word-regexp         require whole-word matches
-A, --after-context N     include N following semantic elements
-B, --before-context N    include N preceding semantic elements
-C, --context N           include N preceding and N following semantic elements
--type TYPE[,TYPE...]     restrict semantic unit types
--layer LAYER             body, furniture, background, or all
--page LIST               restrict to pages and inclusive ranges
--within-xpath XPATH      restrict source scope
--section                 expand --within-xpath heading to its section
--view VIEW               visible, metadata, or all
--physical                do not logically join threaded fragments
--offset N                skip the first N ordered results
--limit N                 maximum hits
--max-chars N             maximum characters per returned unit
--max-output-chars N      maximum total textual output
--all                     remove hit/output defaults, subject to hard safety caps
-q, --quiet               suppress normal output
-c, --count               count matching semantic units per input
-l, --files-with-matches  emit only inputs containing matches
-n, --with-xpath          prefix text output with XPath addresses
--format FORMAT           text, json, or jsonl
```

Repeated `-e` options and patterns read through `-f` are ORed. An empty pattern
file contributes no patterns. `-f -` reads patterns from standard input and is
invalid when `-` is also an input document. `-F`, `-i`, and `-w` apply to every
pattern. Word constituents for `-w` are Unicode alphanumeric characters and
underscore.

`--page` accepts a comma-separated set of one-based page numbers and inclusive
ranges, for example `2-4,7,10-12`. Zero, descending ranges, and empty members
are errors. Selection order remains document order rather than option order.

### 8.2 Search units

Search operates on the smallest useful semantic unit that can contain the
complete match:

| Match location | Search-result unit |
| --- | --- |
| Heading | Heading item |
| Paragraph or ordinary text | Text item |
| Footnote | Footnote item |
| Caption | Caption item |
| Code | Code item |
| Formula | Formula item |
| Field key/value/hint | Respective field item |
| List-item content | Logical list item anchored by `ldiv` |
| Table or index content | Origin cell anchored by its OTSL marker |
| Picture semantic child | Nested semantic item |
| Thread fragment | Joined logical item by default |

Containers such as an entire table, list, picture, or field region are not
searched in addition to their children by default because that would duplicate
matches. They become search units when explicitly selected with `--type`.

Inline units covered by a rendered parent are likewise not separate default
search units. Explicit type-filtered search returns the complete parent unit.
`show` remains exact: an inline XPath returns that selected unit, while a
context option includes its containing rendered unit without consuming a
numbered neighbour. Thus `-C 0` adds only that structural parent.

### 8.3 Text projection

The default `visible` view includes:

- headings and ordinary text;
- captions and footnotes;
- list markers and list-item content;
- table and index cell content;
- field headings, keys, values, hints, and checkboxes;
- formulas and code;
- chart tabular content;
- semantic content nested in pictures.

It excludes:

- descriptions and summaries;
- custom metadata;
- `href` and `src` URIs;
- page headers and footers unless explicitly requested;
- furniture and background layers.

The `metadata` view searches descriptions, summaries, and custom textual
metadata. The `all` view combines visible content and metadata.

Use Docling item serializers and table cell APIs for semantic text projection.
Do not flatten the original XML with `string(.)`, which would mix element-head
metadata with visible content.

### 8.4 Threads

Threaded fragments with the same host type and thread ID form one logical
search unit by default.

A logical result contains:

- a logical XPath selecting all physical host elements;
- the ordered physical XPaths;
- fragment count;
- page list;
- joined semantic text;
- match offsets in the joined text.

`--physical` searches and returns each fragment independently.

### 8.5 Matching and offsets

Match offsets are zero-based, end-exclusive offsets in the returned unit's
normalized semantic text. Offsets are metadata, not part of the XPath.

For rich serialization where exact source character offsets are unavailable,
offsets refer to serialized result text and must be labelled accordingly.

### 8.6 Ordering and deduplication

Inputs are processed in command-line order. Results within each input are
ordered by semantic document reading order. Table cells are ordered by origin
row and column. Multiple matches in one unit produce one result with multiple
match spans.

If several selected XPaths resolve to the same logical Docling item, the
logical result is emitted once unless `--physical` is active.

`--offset` and `--limit` apply to the combined ordered result sequence after
filtering and deduplication. They do not change the context attached to a
selected result.

### 8.7 Summary output modes

`-q`, `-c`, and `-l` are mutually exclusive:

- `-q` emits no standard output, scans every requested input, and preserves
  strict error reporting;
- `-c` emits the number of matching semantic units for each input; multiple
  match spans in one unit count once;
- `-l` emits each matching input once, in command-line order.

These modes apply all search and scope filters but do not render result text or
context. Combining them with `-A`, `-B`, `-C`, `--context-scope`, `--offset`,
`--limit`, `--max-chars`, `--max-output-chars`, or `--all` is an option error.
Text, JSON, and JSONL output remain available for `-c` and `-l`; `-q` never
emits a result or summary object. Convenience hit and textual-output defaults
do not cap these modes, but hard safety caps remain. Diagnostics still go to
standard error.

## 9. Element neighbourhood and structural context

The grep-style context flags are first-class search options. They operate like
GNU grep's context flags, except that the unit is a semantic document element
rather than a line.

### 9.1 Context options

```text
-A, --after-context N     N following semantic elements
-B, --before-context N    N preceding semantic elements
-C, --context N           N preceding and N following semantic elements
--context-scope SCOPE     auto, container, section, page, or document
--no-ancestors            omit heading/container ancestry
```

`-C 30` therefore returns up to 30 elements before and 30 elements after each
matching element. `-B 5 -A 10` returns up to 5 before and 10 after. The match
itself is not counted.

Context flags accept non-negative integers. When flags are combined, `-C N`
sets both sides and a later `-A` or `-B` overrides its respective side. A later
`-C` resets both sides.

Every result record includes compact heading/container ancestry by default.
Normal text output omits that metadata; JSON and JSONL retain it. Neighbour
elements are included only when `-A`, `-B`, or `-C` is requested.

The default scope is `document`, matching grep's file-wide line sequence.
`--context-scope auto` opts into type-aware containment.

### 9.2 What counts as an element

The neighbourhood sequence uses the same canonical semantic units as search:
headings, paragraphs, list items, table origin cells, field keys/values/hints,
captions, footnotes, formulas, code items, pictures, and other leaf Docling
items in reading order.

Structural containers are not separately counted when their children are in
the sequence. This prevents a table, its row, and its cells from consuming
three context positions for the same content. Threaded fragments count as one
logical element unless `--physical` is active.

Every selected unit maps to its overlapping source or semantic span in that
sequence. For a nested inline unit, the span includes its represented parents;
for a structural container, the span covers its represented descendants. `-B`
counts backward from the start of the span and `-A` forward from its end.
Search coalesces inline matches into the rendered parent; exact `show` keeps
the selected unit and exposes that parent separately when context is requested.

Search filters determine which elements may match, but do not filter the
neighbourhood. For example, `--type heading -C 2` finds headings and returns
the two surrounding semantic elements even when those elements are paragraphs
or list items. Input-boundary filters such as `--within-xpath`, `--section`,
`--page`, and `--layer` still clamp the neighbourhood.

### 9.3 Neighbourhood scopes

- `document`: use global semantic reading order;
- `auto`: use the containing list for a list item, table cell order for a table
  cell, field region for a field item, and nearest section for ordinary items;
  fall back to the immediate container and then the document;
- `container`: use siblings under the immediate semantic parent;
- `section`: use semantic reading order inside the nearest heading subtree;
- `page`: use page-filtered semantic reading order.

The selected scope is a hard boundary. If only three elements exist before a
match, `-B 5` returns three; it never crosses that boundary to fill the quota.

### 9.4 Context grouping

Normal text output merges overlapping or touching context windows from the
same semantic sequence. Emit every semantic element once in document order and
promote every actual hit to a match even when it first appeared as another
hit's context. A standalone `--` separates only disjoint context groups; never
place it between adjacent semantic elements inside one group. Without context,
matching elements are emitted consecutively without a group separator.

With `-n` or `--with-xpath`, each physical output line is prefixed with its
short semantic XPath, omitting the `/d:doclang` root and `d:` namespace
prefixes. A colon separates the XPath from the directly matching or selected
element, while a hyphen separates the XPath from a context element:

```text
/text[16]-context before
/text[17]:matching text
/text[18]-context after
```

Document order indicates whether hyphen-prefixed context is before or after the
colon-prefixed match. Multiline semantic elements repeat the same prefix on
every physical line. When search receives multiple input documents, prepend
the document name as grep does; use the same colon or hyphen between all
locator fields.

Without `-n`, single-input normal output contains semantic text only.
Multi-input search still prefixes each physical line with its document name to
preserve source identity, using a colon for matches and a hyphen for context.
On a color-capable terminal, search highlights the matching substring;
redirected output has no ANSI escapes and `NO_COLOR` disables highlighting.

JSON and JSONL keep one result object per matching element. Each result has
ordered `before` and `after` arrays, so overlapping context may appear in more
than one machine-readable result even though normal text output merges it.
Every neighbour contains its XPath, type, page, and bounded semantic text.

### 9.5 Structural metadata

Element neighbourhood is distinct from type-aware structural metadata. The
following relationships are attached compactly and do not consume `-A`, `-B`,
or `-C` positions:

| Result type | Structural metadata |
| --- | --- |
| Ordinary text/code/formula | Heading chain and semantic parent |
| Heading | Ancestor heading chain and section identity |
| List item | Containing list and nested-list ancestry |
| Table/index cell | Table caption, row/column headers, and spanning information |
| Field key/value/hint | Containing field item and region, sibling key/value/hint, field heading ancestry |
| Picture | Caption, description/summary when requested, semantic children, section ancestry |
| Caption/footnote | Referencing floating item and section ancestry |
| Threaded item | All fragments, pages, and normal context of the merged semantic item |

### 9.6 Context implementation

Context should use existing Docling structures:

- heading parents after document hierarchization;
- `parent` and `children` references for ancestors and siblings;
- `iterate_items()` for bounded reading-order neighbors;
- `TableData.grid` for table context;
- field-region/item relationships for forms;
- floating-item captions, footnotes, and references;
- chunk metadata for heading chains and contributing items;
- provenance for pages and images.

Every returned context unit must be translated back through the source map and
include its XPath.

## 10. `show`

`show` resolves XPath-selected source nodes to logical Docling units.

```bash
dlgrep show report.dclx '/d:doclang/d:text[17]'
dlgrep show report.dclx '/d:doclang/d:list[2]/d:ldiv[3]'
dlgrep show report.dclx '/d:doclang/d:table[3]/d:fcel[7]' -C 1
dlgrep show report.dclx '/d:doclang/d:heading[4]' --section
```

Default behavior:

1. evaluate the XPath;
2. resolve selected source nodes through the source map;
3. deduplicate identical logical targets;
4. for `--section`, select the ordered canonical semantic units in every
   selected heading subtree and deduplicate overlapping selections;
5. serialize every selected logical unit independently through Docling;
6. attach structural context and source metadata to each result record;
7. enforce output bounds independently for each result.

Options:

```text
--raw                     return exact selected XML instead of semantic output
--section                 select ordered elements in a heading subtree
-A, -B, -C                include neighbouring semantic elements
--context-scope SCOPE     choose neighbor scope
--max-chars N             bound serialized text
-n, --with-xpath          prefix text output with XPath addresses
--format FORMAT           text, json, or jsonl
```

If a selected XML node has no semantic binding, `show` falls back to raw XML
and states that no Docling semantic target exists.

Section selection emits ordinary addressable records, including the heading
anchor, in semantic document order. It preserves every element's own XPath,
logical type, pages, contributing items, and structured table-cell context.
There is no aggregate section result. Until span context is implemented,
`show --section` rejects `-A`, `-B`, and `-C`.

## 11. `select`

`select` is the raw XML operation.

```bash
dlgrep select report.dclx '/d:doclang//d:picture[@class="chart"]'
dlgrep select report.dclx '/d:doclang//*[d:xref]'
dlgrep select report.dclx 'count(/d:doclang/d:page_break) + 1'
```

The `d` namespace is predefined. XPath node selections return nodes in document
order. Scalar string, number, and boolean results are returned as scalars.

Generated and documented locators use the XPath 1.0-compatible subset.
Supporting additional XPath versions may be added later without changing
generated addresses.

Options:

```text
--semantic                also report bound semantic target information
--limit N                 bound node results
--max-chars N             bound each serialized node
--all                     remove default result bounds, subject to hard caps
--format FORMAT           xml, text, json, or jsonl
```

## 12. `outline`

`outline` returns the hierarchized heading structure with XPath addresses.

```bash
dlgrep outline report.dclx
dlgrep outline report.dclx --depth 3 --format json
dlgrep outline report.dclx --include-summaries
dlgrep outline report.dclx --all-items
```

Default output contains titles and section headings. `--all-items` uses the
general Docling outline mode. `--include-summaries` includes existing metadata
summaries; it never generates new summaries.

Docling outline refs must be translated to XPath before output.

## 13. `inspect`

`inspect` returns a bounded structural inventory.

```bash
dlgrep inspect report.dclx
dlgrep inspect report.dclx --format json
```

The inventory includes:

- input type and document SHA-256;
- DocLang version and namespace;
- page count and available page images;
- titles and heading counts by level;
- text, list, list-item, table, index, cell, picture, code, formula, field,
  caption, and footnote counts;
- thread and physical-fragment counts;
- cross-reference and hyperlink counts;
- content counts by layer;
- archive asset counts;
- source-map coverage and unbound semantic/source counts.

The command reports structure only. It does not emit all document content.

## 14. `image`

Scope note: `image` is deferred beyond the initial text-retrieval core (see §19,
Iteration 3). Agent consumers of this CLI want text tokens that explain the
document, not images, and image extraction is better served by dedicated
Docling API facades than by a grep tool. The command is specified here for
completeness; it is not part of the first useful release.

`image` retrieves visual evidence associated with an XPath.

```bash
dlgrep image report.dclx '/d:doclang/d:table[3]/d:fcel[7]'
dlgrep image report.dclx '/d:doclang/d:picture[5]' --asset
dlgrep image report.dclx '/d:doclang/d:heading[4]' --page
```

Default resolution order:

1. a local picture `src` asset when the selected picture has one;
2. a provenance crop when a bound Docling item has geometry and a page image;
3. the containing page image.

Options:

```text
--asset                   require the referenced local picture asset
--crop                    require a provenance crop
--page                    return the containing page
--output PATH             write to a selected path
```

Remote URIs are reported but never fetched. If the required page image,
geometry, or asset is unavailable, the command fails with a precise reason.

## 15. XPath addresses

### 15.1 Generated XPath rules

Generated XPath expressions are:

- absolute;
- namespace-qualified with `d`;
- independent of source prefix choice;
- based on element siblings only;
- positional among same-name element siblings;
- anchored to real XML elements whenever possible.

Examples:

```xpath
/d:doclang/d:heading[1]
/d:doclang/d:text[17]
/d:doclang/d:list[2]/d:ldiv[3]
/d:doclang/d:table[3]/d:fcel[7]
/d:doclang/d:field_region[2]/d:field_item[4]/d:value[1]
/d:doclang/d:picture[5]/d:caption[1]
```

Do not generate paths based on `node()` positions or indentation text nodes.

### 15.2 Logical XPath

When a logical unit has several physical source elements, output contains:

- `xpath`: a canonical XPath selecting the logical source set when practical;
- `xpaths`: ordered physical absolute XPaths;
- `cardinality`: physical node count.

Example:

```xpath
/d:doclang//d:text[d:thread[@thread_id = 42]]
```

### 15.3 Section selection

A heading XPath combined with `--section` selects an ordered span of ordinary
logical units. The heading and each canonical descendant remain separate,
addressable results. Section selection does not derive a new logical unit or
replace any selected unit's `logical_type`.

### 15.4 Derived units

XPath anchors a derived logical unit:

- `ldiv` XPath plus list-item semantics;
- OTSL marker XPath plus cell semantics;
- page-range XPath plus page semantics.

The expansion mode is explicit in machine output as `logical_type`.

### 15.5 Document identity

Every machine-readable address includes SHA-256 of the raw DocLang
`document.xml` bytes. This prevents silently reusing positional XPath against a
different document version.

For `.dclg` and `.xml`, these are the input bytes. For `.dclx`, these are the
uncompressed `document.xml` bytes, so the same XML has the same identity inside
or outside an archive.

## 16. Output

### 16.1 Default bounds

Provisional defaults:

```text
maximum hits:                 20
maximum text per result:      2,000 characters
maximum total textual output: 20,000 characters
default layer:                body
default view:                 visible
thread fragments:             logically joined
page headers/footers:         excluded
```

Truncated output must say that truncation occurred. `--all` removes convenience
defaults but not hard parser, memory, archive, or output safety caps.

Opaque continuation cursors are not required initially. Deterministic
`--offset` and `--limit` are sufficient for a local CLI.

### 16.2 Human output

Normal text output is content-oriented. `search` and `show` must not print a
standalone document heading or `XPath:`, `Type:`, `Page:`, `Section:`, table,
or other metadata lines. That metadata belongs to JSON and JSONL.

Without `-n`, single-input `search` and `show` emit only the serialized
semantic text:

```text
The company reported gross margin of 48.2%.
```

Multi-input search prefixes the document name as grep does, because otherwise
the source of each result would be lost.

With `-n` or `--with-xpath`, use XPath as the semantic equivalent of grep's
line number. Omit the `/d:doclang` root and `d:` namespace prefixes to keep
human output compact. A colon marks the matching or explicitly selected unit
and a hyphen marks context. Repeat the prefix for every physical line of a
multiline unit:

```text
/table[3]/fcel[6]-Operating margin was 21.4%.
/table[3]/fcel[7]:The company reported gross margin of 48.2%.
/table[3]/fcel[8]-Net margin was 18.1%.
```

Human search output highlights only the matching substring when standard
output is a terminal, including when `-n` is absent. It must emit no ANSI
escapes when piped or redirected and must respect the `NO_COLOR` environment
variable. `show` has no search substring to highlight. Machine-readable output
is never colorized.

### 16.3 JSON result

```json
{
  "document": "report.dclx",
  "sha256": "f9c8...",
  "xpath": "/d:doclang/d:table[3]/d:fcel[7]",
  "xpaths": [
    "/d:doclang/d:table[3]/d:fcel[7]"
  ],
  "cardinality": 1,
  "logical_type": "table_cell",
  "page": 37,
  "text": "The company reported gross margin of 48.2%.",
  "matches": [
    {
      "start": 21,
      "end": 33,
      "text": "gross margin"
    }
  ],
  "context": {
    "before": [
      {
        "xpath": "/d:doclang/d:table[3]/d:fcel[6]",
        "logical_type": "table_cell",
        "page": 37,
        "text": "Operating margin was 21.4%."
      }
    ],
    "after": [
      {
        "xpath": "/d:doclang/d:table[3]/d:fcel[8]",
        "logical_type": "table_cell",
        "page": 37,
        "text": "Net margin was 18.1%."
      }
    ],
    "headings": [
      "Financial Results",
      "Margins"
    ],
    "table_caption": "Quarterly results",
    "row_headers": [
      "Q4 2025"
    ],
    "column_headers": [
      "Gross margin"
    ]
  },
  "truncated": false
}
```

JSONL emits one result object per line. Diagnostic summaries go to standard
error so standard output remains pipeline-safe.

### 16.4 Exit codes

Search follows grep conventions:

- `0`: at least one match;
- `1`: no matches;
- `2`: input, parse, XPath, validation, or runtime error.

`-q` does not hide input errors: the command scans every requested input, and
an error produces `2` even if another input matched.

Other subcommands return `0` on success and `2` on failure.

Typer/Click usage errors, including invalid choices, invalid numeric ranges,
missing arguments, and incompatible options, return `2`. Expected document
and query errors are rendered as concise diagnostics on standard error rather
than Typer tracebacks.

## 17. Determinism

Given the same document bytes and options, the CLI must produce the same:

- result order;
- generated XPath addresses;
- semantic text projection;
- context selection;
- snippets and truncation boundaries;
- JSON field values.

Do not use model calls, embeddings, random ranking, locale-dependent ordering,
or unstable object identities.

## 18. Implementation strategy

### 18.1 Preferred source-map integration

Add an optional source-binding recorder sidecar to DocLang deserialization. The
recorder records a binding at the point each item or structural unit is
created or merged.

Required hook points:

1. semantic element dispatch and item creation;
2. heading creation and thread merging;
3. `ldiv` to `ListItem` creation;
4. OTSL origin marker to `TableCell` creation;
5. field and picture item creation;
6. thread fragment merge into an existing item;
7. page-break advancement.

The ordinary deserializer API and output remain unchanged when no recorder is
provided.

Hook point 4 is the one that cannot be satisfied by observation alone. The OTSL
table path re-serializes cell nodes and re-parses them from a string, so a
`TableCell` currently retains no link to its source marker. Binding cells to
XPath without post-hoc text matching therefore requires the deserializer to
carry a stable origin identity through OTSL parsing. This is a docling-core
extension, not a CLI concern; see §4.6.

The source map lives at the deserialization boundary where the mapping is
authoritative; the CLI only consumes the completed map.

### 18.2 Reuse points

Use:

- one Typer application with typed `Annotated` arguments and options;
- Docling's default-command `TyperGroup` pattern for bare search syntax;
- DocLang validation and archive rules;
- `DocLangDocDeserializer` for semantic reconstruction;
- package-local `DoclingDocument._hierarchize()`;
- `DoclingDocument.iterate_items()` for semantic order and filters;
- `OutlineDocSerializer` for outlines;
- Markdown/HTML serializers for rich semantic output;
- `SerializationResult.spans` to retain contributing items;
- `HierarchicalChunker` metadata for heading context;
- `TableData.grid` and cell roles for table context;
- `DocItem.get_image()` for provenance crops;
- standard-library hashing, regex, ZIP, and path handling.

Do not build a second document hierarchy, table model, chunker, image cropper,
or custom locator language.

### 18.3 Internal search index

For one CLI invocation, construct an in-memory ordered list of searchable
semantic units. Each unit contains:

```text
semantic target
logical type
projected text
source XPath(s)
page(s)
layer
parent/container target
```

No persistent index or database is required. Add caching only if profiling
shows repeated deserialization dominates real agent workflows.

## 19. Iteration plan

### Iteration 0: source-map spike

Purpose: prove XPath and Docling semantics can coexist without text matching.

Deliverables:

- load representative `.dclg` fixtures;
- create bindings for ordinary text, headings, list items, table cells, fields,
  pictures, and threads;
- hierarchize the document and prove refs and bindings remain valid;
- resolve XPath to semantic target and semantic target back to XPath;
- document constructs that remain unbound.

Exit criteria:

- no source binding relies on comparing text;
- all returned XPaths reselect the intended source nodes;
- threaded fragments map many-to-one correctly;
- list virtual text maps to `ldiv`;
- table origin cells map to the correct grid positions.

### Iteration 1: first useful CLI

Deliverables:

- a Typer application with typed commands, generated help, `--version`, and
  bare-argument routing to the explicit `search` command;
- `.dclg`, `.xml`, `.dclx`, and stdin loading;
- safe parsing and document hashing;
- `select`, `inspect`, and `outline`;
- positional, repeated, and file-supplied regex and fixed-string patterns;
- case-insensitive and whole-word matching;
- semantic search units and visible-text projection;
- XPath output and JSON/JSONL;
- bounded results with deterministic offset/limit slicing;
- quiet, count, and matching-input output modes;
- grep exit codes and terminal-safe, content-only human output;
- `-n`/`--with-xpath` prefixes with grep-style colon and hyphen separators;
- `show` for ordinary items, lists, cells, fields, pictures, and sections;
- heading ancestry and `-A`, `-B`, `-C` structural context;
- page-list, layer, type, and within-XPath filters;
- logical text-thread joining.

Exit criteria:

- an agent can inspect, locate, retrieve, and expand evidence without receiving
  the full document;
- every search and context unit has an XPath address;
- structural context is obtained from Docling relationships rather than XML
  sibling lines;
- output remains bounded for broad searches.

### Iteration 2: rich structural context

Deliverables:

- table row/column header context and spanning metadata;
- adjacent cell navigation;
- nested-list context;
- field key/value/hint context;
- caption, footnote, and floating-item relationships;
- complete list and table thread handling;
- chunk-backed context where it improves large-section retrieval;
- source-map coverage diagnostics in `inspect`.

Exit criteria:

- table, list, field, and picture results provide useful type-specific context;
- context selection is deterministic and covered by focused fixtures;
- no feature exposes Docling refs as the only reusable address.

### Iteration 3: visual evidence (optional, deferred)

This iteration is out of scope for the initial text-retrieval core and may be
dropped. Text retrieval, not image extraction, is the value for agent
consumers, and images are better served by dedicated Docling API facades (see
§14).

Deliverables:

- attach `.dclx` page images to semantic pages;
- `image` page retrieval;
- provenance crops;
- local `src` asset extraction;
- thumbnail sizing and explicit output paths.

Exit criteria:

- page, crop, and asset modes fail clearly when prerequisites are absent;
- archive assets cannot escape the archive boundary;
- remote resources are never fetched.

## 20. Test plan

### 20.1 Source-map invariants

Test that:

- every bound XPath selects its source node;
- every searchable semantic item has a binding or documented derived anchor;
- reverse mappings survive hierarchization;
- repeated identical text does not affect identity;
- different source namespace prefixes produce equivalent generated XPaths;
- thread fragments map to one logical item in source order;
- virtual list and cell text use marker anchors;
- rich and spanning table cells map to origin markers.

### 20.2 Structural context

Focused fixtures should cover:

- exact `-A`, `-B`, and `-C` element counts at normal and bounded edges;
- combined context-flag precedence in command-line order;
- match-only type filters with unfiltered neighbouring element types;
- document, page, section, container, and auto scope boundaries;
- merged overlapping and touching human-readable neighbourhoods, disjoint
  `--` group separators, and per-match JSON windows;
- flat and hierarchized heading sections;
- text before and after peer headings;
- nested lists;
- OTSL column headers, row headers, sections, and merged cells;
- field regions containing keys, values, hints, and checkboxes;
- pictures with captions, descriptions, nested content, and sources;
- content on body, furniture, and background layers;
- multi-page and multi-column threads;
- page headers and footers.

### 20.3 CLI behavior

Test:

- top-level and subcommand `-h`/`--help`, `--version`, and explicit `search`;
- bare search routing through the default-command Typer group;
- Typer rejection of invalid choices, ranges, missing arguments, and unknown
  options before document loading;
- grep exit codes;
- process exit codes through both the console `main` entry point and the Typer
  application;
- deterministic result ordering;
- positional, repeated `-e`, file-supplied `-f`, fixed, regex,
  case-insensitive, and whole-word matching;
- empty pattern files and the `-f -`/document-stdin conflict;
- multiple inputs and stdin;
- comma-separated page selection and invalid page-range syntax;
- `--offset` and `--limit` across multiple ordered inputs;
- quiet, count, and matching-input modes, including incompatible options and
  strict quiet-mode errors;
- namespace binding;
- XPath node and scalar selection;
- output limits and truncation markers;
- JSON and JSONL schemas;
- omission of document, type, page, section, and structural metadata headers
  from normal `search` and `show` text output;
- `-n`/`--with-xpath` aliases, multiline XPath prefixes, match colons, context
  hyphens, multi-input filename prefixes, and standalone `--` separators only
  between disjoint context groups;
- TTY-only match highlighting with and without `-n`, redirected output, and
  `NO_COLOR`;
- `.dclx` safety checks;
- missing page images, geometry, and assets;
- invalid XML, invalid XPath, and no-match behavior.

Prefer existing DocLang and `docling-core` fixtures. Add one small end-to-end
fixture only where current fixtures cannot exercise the full source-to-context
flow.

## 21. Acceptance workflow

The first useful release must support this workflow:

```bash
dlgrep inspect contract.dclx --format json

dlgrep outline contract.dclx --depth 3 --format json

dlgrep -i \
  'termination|terminate|cancellation|cancel' \
  contract.dclx \
  --type heading,text,list_item,table_cell,footnote \
  -C 30 \
  --limit 12 \
  --format json

dlgrep show contract.dclx \
  '/d:doclang/d:heading[@level="2"][normalize-space(.)="Termination"]' \
  --section \
  --max-chars 5000

dlgrep -i \
  'except|unless|provided that|notwithstanding|subject to' \
  contract.dclx \
  --within-xpath \
  '/d:doclang/d:heading[@level="2"][normalize-space(.)="Termination"]' \
  --section \
  -C 1 \
  --format json
```

The returned evidence must contain bounded semantic text, structural context,
and reusable XPath addresses. No complete-document prompt or model call is
required.

## 22. Decisions for iteration

The following are intentionally left open in this draft:

1. final default character limits after testing representative large
   documents;
2. whether XPath 3.1 selection is required beyond the generated XPath
   1.0-compatible subset;
3. exact stable JSON schema naming before the first public release.

These decisions do not change the core architecture: XPath remains the public
source address, `DoclingDocument` provides semantic behavior, and the
deserialization source map connects them.
