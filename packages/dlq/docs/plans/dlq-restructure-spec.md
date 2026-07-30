# dlq restructure spec

Status: proposed. Supersedes the CLI surface described in
[`document-grep-cli-plan.md`](document-grep-cli-plan.md); everything that plan
says about loading, addressing, source mapping, bounding, and exit codes still
holds.

## Motivation

`dlgrep` already ships five commands (`search`, `inspect`, `outline`, `select`,
`show`), of which only one is grep. The name understates the tool, and — more
concretely — the grep-shaped default command is what blocks plain listing. Today
the only way to enumerate all values in a form is:

```bash
dlgrep '' doc.dclg --type value
```

The empty pattern is the tell: a pattern-mandatory interface bent into a listing
tool. Nothing is published to PyPI yet and `dlq` is free, so the rename is a
clean break with no compatibility surface.

## Steps

Ordered by dependency. Each step is independently shippable and independently
revertable.

| # | Step | Shape |
|---|---|---|
| 1 | Rename `dlgrep` → `dlq` | mechanical, zero behaviour change |
| 2 | `grep` as explicit subcommand; add `list` | small CLI change |
| 3 | Drop `page` from `--context-scope` | deletion + tests |
| 4 | Add `fields` command | new feature |
| 5 | Fix `<!-- missing-text -->` in docling-core | **deferred**, upstream |

### Spec conformance

Checked against `/Users/cau/Documents/Development/doclang`: `spec.md` §Fields
(L725-880), `doclang/doclang.sch` pattern `field-structure-placement`
(L299-332), `doclang/doclang.xsd` (L642-670), and
`examples/form/form-examples.md`. Findings are folded into steps 3 and 4;
normative rules are cited inline where they constrain the design.

---

## Step 1 — rename `dlgrep` → `dlq`

Pure rename. No behaviour change, no new options. One commit so review is a
diff-stat glance.

### Moves

```
packages/dlgrep/        → packages/dlq/
packages/dlq/dlgrep/    → packages/dlq/dlq/
```

### Identifier renames

| From | To |
|---|---|
| `dlgrep` (distribution name) | `dlq` |
| `dlgrep` (import package) | `dlq` |
| `dlgrep = "dlgrep.cli:main"` (console script) | `dlq = "dlq.cli:main"` |
| `DlgrepError` | `DlqError` |
| `docling-core[dlgrep]` (extra) | `docling-core[dlq]` |
| `pytest-dlgrep` (pre-commit hook id) | `pytest-dlq` |
| `dist/dlgrep/` (build output dir) | `dist/dlq/` |
| `dlgrep-` (tempdir prefix) | `dlq-` |

### Files touched

Package: `packages/dlq/pyproject.toml`, `README.md`, `dlq/__init__.py`,
`dlq/__main__.py`, `dlq/cli.py`, `dlq/document.py`, `tests/test_cli.py`,
`docs/plans/*.md`.

Repo root: `pyproject.toml` (extra name at `:71`, dependency-group entry at
`:128`, workspace member at `:140`, workspace source at `:143`, mypy comment at
`:251`), `.pre-commit-config.yaml` (four path globs and the hook id), `uv.lock`.

CI: `.github/scripts/build-packages.sh`, `.github/scripts/release.sh`,
`.github/workflows/checks.yml`, `.github/workflows/pypi.yml` (including the
PyPI trusted-publisher environment URL `https://pypi.org/p/dlq`).

### Not doing

No `dlgrep` console-script alias. Nothing is published, so there is no muscle
memory to preserve and no back-compat to maintain. Add one later if the grep
framing turns out to matter for adoption.

### Description string

`"Semantic grep for DocLang documents"` → `"Query DocLang documents: grep, list,
outline, and XPath over semantic document structure"`. Keywords gain `query`.

### Verification

`uv sync --all-extras --all-packages`, `uv run pytest packages/dlq/tests`,
`uv run pre-commit run --all-files`, `python -m dlq --help`. Grepping the tree
for `dlgrep` returns only intentional prose in changelog history.

---

## Step 2 — `grep` subcommand and `list`

### Remove the default-command routing

Delete `_DefaultCommandGroup` (`cli.py:45-53`). Under `dlq`, a bare
`dlq PATTERN FILE` must not silently mean grep; the group falls back to Typer's
default `no_args_is_help` behaviour and unknown first arguments produce a normal
usage error.

Rename the `search` command to `grep`. The function keeps its options verbatim;
only the command name and docstring change.

### Add `list`

```
dlq list INPUT... [OPTIONS]
```

`list` enumerates semantic units without a pattern. Implementation is
`_search` invoked with an empty regex list — every filter and both bounds
already exist, so this is an entry point, not new machinery.

Options carried over from `grep`, unchanged:

`--type`, `--layer`, `--page`, `--within-xpath`, `--section`, `--view`,
`--offset`, `--limit`, `--max-chars`, `--max-output-chars`, `--all`,
`-n/--with-xpath`, `--format`, `--validate`, `-c/--count`,
`-l/--files-with-matches`, `-q/--quiet`.

Options **not** carried over (pattern-specific): `-e`, `-f`, `-F`, `-i`, `-w`,
`-A`, `-B`, `-C`, `--context-scope`. Context flags are meaningless without a
hit to be adjacent to.

Cursor-style paging is `--offset`/`--limit`, which already exist:

```bash
dlq list doc.dclg --offset 40 --limit 10        # units 41..50 in reading order
dlq list doc.dclg --type table_cell --page 3
```

### Exit codes

`list` follows the grep contract: `0` when at least one unit is emitted, `1`
when the filters select nothing, `2` on input, query, or usage error.

### Tests

- `list` with no filters emits every unit `grep ''` emits, in the same order.
- `--offset`/`--limit` window correctly and are stable across runs.
- Empty selection exits `1`.
- Pattern-specific options are rejected with a usage error.
- `dlq PATTERN FILE` (no subcommand) is a usage error, not a search.

---

## Step 3 — drop `page` from `--context-scope`

`--context-scope` currently offers `auto`, `container`, `section`, `page`,
`document`. Remove `page`.

`page` clips the context neighbourhood to units on the same physical page. That
boundary is a layout accident: a passage spanning a page break loses its context
for no semantic reason, and page *targeting* is already covered by `--page`.

The spec makes this concrete. §Page Break with Continuation (spec.md L1888)
requires content spanning a page break to be emitted as separate fragments
sharing a `<thread thread_id="N"/>`:

```xml
<text><thread thread_id="1"/>This paragraph spans across</text>
<page_break/>
<text><thread thread_id="1"/>multiple pages.</text>
```

A page-bounded context window is guaranteed to cut such a thread in half — it
severs precisely the relationship DocLang went out of its way to encode. No
other scope has that property.

(Whether dlq should merge threaded fragments into one unit at all is a separate
pre-existing question — `thread` is currently unhandled in `document.py`. Out of
scope here; noted so it is not mistaken for fallout from this deletion.)

Changes: drop `"page"` from the `ContextScope` literal (`cli.py:42`) and remove
its branch in `LoadedDocument._scope_units` (`document.py:418-423`).

`document` remains the default. `auto` stays as an opt-in policy — it selects
`container` for table cells and list items, `section` otherwise — but does not
become the default, because `-C N` should keep meaning plain reading-order
neighbours unless the caller asks for something else.

### Tests

`--context-scope` currently has **zero** coverage in `tests/test_cli.py`. This
step adds one test per surviving scope, asserting the boundary actually clips:

- `document` — context crosses a heading boundary.
- `section` — context stops at the heading boundary.
- `container` — a table cell's context is limited to cells of the same table;
  a list item's to siblings of the same list.
- `auto` — resolves to `container` for a table cell and to `section` for a
  paragraph.

Without these, the deletion is not provably safe.

---

## Step 4 — `fields` command

### Rationale

Key-values are a first-class DocLang structure (`field_region` / `field_item` /
`key` / `value`), just as headings are. `outline` exists because headings
deserve a dedicated, XPath-free listing with a record shape of their own
(`depth`). `fields` is the same argument for key-values, whose record shape is a
pair.

Today the closest available thing requires XPath knowledge and still never pairs
key with value:

```bash
dlq list doc.dclg --within-xpath '//value[not(normalize-space())]/../key'
```

### Signature

```
dlq fields INPUT [--fillable] [--empty] [--checkbox selected|unselected|all]
                 [-n/--with-xpath] [--format text|json|jsonl] [--validate]
```

### What the spec forces

Four constraints from §Fields that a naive flat listing gets wrong:

1. **Key and value are descendants, not children.** "Any `key` or `value`
   element must be a descendant of a `field_item` (not necessarily a direct
   child)" (spec.md L745). The §Fields "mixed content" example nests values
   inside a `<list>`. Ownership is by **nearest ancestor** `field_item` — the
   Schematron scopes the 0..1 key rule to "its own descendant scope, excluding
   descendants that belong to nested `field_item` elements" (L747,
   `doclang.sch:325`). In practice docling-core's deserializer already
   hierarchizes this correctly, so `fields` walks `FieldItem` items in the
   document tree rather than re-deriving ownership from XPath.

2. **Field regions and field items nest.** `examples/form/form-examples.md`
   §"Nesting forms and using form headings" nests `field_region` inside
   `field_region`, and `test/data/doc/kv_nested.out.dclg.xml` nests a whole
   `field_region` inside a `<value>`. A flat listing renders the outer pair as
   `A:` — which is not an empty value, it is a container. Emitting it would be a
   false positive under `--empty`.

3. **`<hint>` is first-class.** A hint "describes a format, example, or
   additional description" for a fillable value (spec.md L736). Hints appear as
   siblings *following* the element they describe, and document order is the
   only association — `kv_nested.out.dclg.xml` alternates
   `key, hint, value, hint, value, hint`. For an unfilled field the hint is the
   only content there is, so `--empty` output without hints is close to useless.

4. **Empty text ≠ empty value; `class` carries the intent.** `class="fillable"`
   marks "an empty or editable field that can be filled in", `read_only`
   (default) a pre-filled one (spec.md L747-749). This survives deserialization
   as `FieldValueItem.kind`. A `<value>` may also have empty text while holding
   children (a `<checkbox/>`, a nested `field_region`) — verified in
   `kv.out.dclg.xml`, where four values have `text == "" and children == 1`.

5. **A checkbox in a field item is a slot to fill.** `<checkbox>` is an empty
   element allowed in "any context that allows raw text content" (spec.md
   L2697-2711), so it appears in field items in three different encodings, all
   drawn from the spec's own examples — see below.

### Slots: what counts as a fillable position

A checkbox that must be ticked is a fillable position exactly like a blank
`<value>`, but the spec permits three encodings and does not normatively pick
one. All three are in the spec's own form examples, and all three deserialize
to checkbox items **within the field item's own scope**:

| Encoding | Source | Result |
|---|---|---|
| (a) inside `<value>` | `kv.out.dclg.xml` | checkbox item nested under a `field_value` |
| (b) inside `<text>` in a field item | spec.md L1078 | checkbox item as direct child of the field item |
| (c) direct child of `field_item` | spec.md L1104 | checkbox item as direct child of the field item |

`fields` therefore works on **slots**, not on `<value>` elements alone. A field
item's slots, in document order, are:

1. its owned `<value>` elements, and
2. checkbox items in its own scope that are not already inside one of those
   values.

One row per slot. This makes all three encodings produce the same output shape,
and is the only rule that does not silently drop real form fields.

Checkbox items carry their own label text: `<checkbox class="unselected"/>Visitor`
deserializes to a single item labelled `checkbox_unselected` with
`text == "Visitor"`. A slot renders as `[ ] Visitor` / `[x] Visitor`.

**Not slots**, and excluded for free by the scoping rule rather than by a
special case:

- Checkboxes outside any field item — `<text><checkbox/>TODO</text>` in
  `checkboxes.out.dclg.xml` is static content, never reachable from a field item.
- List-marker checkboxes — `<ldiv><marker><checkbox/></marker></ldiv>`
  (spec.md L547). Verified: the deserializer absorbs these into the list item's
  marker and never emits a checkbox item, so they cannot be mistaken for slots
  even when the list is nested inside a field item.

### Record shape

One row per slot, key repeated. Every other dlq command holds the
invariant **one row = one XPath**; joining a multi-value field item into
`key: a; b; c` produces a row with no single address, and multi-value field
items are normative (spec.md L809, "Field item with multiple values").

A field item with no owned `<value>` emits `key:` addressed at the key. A field
item with no owned `<key>` emits `: value` — also normative (spec.md L822,
"Field item without a key").

`depth` follows the nesting level, matching `outline`'s record shape. A value
whose content is a nested `field_region` emits **no row of its own**; its
nested field items emit their rows at `depth + 1`.

```console
$ dlq fields kv_nested.dclg
A:
  AA: AAA  [hint: Some explanation for value AAA]
  AA: AAB  [hint: Some explanation for value AAB]
  AB: ABA
  AB: ABB

$ dlq fields form.dclg --empty
I am in the United States as a:
Country of Citizenship:

$ dlq fields form.dclg --checkbox unselected
I am in the United States as a: [ ] Visitor
Apt.: [ ]
```

Hints are appended inline as `[hint: ...]` whenever present — a uniform rule,
not mode-dependent, and hints are rare enough that the noise is negligible.
With `-n`, each line is prefixed by the value's XPath (or the key's, for a
valueless field item), matching `outline`'s `-n` convention.

JSON/JSONL:

```json
{
  "document": "kv.dclg",
  "key": "name",
  "value": "",
  "kind": "fillable",
  "checkbox": null,
  "empty": true,
  "hints": ["Enter your first and last name"],
  "depth": 1,
  "key_xpath": "/d:doclang/d:field_region[1]/d:field_item[2]/d:key[1]",
  "value_xpath": "/d:doclang/d:field_region[1]/d:field_item[2]/d:value[4]",
  "pages": [1]
}
```

`kind`, `checkbox`, and `empty` are the three orthogonal facts described below;
all three appear on every row so a caller can always see which basis a row
qualified on.

This closes the pairing gap directly. No `jq` post-processing.

### Three orthogonal facts per slot

A slot is described by three independent facts. None is derived from another,
and no filter conflates them.

| Fact | Values | Source |
|---|---|---|
| `kind` | `fillable` \| `read_only` \| `null` | `class` on `<value>`; `null` when the slot is a bare checkbox |
| `checkbox` | `selected` \| `unselected` \| `null` | `class` on `<checkbox>`; `null` when the slot holds no checkbox |
| `empty` | `true` \| `false` | slot has no text and no content of any kind |

A checkbox slot is **never** `empty` — it has content, namely a checkbox.
Whether that box is ticked is the `checkbox` fact, not the `empty` fact.

Rows are still suppressed for one case: a value whose content is a nested
`field_region` or `field_item` is a container, not a leaf slot. In
`kv_nested.out.dclg.xml` the outer value has `text == ""` and one child, and
that child is a whole nested field region — emitting it as `A:` would be wrong
on both counts. Its nested field items emit their own rows at `depth + 1`.

**`empty` is a property of the item tree, never of the rendered string.** A
value containing only a checkbox renders as `[ ] Visitor` — a non-empty string
for a slot whose `empty` is false and whose `checkbox` is `unselected`. Any
implementation that decides either fact from serialized text is wrong.

### Filters

Three flags, one per fact, AND-combined:

```
--fillable            slots that can be filled in
--empty               slots with no content at all
--checkbox STATE      checkbox slots; STATE is selected | unselected | all
```

`--checkbox` follows the existing `--layer` convention of an explicit `all`
member rather than an optional-value flag.

| Question | Query |
|---|---|
| boxes still to tick | `--checkbox unselected` |
| blanks still to write in | `--empty` |
| everything that can be filled | `--fillable` |
| declared-fillable blanks | `--fillable --empty` |
| ticked boxes | `--checkbox selected` |

`--empty --checkbox` returns nothing by construction, since a checkbox slot has
content. That is a consistent consequence of orthogonality, not a special case.

### What `--fillable` means

`kind == "fillable"` **or** the slot is a checkbox.

The checkbox clause is not a heuristic — it is forced by the schema. `class` is
defined on `<value>` only (spec.md L736), so a bare `<checkbox>` in encodings
(b) and (c) has no way to declare fillability even when the producer wants to.
A checkbox inside a field item is a fillable position by construction; static
checkbox label text is already excluded by the slot scoping rule, so there is
nothing left for the clause to over-select.

### Why `--empty` is needed as well

`--fillable` trusts `class="fillable"`. Producers frequently omit it, and
`read_only` is the schema default — so an unmarked blank silently becomes
"pre-filled". This is not hypothetical; the spec's own form example at
spec.md L1078 contains

```xml
<field_item>
    <key>I am in the United States as a:</key>
    <text><checkbox class="unselected"/>Visitor</text>
    <value></value>
</field_item>
```

whose `<value></value>` — plainly a blank to be filled — deserializes to
`kind == "read_only"` and is therefore missed by `--fillable`.

So `--fillable` is the correct query against a well-formed document, and
`--empty` is the structural fallback that finds technically-fillable values when
the producer did not mark them. Both are documented as such: `--fillable` alone
under-reports on real forms, and `--empty` alone cannot tell a blank to fill
from an extraction artifact.

### Known limitation: no OR

Filters AND. "Everything still needing attention" is the union of `--empty` and
`--checkbox unselected`, which needs two invocations:

```bash
dlq fields form.dclg --empty --format jsonl >  todo.jsonl
dlq fields form.dclg --checkbox unselected --format jsonl >> todo.jsonl
```

Composing filters with OR is a query language, and dlq already has one for
arbitrary structural conditions — XPath, via `select` and `--within-xpath`.
Revisit only if this specific union turns out to be the dominant use, in which
case a single named `--unfilled` covering it beats a general OR.

`--filled` is not included — it is the inverse of a listing you can already
read.

### Implementation notes

- **Bypass `display_unit`.** Inline key-values live inside a `<text>` element,
  and the parent-collapsing at `document.py:238` swallows them into the
  enclosing paragraph — which is why `--type key` today prints
  `/field_region[2]/text[1]` instead of the key's own address. `fields` reads
  key/value units directly.
- **Keys are not always text.** `test/data/doc/kv_invoice.out.dclg.xml` has
  `<key><picture/></key>`. Such a key renders as an empty string, giving
  `: +123-456-7890`. Honest output, not an error.
- **`field_heading` rows are not emitted.** Headings label groups of field items
  and carry an optional `level` (spec.md L734), but including them would make
  the record shape polymorphic for marginal gain. `depth` already conveys the
  nesting. Revisit if grouped output is actually wanted.
- **`<marker>` is ignored.** `kv_nested.out.dclg.xml` and form-01 put markers
  (`1.`, `14.`) inside field items. Not part of the key-value pair.
- **Independent of step 5.** `fields` reads units directly rather than asking
  the serializer to render a `field_item`, so it is unaffected by the
  fallback-serializer bug and does not wait on it.

### Findings for upstream (not blocking)

Two gaps surfaced while checking this against the reference, both worth raising
in `doclang` rather than working around here:

1. **The spec does not say where a checkbox option lives in a field item.** Its
   own examples use three different encodings (L1078, L1104, versus the
   `<value>`-wrapped form docling-core emits). The slot rule above tolerates all
   three, but producers would benefit from a normative recommendation.
2. **`class="fillable"` is unreachable for a bare checkbox.** `class` is defined
   on `<value>` only (spec.md L736), so encodings (b) and (c) have no way to
   declare fillability even when the producer wants to. This is the structural
   reason `--empty` has to exist, not merely a producer-quality issue.

### Tests

Against `kv.out.dclg.xml`, `kv_invoice.out.dclg.xml`, and
`kv_nested.out.dclg.xml`:

- multi-value field item emits one row per value, key repeated;
- valueless field item emits `key:`; keyless field item emits `: value`;
- picture-only key emits an empty key;
- inline key-values embedded in `<text>` are addressed at the key/value, not
  the paragraph;
- nested field regions emit no row for the containing value, and their field
  items carry `depth + 1`;
- hints attach to the preceding key or value in document order;
- JSON records carry both xpaths and round-trip through `dlq show`;
- a document with no field regions exits `1`.

Checkbox and state coverage needs a fixture holding all three encodings side by
side, plus the two non-slot cases — none of the existing test data has them
together:

- each of encodings (a), (b), (c) yields one slot row, with the same shape;
- a checkbox slot has `empty: false` **despite** having no text of its own, and
  regardless of tick state — the regression guard against re-conflating the two
  axes, and against deciding either fact from the rendered `[ ] Visitor`;
- `--checkbox unselected` and `--checkbox selected` partition `--checkbox all`;
- `--empty` never returns a checkbox slot; `--empty --checkbox all` is empty;
- a static checkbox outside any field item yields no row;
- a list-marker checkbox inside a field item yields no row;
- `--empty` selects blank values regardless of `kind`, and includes the
  spec.md L1078 case whose `<value></value>` deserializes to `read_only`;
- `--fillable` **misses** that same L1078 value — asserted explicitly, so the
  documented limitation stays true;
- `--fillable` does select bare-checkbox slots, whose `kind` is `null`;
- `--fillable --empty` intersects;
- `kind`, `checkbox`, and `empty` are populated on every row.

---

## Step 5 — fix `<!-- missing-text -->` in docling-core (deferred)

**Deferred.** Recorded here as diagnosis only; not part of this series and not a
prerequisite for any step above. Tracked separately in docling-core.

### Diagnosis

Not a dlq bug. `FieldRegionItem` and `FieldItem`
(`docling_core/types/doc/items/form.py:11,20`) are bare `DocItem` subclasses —
structural containers that hold children but carry no text of their own. The
dispatch chain in `serializer/common.py:430-490` tests for `TextItem`,
`TableItem`, `PictureItem`, `KeyValueItem`, `FormItem`, and `_PageBreakNode`,
then falls through. `MarkdownFallbackSerializer` (`markdown.py:821-842`)
recurses only for `GroupItem`; every other `DocItem` gets the literal
`<!-- missing-text -->`.

`PlainTextDocSerializer` inherits `MarkdownDocSerializer` without overriding
`fallback_serializer`, so dlq sees the same string.

### Blast radius

Much wider than `dlq --type field_item`. Plain `export_to_markdown()` on any
document with key-values is degraded:

```
<!-- missing-text -->

<!-- missing-text -->

<!-- image -->

+123-456-7890
```

Every `field_region` and every `field_item` becomes noise.

### Shape of the fix

Not a generic fallback tweak. Routing `FieldRegionItem` / `FieldItem` through
the fallback's `GroupItem` branch would join children with `\n\n`, rendering a
field item as `key\n\nvalue` — which is a formatting decision smuggled into a
container that has no business making one.

How a `field_region` or a `field_item` prints is the **serializer's**
responsibility, and each format answers it differently: Markdown may want
`**key:** value`, plain text `key: value`, LaTeX something else again. The fix
is therefore a real field serializer — a `BaseFieldSerializer` alongside the
existing `key_value_serializer` and `form_serializer`, dispatched from
`common.py` and implemented per format — not a change to
`MarkdownFallbackSerializer`.

That also settles what `<!-- missing-text -->` should mean: a genuinely textless
item, not "a container the dispatch chain forgot to name".

### Consequence for dlq

None, until it lands. `fields` (step 4) does not depend on it. `--type
field_item` stays in the type filter and stays unhelpful; no dlq-side workaround
is warranted for an upstream bug with a known owner.

---

## Out of scope

- No `--filled` on `fields`.
- No `--unfilled` union flag, and no OR-composition of filters.
- No `field_heading` rows in `fields` output.
- No inference of fillability beyond `kind`, checkbox presence, and structural
  emptiness — no guessing from key wording, underscores, or geometry.
- No predicate DSL. Arbitrary structural conditions remain XPath via
  `--within-xpath` and `select`.
- No `dlgrep` alias.
- No `<thread>` merging of page-split units. Pre-existing gap, unrelated to
  these changes.
- No changes to loading, addressing, source mapping, bounding, or exit-code
  semantics.

## Verification for the whole series

```bash
uv sync --all-extras --all-packages
uv run pytest packages/dlq/tests test/
uv run pre-commit run --all-files
python -m dlq --help
```

Step 4 additionally validates its fixtures against the DocLang schema
(`dlq fields --validate`), since the nesting and ownership rules it implements
are normative.
