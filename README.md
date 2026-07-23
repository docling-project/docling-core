# dlgrep

`dlgrep` is grep for structured documents. It searches Docling semantic units
instead of flattened lines and returns reusable DocLang XPath addresses.

A hit can therefore tell you:

- which section and page it belongs to;
- whether it is a heading, paragraph, list item, or table cell;
- which headers and caption explain a table value;
- which XPath retrieves the exact item or containing section.

It is deterministic, read-only, bounded by default, and has grep-compatible
exit codes.

## Install

```bash
uv sync
uv run dlgrep --help
```

`dlgrep` reads DocLang (`.dclg`, `.dclg.xml`, or `.xml`) and DocLang archives
(`.dclx`). Convert source documents once with
[Docling](https://github.com/docling-project/docling):

```bash
docling report.pdf handbook.docx --to dclx --output converted

uv run dlgrep -i 'termination|cancellation' converted/report.dclx -C 2
```

`dlgrep` deliberately does not perform PDF or Word conversion itself.

## Real document examples

The output below comes from these Docling test fixtures:

- `2206.01062.pdf`, the nine-page DocLayNet paper;
- `2305.03393v1-pg9.pdf`, a paper page containing an experimental-results
  table;
- `unit_test_headers_numbered.docx`, with nested numbered headings;
- `docx_lists.docx`, with nested and sibling lists.

They are available under `docling_release/tests/data/{pdf,docx}/sources/` in a
Docling checkout. After converting them to DCLX:

```bash
docling \
  ../docling_release/tests/data/pdf/sources/2206.01062.pdf \
  ../docling_release/tests/data/pdf/sources/2305.03393v1-pg9.pdf \
  ../docling_release/tests/data/docx/sources/unit_test_headers_numbered.docx \
  ../docling_release/tests/data/docx/sources/docx_lists.docx \
  --to dclx --output converted
```

### See the document before reading it

`inspect` gives a bounded structural inventory without dumping the document:

```console
$ uv run dlgrep inspect converted/2206.01062.dclx | sed -n '3,6p'
Type: dclx
Pages: 9
Headings: 18
Source bindings: 979
```

For a Word document, `outline` reconstructs the actual heading hierarchy and
keeps every heading addressable:

```console
$ uv run dlgrep outline converted/unit_test_headers_numbered.dclx
Test Document	/d:doclang/d:heading[1]
  1 Section 1	/d:doclang/d:heading[2]
    1.1 Section 1.1	/d:doclang/d:heading[3]
    1.2 Section 1.2	/d:doclang/d:heading[4]
      1.2.1 Section 1.2.3	/d:doclang/d:heading[5]
  2 Section 2	/d:doclang/d:heading[6]
    2.1.1 Section 2.1.1	/d:doclang/d:heading[7]
    2.2 Section 2.1	/d:doclang/d:heading[8]
    REFERENCES	/d:doclang/d:heading[9]
```

### Find an anchor, then retrieve its section

Search returns a source address, semantic type, and page:

```console
$ uv run dlgrep -F '6 CONCLUSION' converted/2206.01062.dclx --type heading
converted/2206.01062.dclx
XPath: /d:doclang/d:heading[17]
Type: heading
Page: 8
6 CONCLUSION
```

The returned XPath is input to `show`; `--section` expands the heading to its
semantic subtree rather than an arbitrary number of surrounding lines:

```console
$ uv run dlgrep show converted/2206.01062.dclx \
    '/d:doclang/d:heading[17]' --section --max-chars 500
converted/2206.01062.dclx
XPath: /d:doclang/d:heading[17]
Type: section
Page: 8
6 CONCLUSION

In this paper, we presented the DocLayNet dataset. It provides the document conversion and layout analysis research community a new and challenging dataset to improve and fine-tune novel ML methods on. In contrast to many other datasets, DocLayNet was created by human annotation in order to obtain reliable layout ground-truth on a wide variety of publication- and typesettingstyles. Including a large proportion of documents outside the scientific publishing domain adds … [truncated]
```

The truncation marker is emitted by `dlgrep`; output is bounded instead of
silently flooding a terminal or agent context.

### Search inside tables without flattening them

This searches for a measured runtime in a PDF table. The small `jq` projection
shows the context attached to the matching cell:

```console
$ uv run dlgrep -F '2.73 5.39' converted/2305.03393v1-pg9.dclx \
    --format json |
  jq '.[0] | {xpath, text, page, heading: .context.headings[-1], column: .context.column_headers[0], table: .context.table_caption[0:96]}'
{
  "xpath": "/d:doclang/d:table[1]/d:fcel[8]",
  "text": "2.73 5.39",
  "page": 1,
  "heading": "5.1 Hyper Parameter Optimization",
  "column": "Inference time (secs)",
  "table": "Table 1. HPO performed in OTSL and HTML representation on the same transformer-based TableFormer"
}
```

The result is one table cell—not a match in a flattened page—and carries its
section, column header, table caption, row and column coordinates, and source
XPath in the full JSON record.

### Ask for semantic context, not nearby XML

With `--context-scope auto`, context follows the matched structure. A list hit
returns its sibling list items and heading chain:

```console
$ uv run dlgrep -F 'Third item with numId 2' converted/docx_lists.dclx \
    -C 1 --context-scope auto
converted/docx_lists.dclx
XPath: /d:doclang/d:list[7]/d:ldiv[3]
Type: list_item
Page: 1
Section: Test Document > Test 7:
- /d:doclang/d:list[7]/d:ldiv[2] 2. Second item with numId 2
3. Third item with numId 2
- /d:doclang/d:list[7]/d:ldiv[4] 4. Fourth item with numId 2
```

For prose, automatic context stays within the section; for table cells, it
stays within the table. Explicit scopes are `container`, `section`, `page`,
and `document`.

### Use it like grep in scripts

Search is the default command, regular expressions are the default pattern
syntax, and multiple inputs are processed in order:

```bash
# Print only matching documents.
uv run dlgrep -i 'human annotation' converted/*.dclx -l

# Count semantic units, not matching lines.
uv run dlgrep -i 'inference|runtime' converted/*.dclx -c

# Emit one bounded record per line for a pipeline or agent.
uv run dlgrep -i 'accuracy|performance' converted/*.dclx \
  --page 1-4 --type text,table_cell --limit 10 --format jsonl

# Scope a second search to a previously discovered section.
uv run dlgrep -i 'dataset|annotation' converted/2206.01062.dclx \
  --within-xpath '/d:doclang/d:heading[8]' --section -C 1
```

Exit status is `0` for a match, `1` for no match, and `2` for an input or query
error. `-q` performs a silent existence check while preserving those statuses.

## Command map

```text
dlgrep [SEARCH OPTIONS] PATTERN INPUT...
dlgrep search [SEARCH OPTIONS] PATTERN INPUT...
dlgrep inspect INPUT...
dlgrep outline INPUT
dlgrep show INPUT XPATH
dlgrep select INPUT XPATH
```

Useful search controls include:

- `-F`, `-i`, and `-w` for fixed, case-insensitive, and whole-word matching;
- repeated `-e` or pattern files with `-f`;
- semantic `-A`, `-B`, and `-C` context;
- `--type`, `--class`, `--layer`, `--page`, and `--within-xpath` filters;
- `--limit`, `--max-chars`, and `--max-output-chars` bounds;
- text, JSON, and JSONL output.

Inputs may also be read from standard input with `-`. Run
`uv run dlgrep COMMAND --help` for the complete option set.

## Why the XPath is trustworthy

Every semantic result is bound to its source element during `docling-core`
deserialization. Ordinary items, virtual list items, logically joined text
fragments, and OTSL table cells retain source identity without matching
normalized text back to XML.

`show` resolves those addresses semantically. `select` evaluates XPath against
the original XML when exact source fidelity is more useful:

```bash
uv run dlgrep show report.dclx '/d:doclang/d:heading[4]' --section
uv run dlgrep select report.dclx '/d:doclang//d:table' --format xml
```

The optional `image` command, physical thread-fragment search, and richer
field/list/picture context are intentionally deferred.
