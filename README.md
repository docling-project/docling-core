# dlgrep

<p align="center">
  <strong>Semantic grep for structured documents.</strong><br>
  Find text in headings, sections, lists, and tables—and get its XPath back.
</p>

## What is dlgrep?

`dlgrep` brings the familiar grep workflow to
[DocLang](https://doclang.ai) documents. It searches semantic document units
and returns bounded evidence with reusable XPath addresses.

```console
$ dlgrep -F '6 CONCLUSION' paper.dclx --type heading -n
/heading[17]:6 CONCLUSION
```

## Features

- 🔎 **Semantic search** across headings, paragraphs, captions, footnotes,
  list items, table cells, formulas, code, and metadata
- 🧭 **Structural context** with heading ancestry, list siblings, table
  headers, captions, and document reading order
- 🔗 **Reusable XPath addresses** for every result
- 🎯 **Precise filters** for sections, XPath regions, pages, layers, semantic
  types, and Docling classes
- 📑 **Document navigation** with structural inventory, heading outlines,
  semantic retrieval, and raw XPath selection
- 📦 **DocLang input** from `.dclg`, `.dclg.xml`, `.xml`, `.dclx`, or standard
  input
- 🤖 **Pipeline-friendly output** in text, JSON, and JSONL
- ⚡ **grep-compatible behavior** with regular expressions, fixed strings,
  context flags, counts, file listing, quiet mode, and exit codes
- 🔒 **Local, deterministic, read-only execution** with bounded output

## Quickstart

### 1. Install

```bash
python -m pip install \
  "dlgrep @ git+https://github.com/docling-project/dlgrep.git"
```

### 2. Convert source documents to DocLang

```bash
python -m pip install docling
docling report.pdf handbook.docx --to dclx --output converted
```

### 3. Search

```bash
dlgrep -i 'termination|cancellation' converted/report.dclx \
  -C 2 --context-scope auto
```

Search is the default command, so `dlgrep PATTERN INPUT` and
`dlgrep search PATTERN INPUT` are equivalent.

## Examples

### Inspect a document

```console
$ dlgrep inspect paper.dclx
paper.dclx
Type: dclx
Pages: 9
Semantic units: 612
Elements: caption=9, code=3, formula=4, heading=18, list=5, picture=6, table=5, text=434
Metadata: author=2, date=1, keywords=1
```

### Navigate the heading hierarchy

```console
$ dlgrep outline handbook.dclx
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

### Chain searches with XPath

Capture a section heading's XPath, then pass it directly into another search:

```console
$ conclusion_xpath=$(dlgrep -F '6 CONCLUSION' paper.dclx \
    --type heading --format json | jq -r '.[0].metadata.xpath')
$ echo "$conclusion_xpath"
/d:doclang/d:heading[17]

$ dlgrep -i 'dataset|performance' paper.dclx \
    --within-xpath "$conclusion_xpath" --section \
    --type text --limit 3 --max-chars 240 -n
/text[64]:In this paper, we presented the DocLayNet dataset. It provides the document conversion and layout analysis research community a new and challenging dataset to improve and fine-tune novel ML methods on. In contrast to many other…
/text[65]:From the dataset, we have derived on the one hand reference metrics for human performance on document-layout annotation (through double and triple annotations) and on the other hand evaluated the baseline performance of commonl…
```

The same XPath can retrieve the entire section:

```bash
dlgrep show paper.dclx "$conclusion_xpath" --section
```

### Inspect element neighbourhood

`-B` and `-A` return addressable semantic elements before and after a hit.
Overlapping neighbourhoods are merged; a standalone `--` separates only
disjoint context groups:

```console
$ dlgrep -F 'reference metrics for human performance' paper.dclx \
    --within-xpath "$conclusion_xpath" --section --type text \
    -B 1 -A 1 --context-scope section --max-chars 220 -n
/text[64]-In this paper, we presented the DocLayNet dataset. It provides the document conversion and layout analysis research community a new and challenging dataset to improve and fine-tune novel ML methods on. In co…
/text[65]:From the dataset, we have derived on the one hand reference metrics for human performance on document-layout annotation (through double and triple annotations) and on the other hand evaluated the baseline pe…
/text[66]-To date, there is still a significant gap between human and ML accuracy on the layout interpretation task, and we hope that this work will inspire the research community to close that gap.
```

### Search a table cell with its structural context

```console
$ dlgrep -F '2.73 5.39' paper.dclx --format json |
  jq '.[0] | {xpath: .metadata.xpath, text: .raw_text, page: .page_numbers[0], heading: .headings[-1], column: .metadata.context.column_headers[0], table: .captions[0][0:96]}'
{
  "xpath": "/d:doclang/d:table[1]/d:fcel[8]",
  "text": "2.73 5.39",
  "page": 1,
  "heading": "5.1 Hyper Parameter Optimization",
  "column": "Inference time (secs)",
  "table": "Table 1. HPO performed in OTSL and HTML representation on the same transformer-based TableFormer"
}
```

### Get list-aware context

```console
$ dlgrep -F 'Third item with numId 2' handbook.dclx \
    -C 1 --context-scope auto -n
/list[7]/ldiv[2]-2. Second item with numId 2
/list[7]/ldiv[3]:3. Third item with numId 2
/list[7]/ldiv[4]-4. Fourth item with numId 2
```

### Use dlgrep in shell pipelines

```bash
# List matching documents.
dlgrep -i 'human annotation' documents/*.dclx -l

# Count matching semantic units.
dlgrep -i 'inference|runtime' documents/*.dclx -c

# Emit bounded records for an agent or data pipeline.
dlgrep -i 'accuracy|performance' documents/*.dclx \
  --page 1-4 \
  --type text,table_cell \
  --limit 10 \
  --format jsonl

# Check for a match without output.
if dlgrep -q -F 'CONFIDENTIAL' document.dclx; then
  echo "classified"
fi
```

## Commands

| Command | Description |
| --- | --- |
| `dlgrep PATTERN INPUT...` | Search semantic document units |
| `dlgrep search PATTERN INPUT...` | Explicit search command |
| `dlgrep inspect INPUT...` | Print a structural inventory |
| `dlgrep outline INPUT` | Print the heading hierarchy with XPaths |
| `dlgrep show INPUT XPATH` | Retrieve semantic content at an XPath |
| `dlgrep select INPUT XPATH` | Evaluate XPath against the source XML |

## Search options

| Option | Description |
| --- | --- |
| `-e PATTERN` | Add a search pattern; repeatable |
| `-f FILE` | Read patterns from a file |
| `-F` | Match fixed strings |
| `-i` | Ignore case |
| `-w` | Match whole words |
| `-A N`, `-B N`, `-C N` | Include semantic context |
| `--context-scope SCOPE` | Use `auto`, `container`, `section`, `page`, or `document` context |
| `--type TYPE` | Filter semantic unit types |
| `--class CLASS` | Filter Docling item classes |
| `--layer LAYER` | Filter body, furniture, or background content |
| `--page LIST` | Filter pages and ranges such as `2-4,7` |
| `--within-xpath XPATH` | Restrict search to an XPath selection |
| `--section` | Expand a selected heading to its section |
| `--limit N` | Limit the number of results |
| `-n`, `--with-xpath` | Prefix text output with XPath addresses |
| `--format FORMAT` | Emit `text`, `json`, or `jsonl` |
| `-c`, `-l`, `-q` | Count, list matching files, or run quietly |

Run `dlgrep COMMAND --help` for the complete option set.

XPath input may omit the namespace and document root: `/formula[1]`,
`/doclang/formula[1]`, and `/d:doclang/d:formula[1]` are equivalent.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | At least one match |
| `1` | No matches |
| `2` | Input, query, or usage error |

## Development

```bash
uv sync
uv run pytest
uv run pre-commit run --all-files
```

## License

`dlgrep` is available under the [MIT License](LICENSE).
