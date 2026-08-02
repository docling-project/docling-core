# dclq

> [!WARNING]
> **Experimental:** `dclq` is experimental. Its commands, options, output
> formats, exit codes, and functionality may change in breaking ways without
> prior warning.

<p align="center">
  <strong>Query structured documents.</strong><br>
  Grep, list, outline, and XPath over headings, sections, lists, and
  tables—and get an XPath back for every result.
</p>

## What is dclq?

`dclq` brings the familiar grep workflow to
[DocLang](https://doclang.ai) documents. It queries semantic document units
and returns bounded evidence with reusable XPath addresses.

```console
$ dclq grep 'GPU|CPU' paper.dclg \
    --within-xpath '/heading[13]' --section -n
/list[4]/ldiv[1]:- AWS EC2 VM ... Nvidia L4 GPU ...
/text[42]:All experiments ... GPU acceleration ... x86 CPU ...
```

## Features

- 🔎 **Semantic search** across headings, paragraphs, captions, footnotes,
  list items, table cells, formulas, code, and metadata
- 🧭 **Structural context** with section-aware neighbours, list siblings, table
  headers, captions, and document reading order
- 🔗 **Reusable XPath addresses** for every result
- 🎯 **Precise filters** for sections, XPath regions, pages, layers, and
  semantic types
- 📑 **Document navigation** with structural inventory, heading outlines,
  semantic retrieval, and raw XPath selection
- 📦 **DocLang input** from `.dclg`, `.dclg.xml`, `.xml`, `.dclx`, or standard
  input
- 🤖 **Pipeline-friendly output** in text, JSON, and JSONL
- ⚡ **grep-compatible behavior** with regular expressions, fixed strings,
  context flags, counts, file listing, quiet mode, and exit codes
- 🔒 **Local, deterministic, read-only execution** with bounded output

## Quickstart

### 1. Install from PyPI

```bash
pip install dclq
```

### 2. Convert source documents to DocLang

```bash
python -m pip install docling
docling report.pdf handbook.docx --to dclx --output converted
```

### 3. Search

```bash
dclq grep -i 'termination|cancellation' converted/report.dclx
```

Without a pattern, `dclq list` enumerates units instead:

```bash
dclq list converted/report.dclx --type table_cell --page 3
```

## Examples

### Inspect a document

```console
$ dclq inspect paper.dclx
paper.dclx
Type: dclx
Pages: 9
Semantic units: 612
Elements: caption=9, code=3, formula=4, heading=18, list=5, picture=6, table=5, text=434
Metadata: author=2, date=1, keywords=1
```

### Navigate the heading hierarchy

```console
$ dclq outline paper.dclg
Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion  /heading[1]
  Abstract                                      /heading[2]
  1 Introduction                               /heading[3]
  ...
  3 Design and Architecture                    /heading[5]
  5 Performance                                /heading[11]
  5.2 System Configurations                    /heading[13]
  ...
  6 Applications                               /heading[16]
  ...
  References                                   /heading[19]
```

### Retrieve a section by XPath

```console
$ dclq show paper.dclg '/heading[13]' --section --max-chars 220 -n
/heading[13]:5.2 System Configurations
/text[41]:We schedule our benchmark experiments each on two different systems...
/list[4]/ldiv[1]:- AWS EC2 VM (g6.xlarge)...
/list[4]/ldiv[2]:- MacBook Pro M3 Max (ARM)...
/text[42]:All experiments on the AWS EC2 VM...
/table[1]/ched[1]:Asset
/table[1]/ched[2]:Version
...
```

Each section element keeps its own reusable XPath.

### Query the source XML

JSON includes document identity alongside scalar results:

```console
$ dclq select paper.dclg 'count(//page_break) + 1' --format json
{
  "document": "paper.dclg",
  "sha256": "284b9b63bf3e11a75ffd2ad23c7505a9b5e75407531a13044ceae001e0d1550e",
  "value": 8.0
}

$ dclq select paper.dclg \
    'normalize-space(string(//table[1]/caption))'
Table 1: Versions and configuration options considered for each tested asset. * denotes the default setting.
```

### Search a table cell with its structural context

The direct JSON record includes document identity, match offsets, and
contributing document items; the relevant fields are shown here:

```console
$ dclq grep -F '2.73 5.39' paper.dclx --format json
[
  {
    ...
    "xpaths": ["/d:doclang/d:table[1]/d:fcel[8]"],
    "logical_type": "table_cell",
    "text": "2.73 5.39",
    "pages": [1],
    "cell_context": {
      "column_headers": ["Inference time (secs)"],
      "caption": "Table 1. HPO performed in OTSL and HTML representation on the same transformer-based TableFormer..."
    }
  }
]
```

### Get list-aware context

```console
$ dclq grep -F 'Third item with numId 2' handbook.dclx \
    -C 1 --context-scope auto -n
/list[7]/ldiv[2]-2. Second item with numId 2
/list[7]/ldiv[3]:3. Third item with numId 2
/list[7]/ldiv[4]-4. Fourth item with numId 2
```

### Use dclq in shell pipelines

```bash
# List matching documents.
dclq grep -i 'human annotation' documents/*.dclx -l

# Count matching semantic units.
dclq grep -i 'inference|runtime' documents/*.dclx -c

# Emit bounded records for an agent or data pipeline.
dclq grep -i 'accuracy|performance' documents/*.dclx \
  --page 1-4 \
  --type text,table_cell \
  --limit 10 \
  --format jsonl

# Check for a match without output.
if dclq grep -q -F 'CONFIDENTIAL' document.dclx; then
  echo "classified"
fi
```

## Commands

| Command | Description |
| --- | --- |
| `dclq grep PATTERN INPUT...` | Search semantic document units |
| `dclq list INPUT...` | Enumerate semantic document units |
| `dclq inspect INPUT...` | Print a structural inventory |
| `dclq outline INPUT` | Print the heading hierarchy with XPaths |
| `dclq show INPUT XPATH` | Retrieve semantic content at an XPath |
| `dclq select INPUT XPATH` | Evaluate XPath against the source XML |

## Query options

| Option | Description |
| --- | --- |
| `-e PATTERN` | Add a search pattern; repeatable |
| `-f FILE` | Read patterns from a file |
| `-F` | Match fixed strings |
| `-i` | Ignore case |
| `-w` | Match whole words |
| `-A N`, `-B N`, `-C N` | Include semantic context |
| `--context-scope SCOPE` | Use `auto`, `container`, `section`, or `document` context |
| `--type TYPE` | Filter semantic unit types |
| `--layer LAYER` | Filter body, furniture, or background content |
| `--page LIST` | Filter pages and ranges such as `2-4,7` |
| `--within-xpath XPATH` | Restrict the query to an XPath selection |
| `--section` | Expand a selected heading to its section |
| `--limit N` | Limit the number of results |
| `-n`, `--with-xpath` | Prefix text output with XPath addresses |
| `--format FORMAT` | Emit `text`, `json`, or `jsonl` |
| `-c`, `-l`, `-q` | Count, list matching files, or run quietly |

Run `dclq COMMAND --help` for the complete option set.

XPath input may omit the namespace and document root: `/formula[1]`,
`/doclang/formula[1]`, and `/d:doclang/d:formula[1]` are equivalent.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | At least one result |
| `1` | No results |
| `2` | Input, query, or usage error |

## Development

`dclq` lives in the [docling-core](https://github.com/docling-project/docling-core)
repository as a workspace member under `packages/dclq`, and is released in
lockstep with `docling-core` (same version, exact dependency pin). Work on it
from the repository root:

```bash
uv sync --all-extras --all-packages
uv run pytest packages/dclq/tests
uv run pre-commit run --all-files
```

## License

`dclq` is available under the [MIT License](LICENSE).
