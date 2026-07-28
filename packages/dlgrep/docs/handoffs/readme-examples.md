# Handoff: verified README examples

Verified on 2026-07-26 with the current dlgrep `main` checkout and CLI version
`0.0.0`. Commands below use the checked-out `.venv/bin/dlgrep`.

## Inputs

1. Docling paper:
   [viewer fixture](https://github.com/doclang-project/viewer/blob/main/demo_data/2501.17887/document.xml)
   (`sha256: 284b9b63bf3e11a75ffd2ad23c7505a9b5e75407531a13044ceae001e0d1550e`)
2. DocLang comprehensive fixture:
   `/Users/cau/Documents/Development/doclang/tests/data/valid/ok_comprehensive.dclg`
   (`sha256: bd7d7b64ed562b2dda2149b0b66b7541651d77f1ccb7a38ec689076117faacb6`)

Use generic names such as `paper.dclg` and `comprehensive.dclg` in the public
README. The paths above exist only to make this handoff reproducible.

## Outline

The paper produces a useful, addressable table of contents:

```console
$ dlgrep outline paper.dclg
Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion  /d:doclang/d:heading[1]
  Abstract                                      /d:doclang/d:heading[2]
  1 Introduction                               /d:doclang/d:heading[3]
  2 State of the Art                           /d:doclang/d:heading[4]
  3 Design and Architecture                    /d:doclang/d:heading[5]
  3.1 Docling Document                         /d:doclang/d:heading[6]
  3.2 Parser Backends                          /d:doclang/d:heading[7]
  3.3 Pipelines                                /d:doclang/d:heading[8]
  4 PDF Conversion Pipeline                    /d:doclang/d:heading[9]
  4.1 AI Models                                /d:doclang/d:heading[10]
  5 Performance                                /d:doclang/d:heading[11]
  5.1 Benchmark Dataset                        /d:doclang/d:heading[12]
  5.2 System Configurations                    /d:doclang/d:heading[13]
  5.3 Benchmarking Methodology                 /d:doclang/d:heading[14]
  5.4 Results                                  /d:doclang/d:heading[15]
  6 Applications                               /d:doclang/d:heading[16]
  7 Ecosystem                                  /d:doclang/d:heading[17]
  8 Future Work and Contributions              /d:doclang/d:heading[18]
  References                                   /d:doclang/d:heading[19]
```

The comprehensive fixture demonstrates nested group headings:

```console
$ dlgrep outline comprehensive.dclg
Introduction                                    /d:doclang/d:heading[1]
Document Title: A Comprehensive Guide           /d:doclang/d:heading[2]
Advanced TopicsinMachine Learning               /d:doclang/d:heading[3]
  Section Title                                 /d:doclang/d:group[12]/d:heading[1]
    Subsection                                  /d:doclang/d:group[12]/d:group[1]/d:heading[1]
```

The missing space in `Advanced TopicsinMachine Learning` is the observed
serializer output, not a transcription error.

## Select and show a section

The outline identifies `5 Performance` as `/d:doclang/d:heading[11]`:

```console
$ dlgrep -F '5 Performance' paper.dclg --type heading -n
/heading[11]:5 Performance

$ dlgrep show paper.dclg '/heading[11]' --section --max-chars 600 -n
/heading[11]:5 Performance
/text[38]:In this section, we characterize the conversion speed...
/text[39]:Further, we compare the conversion speed...
```

XPath arguments containing brackets must be quoted in shells such as zsh.
Each section element keeps its own XPath; `--max-chars` applies independently
to each element.

Important fixture behavior: headings `5`, `5.1`, `5.2`, `5.3`, and `5.4` are
all encoded as `level="2"`. They are therefore semantic siblings. The section
anchored at heading 11 contains the introductory paragraphs, not headings
12–15. dlgrep correctly follows DocLang heading levels rather than inferring
hierarchy from numbering in heading text.

`5.2 System Configurations` is a richer section example:

```console
$ dlgrep show paper.dclg '/heading[13]' --section --max-chars 220 -n
/heading[13]:5.2 System Configurations
/text[41]:We schedule our benchmark experiments each on two different systems...
/list[4]/ldiv[1]:- AWS EC2 VM (g6.xlarge)...
/list[4]/ldiv[2]:- MacBook Pro M3 Max (ARM)...
/text[42]:All experiments on the AWS EC2 VM...
/table[1]/ched[1]:Asset
/table[1]/ched[2]:Version
...
```

The same heading can bound a targeted search:

```console
$ dlgrep -i 'GPU|CPU' paper.dclg \
    --within-xpath '/heading[13]' --section --limit 5 --max-chars 220 -n
/list[4]/ldiv[1]:- AWS EC2 VM ... Nvidia L4 GPU ...
/text[42]:All experiments ... GPU acceleration ... x86 CPU ...
```

The comprehensive fixture confirms that `--section` expands a heading while
plain `--within-xpath` does not:

```console
$ dlgrep -F 'Subsection content' comprehensive.dclg \
    --within-xpath '/group[12]/heading[1]' -n
# no result

$ dlgrep -F 'Subsection content' comprehensive.dclg \
    --within-xpath '/group[12]/heading[1]' --section -n
/group[12]/group[1]/text[1]:Subsection content
```

## Non-empty `select` queries

The paper supports useful scalar and structural queries:

```console
$ dlgrep select paper.dclg 'count(//page_break) + 1'
8.0

$ dlgrep select paper.dclg 'count(//xref)' --format json
{
  "document": "paper.dclg",
  "sha256": "284b9b63bf3e11a75ffd2ad23c7505a9b5e75407531a13044ceae001e0d1550e",
  "value": 3.0
}

$ dlgrep select paper.dclg \
    'normalize-space(string(//table[1]/caption))'
Table 1: Versions and configuration options considered for each tested asset. * denotes the default setting.
```

The comprehensive fixture demonstrates source-native attribute filtering with
XPath:

```console
$ dlgrep select comprehensive.dclg \
    '//value[@class="fillable"]//text()[normalize-space()]' --format text
john.doe@example.com
Confidential
john@example.com
+1-555-0123

$ dlgrep select comprehensive.dclg 'count(//field_item)' --format json
{
  "document": "comprehensive.dclg",
  "sha256": "bd7d7b64ed562b2dda2149b0b66b7541651d77f1ccb7a38ec689076117faacb6",
  "value": 14.0
}

$ dlgrep select comprehensive.dclg \
    'string(//picture[src][1]/src/@uri)'
https://example.com/image.jpg

$ dlgrep select comprehensive.dclg \
    '//checkbox[@class="selected"]' --format xml
<checkbox xmlns="https://www.doclang.ai/ns/v0" class="selected"/>
<checkbox xmlns="https://www.doclang.ai/ns/v0" class="selected"/>
```

## Context scopes with observable differences

### `auto` versus `section` for a list item

`auto` recognizes a list hit and stays within the list:

```console
$ dlgrep -F 'MacBook Pro M3 Max (ARM), 64GB RAM' paper.dclg \
    -C 2 --context-scope auto -n
/list[4]/ldiv[1]-- AWS EC2 VM ...
/list[4]/ldiv[2]:- MacBook Pro M3 Max (ARM), 64GB RAM, on macOS 14.7
```

Explicit `section` answers a different question: show prose around the list,
not just sibling list items.

```console
$ dlgrep -F 'MacBook Pro M3 Max (ARM), 64GB RAM' paper.dclg \
    -C 2 --context-scope section -n
/text[41]-We schedule our benchmark experiments each on two different systems...
/list[4]/ldiv[1]-- AWS EC2 VM ...
/list[4]/ldiv[2]:- MacBook Pro M3 Max (ARM), 64GB RAM, on macOS 14.7
/text[42]-All experiments on the AWS EC2 VM...
/table[1]/caption[1]-
```

This is a strong README candidate because it shows that the scope changes the
kind of evidence returned, not merely the number of lines.

### `page` versus both default and `auto`

The first table header occurs immediately after a page break. With no explicit
scope, context uses eligible document reading order and reaches onto the
previous page:

```console
$ dlgrep -F Asset paper.dclg --type table_cell -C 3 -n
/list[4]/ldiv[2]-- MacBook Pro M3 Max ...
/text[42]-All experiments on the AWS EC2 VM...
/table[1]/caption[1]-
/table[1]/ched[1]:Asset
/table[1]/ched[2]-Version
/table[1]/ched[3]-OCR
/table[1]/ched[4]-Layout
```

`auto` identifies the hit as a table cell and stays inside the table:

```console
$ dlgrep -F Asset paper.dclg --type table_cell \
    -C 3 --context-scope auto -n
/table[1]/ched[1]:Asset
/table[1]/ched[2]-Version
/table[1]/ched[3]-OCR
/table[1]/ched[4]-Layout
```

`page` retains the table caption on the current page but excludes the list and
paragraph from the previous page:

```console
$ dlgrep -F Asset paper.dclg --type table_cell \
    -C 3 --context-scope page -n
/table[1]/caption[1]-
/table[1]/ched[1]:Asset
/table[1]/ched[2]-Version
/table[1]/ched[3]-OCR
/table[1]/ched[4]-Layout
```

This is the clearest verified justification for an explicit `page` scope.

### `container` versus `auto` for nested groups

The comprehensive fixture has a section containing a nested subsection.
`auto` chooses section context:

```console
$ dlgrep -F 'Introduction text' comprehensive.dclg \
    -C 3 --context-scope auto -n
/group[12]/heading[1]-Section Title
/group[12]/text[1]:Introduction text
/group[12]/group[1]/heading[1]-Subsection
/group[12]/group[1]/text[1]-Subsection content
/text[21]-Lorem ipsum...
```

Explicit `container` stays with the immediate semantic siblings:

```console
$ dlgrep -F 'Introduction text' comprehensive.dclg \
    -C 3 --context-scope container -n
/group[12]/text[1]:Introduction text
/group[12]/group[1]/heading[1]-Subsection
```

This is valid evidence for `container`, although the paper examples are more
compelling for the public README.

## Recommended README additions

Keep the addition small:

1. Add the paper outline and the `5.2 System Configurations` `show --section`
   workflow.
2. Add two `select` examples: page count and table-caption extraction.
3. Add the list-item `auto` versus `section` comparison.
4. Add the table-header `page` comparison if one more example fits.

Do not add every command above. The comprehensive fixture is best kept as
implementation evidence; the paper produces the clearer public narrative.
