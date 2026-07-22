# dlgrep

`dlgrep` is a deterministic, read-only grep for DocLang documents. It searches
Docling semantic units while returning reusable DocLang XPath addresses.

```bash
uv sync

uv run dlgrep -i 'termination|cancellation' contract.dclx -C 2 --format json
uv run dlgrep show contract.dclx '/d:doclang/d:heading[4]' --section
uv run dlgrep select contract.dclx '/d:doclang//d:table'
uv run dlgrep outline contract.dclx --format json
uv run dlgrep inspect contract.dclx --format json
```

Supported inputs are `.dclg`, DocLang `.xml`, `.dclx`, and standard input.
Search supports regex and fixed strings, repeated or file-supplied patterns,
case and word matching, semantic `-A`/`-B`/`-C` context, XPath/section/page/layer/type
filters, bounded text/JSON/JSONL output, and grep exit codes (`0` match, `1` no
match, `2` error).

Every semantic result is bound during `docling-core` deserialization. Ordinary
items, virtual list items, threaded fragments, and OTSL table cells therefore
retain source identity without matching normalized text back to XML.

The optional `image` command, physical thread-fragment search, and iteration-2
rich field/list/picture context are intentionally deferred.
