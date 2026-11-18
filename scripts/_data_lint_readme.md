# `_data_lint.py`

Private data lint for `DoclingDocument` JSON and DocLang (`.dclx` archive, `.dclg` / `.xml` markup) files. It is not part of the
docling-core API and may change without notice.

```bash
uv run python scripts/_data_lint.py FILE.json                 # JSON checks
uv run python scripts/_data_lint.py FILE.dclx                 # DocLang checks, incl. the DCLX round trip
uv run python scripts/_data_lint.py FILE.dclx --pair          # also FILE.json from the same folder, plus the pair checks
uv run python scripts/_data_lint.py --list-checks             # what exists, grouped by type
uv run python scripts/_data_lint.py -q --only dclx.roundtrip *.dclx   # failures of one check over many files
uv run python scripts/_data_lint.py -p --report checks *.dclx     # counts per check over many files
uv run python scripts/_data_lint.py -p --report checks-with-files --only dclx.roundtrip *.dclx   # ... and the files behind the fail / warn counts
```

The kind of check is chosen by the extension (`.json`; `.dclx`, `.dclg`, `.xml`, also `X.dclg.xml`). With `--pair`, the counterpart is the same-stem file in the same folder, in both directions: `X.json` is matched with the
first existing of `X.dclx`, `X.dclg`, `X.dclg.xml`, `X.xml`, and any of those is matched with `X.json`. Several files can be given. `--only` / `--skip` take check slugs or dotted
prefixes (`dclx`, `json.rules`), repeatable or comma-separated. Unknown names, or an `--only` whose checks are all
skipped, are errors. `--report` selects what is printed: `files` (default) the results of each file, `checks` counts
(ok / warn / fail) per check over all files, `checks-with-files` the same plus the files behind the warn and fail
counts. `-q` prints only failures (with `--report files`). `--output json` prints machine-readable results, also for the
checks reports.

## Result and exit code

| Status | Meaning | Exit code |
|---|---|---|
| `fail` | a defect | 1 |
| `warn` | a lint or diagnostic that may be legitimate | 0 |

The last line is `RESULT: FAILED`, `RESULT: WARN (n warning(s))` or `RESULT: OK`.

## What is strict and what is not

- **Strict (DocLang level):** `dclx.roundtrip` (DCLX -> document -> DCLX) and `pair.json-to-dclx` compare the DocLang
  text line by line. Indentation outside `<content>` is normalized, `<content>` is compared verbatim because it
  preserves whitespace. If both pass, the JSON, the DCLX and the deserialized DCLX all serialize to the same DocLang.
- **Diagnostic (document level):** `pair.dclx-to-json` compares fingerprints of the JSON and of the document
  deserialized from the DCLX (item kinds, parent kinds, layers, texts, threads, bboxes; unordered, texts cut at
  80 characters, table cells and meta not compared). A `DoclingDocument` holds more than DocLang expresses, so
  differences can be legitimate. It only warns.
- **Heuristics:** `json.empty-list-wrapper` is a heuristic and only warns.

There is deliberately no JSON -> DocLang -> JSON check as a failure: it would flag information DocLang normalizes.

## Notes

- Grid: bboxes are compared on the document's DocLang grid (`<default_resolution>`, 512 when absent).
- Schematron needs the `doclang[schematron-saxon]` extra. Without it `dclx.schematron` fails with a hint, so a run never
  passes with a check silently missing; use `--skip dclx.schematron` to run without it.
- `DoclingDocument._validate_rules` is used as is, so rules added there show up here. A deserialized DCLX is also
  re-validated through the model (`DoclingDocument.model_validate`), because the deserializer builds documents through the
  API, which does not run the model validators; that is part of `dclx.rules`.
