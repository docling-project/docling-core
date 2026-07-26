"""Command-line interface for dlgrep."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

import click
import typer
from docling_core.experimental.serializer.outline import (
    OutlineDocSerializer,
    OutlineFormat,
    OutlineMode,
    OutlineParams,
)
from docling_core.transforms.deserializer import DocLangSourceTarget
from docling_core.transforms.serializer.plain_text import PlainTextDocSerializer, PlainTextParams
from docling_core.types.doc import (
    SectionHeaderItem,
    TableItem,
    TitleItem,
)
from lxml import etree

from dlgrep import __version__
from dlgrep.document import DlgrepError, LoadedDocument, Unit, _canonical_xpath, _is_element

DEFAULT_LIMIT = 20
DEFAULT_MAX_CHARS = 2_000
DEFAULT_MAX_OUTPUT_CHARS = 20_000
HARD_LIMIT = 10_000
HARD_MAX_CHARS = 1_000_000
HARD_MAX_OUTPUT_CHARS = 10_000_000

OutputFormat = Literal["text", "json", "jsonl"]
ContextScope = Literal["auto", "container", "section", "page", "document"]


class _DefaultCommandGroup(typer.core.TyperGroup):
    """Route bare grep arguments to the search command."""

    default_command = "search"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in {"--help", "-h"}:
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="dlgrep",
    cls=_DefaultCommandGroup,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Search semantic units in DocLang documents and return XPath addresses.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def _record_context(ctx: typer.Context, param: click.Parameter, value: int | None) -> int | None:
    """Record context flags in command-line order, matching grep semantics."""
    if value is not None and isinstance(param, click.Option):
        ctx.meta.setdefault("context_events", []).append((param.opts[0], value))
    return value


def _exit(code: int) -> None:
    if code:
        raise typer.Exit(code)


@app.command()
def search(
    ctx: typer.Context,
    arguments: Annotated[
        list[str],
        typer.Argument(..., metavar="PATTERN/INPUT", help="Pattern followed by one or more DocLang inputs."),
    ],
    regexp: Annotated[list[str] | None, typer.Option("-e", "--regexp", help="Add a search pattern.")] = None,
    pattern_files: Annotated[list[str] | None, typer.Option("-f", "--file", help="Read patterns from a file.")] = None,
    fixed_strings: Annotated[bool, typer.Option("-F", "--fixed-strings", help="Match literal strings.")] = False,
    ignore_case: Annotated[bool, typer.Option("-i", "--ignore-case", help="Ignore case distinctions.")] = False,
    word_regexp: Annotated[bool, typer.Option("-w", "--word-regexp", help="Match whole words.")] = False,
    after_context: Annotated[
        int | None,
        typer.Option("-A", "--after-context", min=0, callback=_record_context, help="Show semantic units after a hit."),
    ] = None,
    before_context: Annotated[
        int | None,
        typer.Option(
            "-B", "--before-context", min=0, callback=_record_context, help="Show semantic units before a hit."
        ),
    ] = None,
    context: Annotated[
        int | None,
        typer.Option("-C", "--context", min=0, callback=_record_context, help="Show semantic units around a hit."),
    ] = None,
    context_scope: Annotated[ContextScope, typer.Option(help="Boundary used for semantic context.")] = "document",
    types: Annotated[list[str] | None, typer.Option("--type", help="Filter semantic unit types.")] = None,
    class_name: Annotated[str | None, typer.Option("--class", help="Filter Docling item classes.")] = None,
    layer: Annotated[
        Literal["body", "furniture", "background", "all"], typer.Option(help="Filter content layers.")
    ] = "body",
    page: Annotated[str | None, typer.Option(help="Filter pages, for example 1,3-5.")] = None,
    within_xpath: Annotated[str | None, typer.Option(help="Search only within an XPath selection.")] = None,
    section: Annotated[
        bool, typer.Option("--section", help="Treat --within-xpath headings as section boundaries.")
    ] = False,
    view: Annotated[Literal["visible", "metadata", "all"], typer.Option(help="Text projection to search.")] = "visible",
    offset: Annotated[int, typer.Option(min=0, help="Skip this many matches.")] = 0,
    limit: Annotated[int | None, typer.Option(min=0, max=HARD_LIMIT, help="Maximum matches to return.")] = None,
    max_chars: Annotated[int | None, typer.Option(min=0, help="Maximum characters per result.")] = None,
    max_output_chars: Annotated[int | None, typer.Option(min=0, help="Maximum characters across results.")] = None,
    all_results: Annotated[bool, typer.Option("--all", help="Use hard limits instead of default limits.")] = False,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Return only the grep exit status.")] = False,
    count: Annotated[bool, typer.Option("-c", "--count", help="Print match counts.")] = False,
    files_with_matches: Annotated[
        bool, typer.Option("-l", "--files-with-matches", help="Print names of matching inputs.")
    ] = False,
    with_xpath: Annotated[
        bool, typer.Option("-n", "--with-xpath", help="Prefix text output with XPath addresses.")
    ] = False,
    output_format: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "text",
    validate: Annotated[bool, typer.Option("--validate", help="Validate DocLang before searching.")] = False,
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
    ] = False,
) -> None:
    """Search DocLang semantic units; the command name may be omitted."""
    _exit(
        _search(
            SimpleNamespace(
                arguments=arguments,
                regexp=regexp or [],
                file=pattern_files or [],
                fixed_strings=fixed_strings,
                ignore_case=ignore_case,
                word_regexp=word_regexp,
                context_events=ctx.meta.get("context_events", []),
                context_scope=context_scope,
                types=types or [],
                class_name=class_name,
                layer=layer,
                page=page,
                within_xpath=within_xpath,
                section=section,
                view=view,
                offset=offset,
                limit=limit,
                max_chars=max_chars,
                max_output_chars=max_output_chars,
                all_results=all_results,
                quiet=quiet,
                count=count,
                files_with_matches=files_with_matches,
                with_xpath=with_xpath,
                format=output_format,
                validate=validate,
            )
        )
    )


@app.command("inspect", no_args_is_help=True)
def inspect_command(
    inputs: Annotated[list[str], typer.Argument(..., metavar="INPUT", help="DocLang inputs to inspect.")],
    output_format: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "text",
    validate: Annotated[bool, typer.Option("--validate", help="Validate DocLang before inspecting.")] = False,
) -> None:
    """Inspect raw and semantic document structure."""
    _exit(_inspect(SimpleNamespace(inputs=inputs, format=output_format, validate=validate)))


@app.command("outline", no_args_is_help=True)
def outline_command(
    input_: Annotated[str, typer.Argument(..., metavar="INPUT", help="DocLang input.")],
    depth: Annotated[int | None, typer.Option(min=0, help="Maximum heading depth.")] = None,
    output_format: Annotated[Literal["text", "json"], typer.Option("--format", help="Output format.")] = "text",
    validate: Annotated[bool, typer.Option("--validate", help="Validate DocLang before outlining.")] = False,
) -> None:
    """Print the semantic heading outline with XPath addresses."""
    _exit(_outline(SimpleNamespace(input=input_, depth=depth, format=output_format, validate=validate)))


@app.command("select", no_args_is_help=True)
def select_command(
    input_: Annotated[str, typer.Argument(..., metavar="INPUT", help="DocLang input.")],
    xpath: Annotated[str, typer.Argument(..., metavar="XPATH", help="XPath expression.")],
    semantic: Annotated[bool, typer.Option("--semantic", help="Include semantic source bindings.")] = False,
    limit: Annotated[int | None, typer.Option(min=0, max=HARD_LIMIT, help="Maximum results to return.")] = None,
    max_chars: Annotated[int | None, typer.Option(min=0, help="Maximum characters per result.")] = None,
    all_results: Annotated[bool, typer.Option("--all", help="Use hard limits instead of default limits.")] = False,
    output_format: Annotated[
        Literal["xml", "text", "json", "jsonl"], typer.Option("--format", help="Output format.")
    ] = "xml",
    validate: Annotated[bool, typer.Option("--validate", help="Validate DocLang before selecting.")] = False,
) -> None:
    """Evaluate an XPath expression against the original DocLang XML."""
    _exit(
        _select(
            SimpleNamespace(
                input=input_,
                xpath=xpath,
                semantic=semantic,
                limit=limit,
                max_chars=max_chars,
                all_results=all_results,
                format=output_format,
                validate=validate,
            )
        )
    )


@app.command("show", no_args_is_help=True)
def show_command(
    ctx: typer.Context,
    input_: Annotated[str, typer.Argument(..., metavar="INPUT", help="DocLang input.")],
    xpath: Annotated[str, typer.Argument(..., metavar="XPATH", help="XPath selecting elements.")],
    raw: Annotated[bool, typer.Option("--raw", help="Return the original XML.")] = False,
    section: Annotated[bool, typer.Option("--section", help="Return the selected heading section.")] = False,
    after_context: Annotated[
        int | None,
        typer.Option("-A", "--after-context", min=0, callback=_record_context, help="Show semantic units after it."),
    ] = None,
    before_context: Annotated[
        int | None,
        typer.Option("-B", "--before-context", min=0, callback=_record_context, help="Show semantic units before it."),
    ] = None,
    context: Annotated[
        int | None,
        typer.Option("-C", "--context", min=0, callback=_record_context, help="Show semantic units around it."),
    ] = None,
    context_scope: Annotated[ContextScope, typer.Option(help="Boundary used for semantic context.")] = "document",
    max_chars: Annotated[int, typer.Option(min=0, help="Maximum characters per result.")] = DEFAULT_MAX_CHARS,
    with_xpath: Annotated[
        bool, typer.Option("-n", "--with-xpath", help="Prefix text output with XPath addresses.")
    ] = False,
    output_format: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = "text",
    validate: Annotated[bool, typer.Option("--validate", help="Validate DocLang before retrieval.")] = False,
) -> None:
    """Retrieve semantic content at one or more XPath addresses."""
    _exit(
        _show(
            SimpleNamespace(
                input=input_,
                xpath=xpath,
                raw=raw,
                section=section,
                context_events=ctx.meta.get("context_events", []),
                context_scope=context_scope,
                max_chars=max_chars,
                with_xpath=with_xpath,
                format=output_format,
                validate=validate,
            )
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run dlgrep and return its grep-compatible exit status."""
    try:
        result = typer.main.get_command(app).main(
            args=list(argv) if argv is not None else None,
            prog_name="dlgrep",
            standalone_mode=False,
        )
        return result if isinstance(result, int) else 0
    except DlgrepError as exc:
        print(f"dlgrep: {exc}", file=sys.stderr)
        return 2
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return exc.exit_code


def _search(args: SimpleNamespace) -> int:
    patterns, inputs, stdin_bytes = _patterns_and_inputs(args)
    regexes = _compile_patterns(patterns, fixed=args.fixed_strings, ignore_case=args.ignore_case, word=args.word_regexp)
    before, after = _context_counts(args.context_events)
    _validate_search_options(args, before, after)

    requested_types = _requested_types(args.types)
    pages = _parse_pages(args.page) if args.page else None
    hits: list[tuple[LoadedDocument, Unit, str, list[dict[str, Any]], list[Unit]]] = []
    counts: list[tuple[str, int]] = []
    errors = False

    for source in inputs:
        try:
            loaded = LoadedDocument.load(
                source, stdin_bytes=stdin_bytes if source == "-" else None, validate=args.validate
            )
            candidates, context_units = _filtered_units(loaded, args, requested_types, pages)
            input_hits = 0
            seen: set[str] = set()
            for unit in candidates:
                matches = _matches(_projected_text(unit, args.view), regexes)
                if not matches:
                    continue
                display = loaded.display_unit(unit)
                if display.xpath in seen:
                    continue
                seen.add(display.xpath)
                text = _projected_text(display, args.view)
                input_hits += 1
                hits.append((loaded, display, text, _matches(text, regexes), context_units))
            counts.append((source, input_hits))
        except DlgrepError as exc:
            errors = True
            print(f"dlgrep: {source}: {exc}", file=sys.stderr)

    any_match = any(count for _, count in counts)
    if args.quiet:
        return 2 if errors else (0 if any_match else 1)
    if args.count or args.files_with_matches:
        _render_summary(counts, files_only=args.files_with_matches, output_format=args.format)
        return 2 if errors else (0 if any_match else 1)

    limit = args.limit if args.limit is not None else (HARD_LIMIT if args.all_results else DEFAULT_LIMIT)
    selected = hits[args.offset : args.offset + limit]
    max_chars = (
        min(args.max_chars, HARD_MAX_CHARS)
        if args.max_chars is not None
        else (HARD_MAX_CHARS if args.all_results else DEFAULT_MAX_CHARS)
    )
    records: list[dict[str, Any]] = []
    for loaded, unit, text, matches, context_units in selected:
        context_before, context_after, context_group = loaded.context_for(
            unit, before, after, args.context_scope, context_units
        )
        records.append(
            _result_record(
                loaded,
                unit,
                text,
                matches,
                context_before,
                context_after,
                max_chars=max_chars,
                context_group=context_group,
            )
        )
    max_output = (
        min(args.max_output_chars, HARD_MAX_OUTPUT_CHARS)
        if args.max_output_chars is not None
        else (HARD_MAX_OUTPUT_CHARS if args.all_results else DEFAULT_MAX_OUTPUT_CHARS)
    )
    _bound_record_text(records, max_output)
    _render_records(
        records,
        args.format,
        with_xpath=args.with_xpath,
        with_filename=len(inputs) > 1,
        context_requested=bool(before or after),
    )
    return 2 if errors else (0 if any_match else 1)


def _inspect(args: SimpleNamespace) -> int:
    if args.inputs.count("-") > 1:
        raise DlgrepError("standard input may be used only once")
    stdin_bytes = sys.stdin.buffer.read() if "-" in args.inputs else None
    records = [
        LoadedDocument.load(
            source,
            stdin_bytes=stdin_bytes if source == "-" else None,
            validate=args.validate,
        ).inventory()
        for source in args.inputs
    ]
    if args.format == "json":
        print(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.format == "jsonl":
        for record in records:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    else:
        for index, record in enumerate(records):
            if index:
                print("--")
            print(record["document"])
            print(f"Type: {record['input_type']}")
            print(f"Pages: {record['page_count']}")
            print(f"Semantic units: {record['semantic_units']}")
            print("Elements: " + ", ".join(f"{key}={value}" for key, value in record["elements"].items()))
            if record["metadata_elements"]:
                print("Metadata: " + ", ".join(f"{key}={value}" for key, value in record["metadata_elements"].items()))
    return 0


def _outline(args: SimpleNamespace) -> int:
    loaded = _load_command_input(args.input, validate=args.validate)
    serialized = OutlineDocSerializer(
        doc=loaded.document,
        params=OutlineParams(
            include_non_meta=True,
            mode=OutlineMode.TABLE_OF_CONTENTS,
            format=OutlineFormat.JSON,
        ),
    ).serialize()
    items_by_ref = {item.self_ref: item for item, _ in loaded.document.iterate_items(with_groups=True)}
    records: list[dict[str, Any]] = []
    for outline_item in json.loads(serialized.text):
        item = items_by_ref.get(outline_item["ref"])
        if not isinstance(item, (TitleItem, SectionHeaderItem)):
            continue
        heading_depth = len(_heading_ancestors(item, loaded)) + 1
        if args.depth is not None and heading_depth > args.depth:
            continue
        target = DocLangSourceTarget(kind="item", item_ref=item.self_ref)
        unit = loaded.unit_for_target(target)
        if unit is not None:
            record = _unit_record(loaded, unit, text=outline_item["title"])
            record["depth"] = heading_depth
            records.append(record)
    if args.format == "json":
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        for record in records:
            print(f"{'  ' * (record['depth'] - 1)}{record['text']}\t{record['xpaths'][0]}")
    return 0


def _select(args: SimpleNamespace) -> int:
    if (args.limit is not None and args.limit < 0) or (args.max_chars is not None and args.max_chars < 0):
        raise DlgrepError("limits must be non-negative")
    if args.limit is not None and args.limit > HARD_LIMIT:
        raise DlgrepError(f"--limit cannot exceed {HARD_LIMIT}")
    loaded = _load_command_input(args.input, validate=args.validate)
    selected = loaded.evaluate_xpath(args.xpath)
    if not isinstance(selected, list):
        scalar_record = {"document": args.input, "sha256": loaded.sha256, "value": selected}
        if args.format == "json":
            print(json.dumps(scalar_record, ensure_ascii=False, indent=2))
        elif args.format == "jsonl":
            print(json.dumps(scalar_record, ensure_ascii=False, sort_keys=True))
        else:
            print(selected)
        return 0

    limit = min(
        args.limit if args.limit is not None else (HARD_LIMIT if args.all_results else DEFAULT_LIMIT),
        HARD_LIMIT,
    )
    max_chars = (
        min(args.max_chars, HARD_MAX_CHARS)
        if args.max_chars is not None
        else (HARD_MAX_CHARS if args.all_results else DEFAULT_MAX_CHARS)
    )
    records: list[dict[str, Any]] = []
    for value in selected[:limit]:
        if _is_element(value):
            xpath = _canonical_xpath(value)
            raw_element = loaded.raw_elements[xpath]
            xml = etree.tostring(raw_element, encoding="unicode", with_tail=False)
            bounded, truncated = _truncate(xml, max_chars)
            record: dict[str, Any] = {"xpaths": [xpath], "xml": bounded}
            if truncated:
                record["truncated"] = True
            if args.semantic and (target := loaded.source_map.targets_by_xpath.get(xpath)) is not None:
                record["semantic"] = {key: value for key, value in asdict(target).items() if value is not None}
            records.append(record)
        else:
            records.append({"value": str(value)})

    if args.format == "xml":
        for record in records:
            print(record.get("xml", record.get("value", "")))
    elif args.format == "text":
        for record in records:
            if "xpaths" in record:
                selected_xpath = record["xpaths"][0]
                if not isinstance(selected_xpath, str):
                    raise DlgrepError("invalid internal XPath result")
                element = loaded.raw_elements[selected_xpath]
                text = etree.tostring(element, method="text", encoding="unicode", with_tail=False)
                print(_truncate(text, max_chars)[0])
            else:
                print(record["value"])
    elif args.format == "jsonl":
        for record in records:
            print(
                json.dumps(
                    {"document": args.input, "sha256": loaded.sha256, **record}, ensure_ascii=False, sort_keys=True
                )
            )
    else:
        print(
            json.dumps(
                {"document": args.input, "sha256": loaded.sha256, "results": records}, ensure_ascii=False, indent=2
            )
        )
    return 0


def _show(args: SimpleNamespace) -> int:
    if args.max_chars < 0:
        raise DlgrepError("--max-chars must be non-negative")
    loaded = _load_command_input(args.input, validate=args.validate)
    selected = loaded.evaluate_xpath(args.xpath)
    if not isinstance(selected, list) or not selected or not all(_is_element(value) for value in selected):
        raise DlgrepError("show XPath must select one or more elements")
    before_count, after_count = _context_counts(args.context_events)
    context_requested = bool(args.context_events)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for element in selected:
        xpath = _canonical_xpath(element)
        raw_element = loaded.raw_elements[xpath]
        if args.raw:
            text, truncated = _truncate(
                etree.tostring(raw_element, encoding="unicode", with_tail=False), args.max_chars
            )
            records.append(
                {
                    "document": args.input,
                    "xpaths": [xpath],
                    "logical_type": "raw",
                    "text": text,
                    **({"truncated": True} if truncated else {}),
                }
            )
            continue
        target = loaded.source_map.targets_by_xpath.get(xpath)
        unit = loaded.unit_for_target(target) if target is not None else None
        if target is None or unit is None:
            text, truncated = _truncate(
                etree.tostring(raw_element, encoding="unicode", with_tail=False), args.max_chars
            )
            records.append(
                {
                    "document": args.input,
                    "xpaths": [xpath],
                    "logical_type": "raw",
                    "text": text,
                    **({"truncated": True} if truncated else {}),
                }
            )
            continue
        if unit.xpath in seen:
            continue
        seen.add(unit.xpath)
        text = unit.text
        logical_type = unit.logical_type
        doc_items: tuple[str, ...] | None = unit.doc_items
        if args.section:
            item = loaded.target_item(target)
            if not isinstance(item, (TitleItem, SectionHeaderItem)):
                raise DlgrepError("--section requires a heading XPath")
            serializer = PlainTextDocSerializer(
                doc=loaded.document,
                params=PlainTextParams(allowed_meta_names=set(), include_annotations=False),
            )
            serialized = serializer.serialize_doc(parts=serializer.get_parts(item=item))
            text = serialized.text
            doc_items = tuple(dict.fromkeys(span.item.self_ref for span in serialized.spans))
            logical_type = "section"
        before, after, context_group = loaded.context_for(
            unit,
            before_count,
            after_count,
            args.context_scope,
            loaded.context_units,
            include_overlapping_before=context_requested and loaded.display_unit(unit) is not unit,
        )
        record = _result_record(
            loaded,
            unit,
            text,
            None,
            before,
            after,
            max_chars=args.max_chars,
            doc_items=doc_items,
            context_group=context_group,
        )
        record["logical_type"] = logical_type
        records.append(record)
    _render_records(
        records,
        args.format,
        with_xpath=args.with_xpath,
        context_requested=context_requested,
    )
    return 0


def _patterns_and_inputs(args: SimpleNamespace) -> tuple[list[str], list[str], bytes | None]:
    stdin_bytes: bytes | None = None
    patterns = list(args.regexp)
    if args.regexp or args.file:
        inputs = list(args.arguments)
    else:
        if len(args.arguments) < 2:
            raise DlgrepError("provide a PATTERN and at least one INPUT")
        patterns = [args.arguments[0]]
        inputs = list(args.arguments[1:])

    if inputs.count("-") > 1:
        raise DlgrepError("standard input may be used as a document only once")
    if "-" in args.file and "-" in inputs:
        raise DlgrepError("standard input cannot supply both patterns and a document")
    if "-" in args.file or "-" in inputs:
        stdin_bytes = sys.stdin.buffer.read()

    for filename in args.file:
        try:
            content = (
                stdin_bytes.decode("utf-8")
                if filename == "-" and stdin_bytes is not None
                else Path(filename).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise DlgrepError(f"could not read pattern file {filename!r}: {exc}") from exc
        patterns.extend(content.splitlines())
    return patterns, inputs, stdin_bytes


def _load_command_input(source: str, *, validate: bool) -> LoadedDocument:
    return LoadedDocument.load(
        source,
        stdin_bytes=sys.stdin.buffer.read() if source == "-" else None,
        validate=validate,
    )


def _compile_patterns(patterns: Iterable[str], *, fixed: bool, ignore_case: bool, word: bool) -> list[re.Pattern[str]]:
    flags = re.IGNORECASE if ignore_case else 0
    regexes: list[re.Pattern[str]] = []
    for pattern in patterns:
        expression = re.escape(pattern) if fixed else pattern
        if word:
            expression = rf"(?<!\w)(?:{expression})(?!\w)"
        try:
            regexes.append(re.compile(expression, flags))
        except re.error as exc:
            raise DlgrepError(f"invalid regular expression {pattern!r}: {exc}") from exc
    return regexes


def _matches(text: str, regexes: Iterable[re.Pattern[str]]) -> list[dict[str, Any]]:
    matches = {(match.start(), match.end(), match.group(0)) for regex in regexes for match in regex.finditer(text)}
    return [{"start": start, "end": end, "text": value} for start, end, value in sorted(matches)]


def _filtered_units(
    loaded: LoadedDocument,
    args: SimpleNamespace,
    requested_types: set[str],
    pages: set[int] | None,
) -> tuple[list[Unit], list[Unit]]:
    within_paths: set[str] | None = None
    section_refs: set[str] = set()
    if args.within_xpath:
        within_paths = loaded.selected_paths(args.within_xpath)
        if args.section:
            for xpath in within_paths:
                target = loaded.source_map.targets_by_xpath.get(xpath)
                item = loaded.target_item(target) if target is not None else None
                if not isinstance(item, (TitleItem, SectionHeaderItem)):
                    raise DlgrepError("--section requires --within-xpath to select headings")
                section_refs.add(item.self_ref)
    elif args.section:
        raise DlgrepError("--section requires --within-xpath")

    def boundary(unit: Unit) -> bool:
        if args.layer != "all" and unit.layer != args.layer:
            return False
        if pages is not None and not (set(unit.pages) & pages):
            return False
        if section_refs and not any(loaded.is_descendant(unit.item_ref, ref) for ref in section_refs):
            return False
        if (
            within_paths
            and not section_refs
            and not any(
                xpath == parent or xpath.startswith(parent + "/") for xpath in unit.xpaths for parent in within_paths
            )
        ):
            return False
        return True

    context_units = [unit for unit in loaded.context_units if boundary(unit)]
    if not ({"page_header", "page_footer"} & requested_types):
        context_units = [unit for unit in context_units if unit.logical_type not in {"page_header", "page_footer"}]
    candidates = [unit for unit in loaded.units if boundary(unit)]
    if requested_types:
        candidates = [unit for unit in candidates if _type_matches(unit.logical_type, requested_types)]
    else:
        candidates = [
            unit
            for unit in candidates
            if not unit.container
            and unit.logical_type not in {"page_header", "page_footer"}
            and loaded.display_unit(unit) is unit
        ]
    if args.class_name:
        candidates = [unit for unit in candidates if _class_matches(loaded, unit, args.class_name)]
    return candidates, context_units


def _requested_types(values: Iterable[str]) -> set[str]:
    aliases = {
        "title": "heading",
        "section_header": "heading",
        "key": "field_key",
        "value": "field_value",
        "hint": "field_hint",
    }
    result: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                result.add(aliases.get(item, item))
    return result


def _type_matches(logical_type: str, requested: set[str]) -> bool:
    return logical_type in requested or ("table_cell" in requested and logical_type in {"index_cell", "chart_cell"})


def _class_matches(loaded: LoadedDocument, unit: Unit, class_name: str) -> bool:
    if unit.item is not None and type(unit.item).__name__.casefold() == class_name.casefold():
        return True
    return any(loaded.raw_elements[xpath].get("class") == class_name for xpath in unit.xpaths)


def _projected_text(unit: Unit, view: str) -> str:
    if view == "metadata":
        return unit.metadata
    if view == "all":
        return unit.all_text
    return unit.text


def _parse_pages(value: str) -> set[int]:
    pages: set[int] = set()
    for member in value.split(","):
        if not member:
            raise DlgrepError("empty member in --page")
        if "-" in member:
            start_text, end_text = member.split("-", maxsplit=1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise DlgrepError(f"invalid page range: {member!r}")
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise DlgrepError(f"invalid page range: {member!r}")
            pages.update(range(start, end + 1))
        elif member.isdigit() and int(member) > 0:
            pages.add(int(member))
        else:
            raise DlgrepError(f"invalid page number: {member!r}")
    return pages


def _context_counts(events: Iterable[tuple[str, int]]) -> tuple[int, int]:
    before = after = 0
    for option, value in events:
        if value < 0:
            raise DlgrepError("context counts must be non-negative")
        if option in {"-C", "--context"}:
            before = after = value
        elif option in {"-B", "--before-context"}:
            before = value
        else:
            after = value
    return before, after


def _validate_search_options(args: SimpleNamespace, before: int, after: int) -> None:
    for name, value in {
        "--offset": args.offset,
        "--limit": args.limit,
        "--max-chars": args.max_chars,
        "--max-output-chars": args.max_output_chars,
    }.items():
        if value is not None and value < 0:
            raise DlgrepError(f"{name} must be non-negative")
    if args.limit is not None and args.limit > HARD_LIMIT:
        raise DlgrepError(f"--limit cannot exceed {HARD_LIMIT}")
    if sum((args.count, args.files_with_matches, args.quiet)) > 1:
        raise DlgrepError("choose only one of --count, --files-with-matches, or --quiet")
    if (args.count or args.files_with_matches or args.quiet) and (
        before
        or after
        or args.offset
        or args.limit is not None
        or args.max_chars is not None
        or args.max_output_chars is not None
        or args.all_results
    ):
        raise DlgrepError("summary modes cannot be combined with context, offset, limit, or --all")


def _result_record(
    loaded: LoadedDocument,
    unit: Unit,
    text: str,
    matches: list[dict[str, Any]] | None,
    before: list[Unit],
    after: list[Unit],
    *,
    max_chars: int,
    doc_items: tuple[str, ...] | None = None,
    context_group: tuple[str, int, int] | None = None,
) -> dict[str, Any]:
    bounded, truncated = _truncate(text, max_chars)
    record = _unit_record(loaded, unit, text=bounded, doc_items=doc_items)
    if matches is not None:
        record["matches"] = matches
    if before or after:
        record["context"] = {
            "before": [_context_record(loaded, value, max_chars) for value in before],
            "after": [_context_record(loaded, value, max_chars) for value in after],
        }
    if table := _table_record(loaded, unit):
        record["table"] = table
    if cell_context := _cell_context(loaded, unit):
        record["cell_context"] = cell_context
    if truncated:
        record["truncated"] = True
    record["_context_group"] = context_group
    return record


def _unit_record(
    loaded: LoadedDocument,
    unit: Unit,
    *,
    text: str | None = None,
    doc_items: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "document": loaded.name,
        "xpaths": list(unit.xpaths),
        "logical_type": unit.logical_type,
        "text": unit.text if text is None else text,
        "pages": sorted(set(unit.pages)),
        "doc_items": list(dict.fromkeys(doc_items if doc_items is not None else unit.doc_items)),
    }


def _context_record(loaded: LoadedDocument, unit: Unit, max_chars: int) -> dict[str, Any]:
    text, truncated = _truncate(unit.text, max_chars)
    record = _unit_record(loaded, unit, text=text)
    if truncated:
        record["truncated"] = True
    return record


def _table_record(loaded: LoadedDocument, unit: Unit) -> dict[str, Any]:
    item = unit.item or loaded.target_item(unit.target)
    if unit.logical_type != "table" or not isinstance(item, TableItem):
        return {}
    cells = []
    for cell in sorted(
        item.data.table_cells, key=lambda value: (value.start_row_offset_idx, value.start_col_offset_idx)
    ):
        record: dict[str, Any] = {
            "row": cell.start_row_offset_idx,
            "column": cell.start_col_offset_idx,
            "text": cell.text,
        }
        if cell.column_header:
            record["role"] = "column_header"
        elif cell.row_header:
            record["role"] = "row_header"
        cells.append(record)
    result: dict[str, Any] = {"cells": cells}
    if caption := item.caption_text(loaded.document):
        result["caption"] = caption
    return result


def _cell_context(loaded: LoadedDocument, unit: Unit) -> dict[str, Any]:
    if unit.target.kind != "table_cell" or unit.item_ref is None or unit.row is None or unit.col is None:
        return {}
    item = loaded.target_item(unit.target)
    if not isinstance(item, TableItem):
        return {"row": unit.row, "column": unit.col}
    grid = item.data.grid
    row_headers = [cell.text for cell in grid[unit.row] if cell.row_header and cell.text]
    column_headers = [row[unit.col].text for row in grid if row[unit.col].column_header and row[unit.col].text]
    result: dict[str, Any] = {"row": unit.row, "column": unit.col}
    if row_headers:
        result["row_headers"] = list(dict.fromkeys(row_headers))
    if column_headers:
        result["column_headers"] = list(dict.fromkeys(column_headers))
    if caption := item.caption_text(loaded.document):
        result["caption"] = caption
    return result


def _heading_ancestors(item: Any, loaded: LoadedDocument) -> list[Any]:
    result: list[Any] = []
    while item.parent is not None:
        item = item.parent.resolve(loaded.document)
        if isinstance(item, (TitleItem, SectionHeaderItem)):
            result.append(item)
    return list(reversed(result))


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 1)] + ("…" if limit else ""), True


def _bound_record_text(records: list[dict[str, Any]], limit: int) -> None:
    remaining = limit
    for record in records:
        context = record.get("context", {"before": [], "after": []})
        values = [*context["before"], record, *context["after"]]
        for value in values:
            text = value["text"]
            if len(text) <= remaining:
                remaining -= len(text)
                continue
            value["text"], _ = _truncate(text, remaining)
            value["truncated"] = True
            remaining = 0


def _render_records(
    records: list[dict[str, Any]],
    output_format: str,
    *,
    with_xpath: bool = False,
    with_filename: bool = False,
    context_requested: bool = False,
) -> None:
    if output_format == "json":
        print(
            json.dumps(
                [_public_record(record) for record in records],
                ensure_ascii=False,
                indent=2,
            )
        )
    elif output_format == "jsonl":
        for record in records:
            print(json.dumps(_public_record(record), ensure_ascii=False, sort_keys=True))
    else:
        hits = {(record["document"], record["xpaths"][0]): record for record in records}
        groups: list[dict[str, Any]] = []
        for record in records:
            context_values = record.get("context", {"before": [], "after": []})
            values = [*context_values["before"], record, *context_values["after"]]
            sequence, start, end = record.get("_context_group") or ((), -1, -1)
            if (
                context_requested
                and groups
                and groups[-1]["document"] == record["document"]
                and sequence
                and groups[-1]["sequence"] == sequence
                and start <= groups[-1]["end"] + 1
            ):
                known = {value["xpaths"][0] for value in groups[-1]["values"]}
                groups[-1]["values"].extend(value for value in values if value["xpaths"][0] not in known)
                groups[-1]["end"] = max(groups[-1]["end"], end)
            else:
                groups.append(
                    {
                        "document": record["document"],
                        "sequence": sequence,
                        "end": end,
                        "values": values,
                    }
                )

        for index, group in enumerate(groups):
            if index and context_requested:
                print("--")

            for value in group["values"]:
                hit = hits.get((group["document"], value["xpaths"][0]))
                separator = ":" if hit is not None else "-"
                text = _highlight(hit["text"], hit.get("matches", [])) if hit is not None else value["text"]
                fields = [group["document"]] if with_filename else []
                if with_xpath:
                    fields.append(value["xpaths"][0].removeprefix("/d:doclang").replace("/d:", "/"))
                prefix = separator.join(fields) + separator if fields else ""
                print("\n".join(prefix + line for line in text.split("\n")))


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _render_summary(counts: list[tuple[str, int]], *, files_only: bool, output_format: str) -> None:
    records = [
        ({"document": source} if files_only else {"document": source, "count": count})
        for source, count in counts
        if count or not files_only
    ]
    if output_format == "json":
        print(json.dumps(records, ensure_ascii=False, indent=2))
    elif output_format == "jsonl":
        for record in records:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    else:
        for record in records:
            print(record["document"] if files_only else f"{record['document']}:{record['count']}")


def _highlight(text: str, matches: list[dict[str, Any]]) -> str:
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ:
        return text
    spans: list[tuple[int, int]] = []
    for start, end in sorted((max(0, match["start"]), min(len(text), match["end"])) for match in matches):
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    for start, end in reversed(spans):
        if start < end:
            text = f"{text[:start]}\033[1;31m{text[start:end]}\033[0m{text[end:]}"
    return text
