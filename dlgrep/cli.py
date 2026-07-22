"""Command-line interface for dlgrep."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from docling_core.transforms.deserializer import DocLangSourceTarget
from docling_core.types.doc import (
    SectionHeaderItem,
    TableItem,
    TitleItem,
)
from lxml import etree

from dlgrep import __version__
from dlgrep.document import DlgrepError, LoadedDocument, Unit, _canonical_xpath, _is_element

COMMANDS = {"inspect", "outline", "select", "show"}
DEFAULT_LIMIT = 20
DEFAULT_MAX_CHARS = 2_000
DEFAULT_MAX_OUTPUT_CHARS = 20_000
HARD_LIMIT = 10_000
HARD_MAX_CHARS = 1_000_000
HARD_MAX_OUTPUT_CHARS = 10_000_000


class ContextAction(argparse.Action):
    """Record context flags in command-line order."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        events = list(getattr(namespace, self.dest, []) or [])
        events.append((option_string or "", int(values)))
        setattr(namespace, self.dest, events)


def main(argv: Sequence[str] | None = None) -> int:
    """Run dlgrep and return its grep-compatible exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--version"]:
        print(__version__)
        return 0
    try:
        if arguments and arguments[0] in COMMANDS:
            command = arguments.pop(0)
            return {
                "inspect": _inspect,
                "outline": _outline,
                "select": _select,
                "show": _show,
            }[command](_command_parser(command).parse_args(arguments))
        return _search(_search_parser().parse_args(arguments))
    except DlgrepError as exc:
        print(f"dlgrep: {exc}", file=sys.stderr)
        return 2


def _search_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlgrep",
        description="Search semantic units in DocLang documents and return XPath addresses.",
        epilog="Subcommands: inspect, outline, select, show",
        allow_abbrev=False,
    )
    parser.add_argument("-e", "--regexp", action="append", default=[], help="add a search pattern")
    parser.add_argument("-f", "--file", action="append", default=[], help="read patterns from a file")
    parser.add_argument("-F", "--fixed-strings", action="store_true")
    parser.add_argument("-i", "--ignore-case", action="store_true")
    parser.add_argument("-w", "--word-regexp", action="store_true")
    _add_context_arguments(parser)
    parser.add_argument("--type", dest="types", action="append", default=[])
    parser.add_argument("--class", dest="class_name")
    parser.add_argument("--layer", choices=["body", "furniture", "background", "all"], default="body")
    parser.add_argument("--page")
    parser.add_argument("--within-xpath")
    parser.add_argument("--section", action="store_true")
    parser.add_argument("--view", choices=["visible", "metadata", "all"], default="visible")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--max-output-chars", type=int)
    parser.add_argument("--all", dest="all_results", action="store_true")
    summaries = parser.add_mutually_exclusive_group()
    summaries.add_argument("-q", "--quiet", action="store_true")
    summaries.add_argument("-c", "--count", action="store_true")
    summaries.add_argument("-l", "--files-with-matches", action="store_true")
    parser.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("arguments", nargs="+", metavar="PATTERN/INPUT")
    return parser


def _command_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"dlgrep {command}", allow_abbrev=False)
    if command == "inspect":
        parser.add_argument("inputs", nargs="+")
        parser.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
        parser.add_argument("--validate", action="store_true")
    elif command == "outline":
        parser.add_argument("input")
        parser.add_argument("--depth", type=int)
        parser.add_argument("--format", choices=["text", "json"], default="text")
        parser.add_argument("--validate", action="store_true")
    elif command == "select":
        parser.add_argument("input")
        parser.add_argument("xpath")
        parser.add_argument("--semantic", action="store_true")
        parser.add_argument("--limit", type=int)
        parser.add_argument("--max-chars", type=int)
        parser.add_argument("--all", dest="all_results", action="store_true")
        parser.add_argument("--format", choices=["xml", "text", "json", "jsonl"], default="xml")
        parser.add_argument("--validate", action="store_true")
    elif command == "show":
        parser.add_argument("input")
        parser.add_argument("xpath")
        parser.add_argument("--raw", action="store_true")
        parser.add_argument("--section", action="store_true")
        _add_context_arguments(parser)
        parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
        parser.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
        parser.add_argument("--validate", action="store_true")
    return parser


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(context_events=[])
    parser.add_argument("-A", "--after-context", dest="context_events", action=ContextAction, type=int)
    parser.add_argument("-B", "--before-context", dest="context_events", action=ContextAction, type=int)
    parser.add_argument("-C", "--context", dest="context_events", action=ContextAction, type=int)
    parser.add_argument(
        "--context-scope",
        choices=["auto", "container", "section", "page", "document"],
        default="document",
    )
    parser.add_argument("--no-ancestors", action="store_true")


def _search(args: argparse.Namespace) -> int:
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
            for unit in candidates:
                text = _projected_text(unit, args.view)
                matches = _matches(text, regexes)
                if not matches:
                    continue
                input_hits += 1
                hits.append((loaded, unit, text, matches, context_units))
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
    records = [
        _result_record(
            loaded,
            unit,
            text,
            matches,
            *loaded.context_for(unit, before, after, args.context_scope, context_units),
            max_chars=max_chars,
            ancestors=not args.no_ancestors,
        )
        for loaded, unit, text, matches, context_units in selected
    ]
    max_output = (
        min(args.max_output_chars, HARD_MAX_OUTPUT_CHARS)
        if args.max_output_chars is not None
        else (HARD_MAX_OUTPUT_CHARS if args.all_results else DEFAULT_MAX_OUTPUT_CHARS)
    )
    _bound_record_text(records, max_output)
    _render_records(records, args.format)
    return 2 if errors else (0 if any_match else 1)


def _inspect(args: argparse.Namespace) -> int:
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
            print(f"SHA-256: {record['sha256']}")
            print(f"Type: {record['input_type']}")
            print(f"Pages: {record['pages']}")
            print(f"Headings: {sum(record['headings_by_level'].values())}")
            print(f"Source bindings: {record['source_map']['bound_xpaths']}")
            print("Elements: " + ", ".join(f"{key}={value}" for key, value in record["source_counts"].items()))
    return 0


def _outline(args: argparse.Namespace) -> int:
    loaded = _load_command_input(args.input, validate=args.validate)
    records: list[dict[str, Any]] = []
    for item, _ in loaded.document.iterate_items(with_groups=True):
        if not isinstance(item, (TitleItem, SectionHeaderItem)):
            continue
        heading_depth = len(_heading_ancestors(item, loaded)) + 1
        if args.depth is not None and heading_depth > args.depth:
            continue
        target = DocLangSourceTarget(kind="item", item_ref=item.self_ref)
        xpaths = loaded.source_map.xpaths_by_target.get(target, [])
        if xpaths:
            records.append({"xpath": xpaths[0], "xpaths": xpaths, "depth": heading_depth, "text": item.text})
    if args.format == "json":
        print(
            json.dumps(
                {"document": args.input, "sha256": loaded.sha256, "headings": records}, ensure_ascii=False, indent=2
            )
        )
    else:
        for record in records:
            print(f"{'  ' * (record['depth'] - 1)}{record['text']}\t{record['xpath']}")
    return 0


def _select(args: argparse.Namespace) -> int:
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
            record = {"xpath": xpath, "xml": bounded, "truncated": truncated}
            if args.semantic and (target := loaded.source_map.targets_by_xpath.get(xpath)) is not None:
                record["semantic"] = asdict(target)
            records.append(record)
        else:
            records.append({"value": str(value)})

    if args.format == "xml":
        for record in records:
            print(record.get("xml", record.get("value", "")))
    elif args.format == "text":
        for record in records:
            if "xpath" in record:
                selected_xpath = record["xpath"]
                if not isinstance(selected_xpath, str):
                    raise DlgrepError("invalid internal XPath result")
                element = loaded.raw_elements[selected_xpath]
                print("".join(element.itertext()))
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


def _show(args: argparse.Namespace) -> int:
    if args.max_chars < 0:
        raise DlgrepError("--max-chars must be non-negative")
    loaded = _load_command_input(args.input, validate=args.validate)
    selected = loaded.evaluate_xpath(args.xpath)
    if not isinstance(selected, list) or not selected or not all(_is_element(value) for value in selected):
        raise DlgrepError("show XPath must select one or more elements")
    before_count, after_count = _context_counts(args.context_events)
    records: list[dict[str, Any]] = []
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
                    "sha256": loaded.sha256,
                    "xpath": xpath,
                    "logical_type": "raw",
                    "text": text,
                    "truncated": truncated,
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
                    "sha256": loaded.sha256,
                    "xpath": xpath,
                    "logical_type": "raw",
                    "text": text,
                    "truncated": truncated,
                    "semantic": False,
                }
            )
            continue
        text = unit.text
        logical_type = unit.logical_type
        if args.section:
            item = loaded.target_item(target)
            if not isinstance(item, (TitleItem, SectionHeaderItem)):
                raise DlgrepError("--section requires a heading XPath")
            text = "\n\n".join(
                candidate.text
                for candidate in loaded.context_units
                if loaded.is_descendant(candidate.item_ref, item.self_ref)
            )
            logical_type = "section"
        before, after = loaded.context_for(unit, before_count, after_count, args.context_scope, loaded.context_units)
        record = _result_record(
            loaded,
            unit,
            text,
            [],
            before,
            after,
            max_chars=args.max_chars,
            ancestors=not args.no_ancestors,
        )
        record["logical_type"] = logical_type
        records.append(record)
    _render_records(records, args.format)
    return 0


def _patterns_and_inputs(args: argparse.Namespace) -> tuple[list[str], list[str], bytes | None]:
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
    args: argparse.Namespace,
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
            if not unit.container and unit.logical_type not in {"page_header", "page_footer"}
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
    if view == "all" and unit.metadata:
        return f"{unit.text}\n{unit.metadata}".strip()
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


def _validate_search_options(args: argparse.Namespace, before: int, after: int) -> None:
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
    matches: list[dict[str, Any]],
    before: list[Unit],
    after: list[Unit],
    *,
    max_chars: int,
    ancestors: bool,
) -> dict[str, Any]:
    bounded, truncated = _truncate(text, max_chars)
    context: dict[str, Any] = {
        "before": [_context_record(value, max_chars) for value in before],
        "after": [_context_record(value, max_chars) for value in after],
    }
    if ancestors:
        context["headings"] = loaded.heading_chain(unit)
    context.update(_table_context(loaded, unit))
    return {
        "document": loaded.name,
        "sha256": loaded.sha256,
        "xpath": unit.xpath,
        "xpaths": list(unit.xpaths),
        "cardinality": len(unit.xpaths),
        "logical_type": unit.logical_type,
        "page": unit.page,
        "pages": list(unit.pages),
        "layer": unit.layer,
        "text": bounded,
        "matches": matches,
        "context": context,
        "truncated": truncated,
    }


def _context_record(unit: Unit, max_chars: int) -> dict[str, Any]:
    text, truncated = _truncate(unit.text, max_chars)
    return {
        "xpath": unit.xpath,
        "logical_type": unit.logical_type,
        "page": unit.page,
        "pages": list(unit.pages),
        "text": text,
        "truncated": truncated,
    }


def _table_context(loaded: LoadedDocument, unit: Unit) -> dict[str, Any]:
    if unit.target.kind != "table_cell" or unit.item_ref is None or unit.row is None or unit.col is None:
        return {}
    item = loaded.target_item(unit.target)
    if not isinstance(item, TableItem):
        return {"row": unit.row, "column": unit.col}
    grid = item.data.grid
    row_headers = [cell.text for cell in grid[unit.row] if cell.row_header and cell.text]
    column_headers = [row[unit.col].text for row in grid if row[unit.col].column_header and row[unit.col].text]
    return {
        "table_caption": item.caption_text(loaded.document),
        "row": unit.row,
        "column": unit.col,
        "row_headers": list(dict.fromkeys(row_headers)),
        "column_headers": list(dict.fromkeys(column_headers)),
    }


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
    marker = "… [truncated]"
    return text[: max(0, limit - len(marker))] + marker[:limit], True


def _bound_record_text(records: list[dict[str, Any]], limit: int) -> None:
    remaining = limit
    for record in records:
        values = [*record["context"]["before"], record, *record["context"]["after"]]
        for value in values:
            text = value["text"]
            if len(text) <= remaining:
                remaining -= len(text)
                continue
            value["text"], _ = _truncate(text, remaining)
            value["truncated"] = True
            remaining = 0


def _render_records(records: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(records, ensure_ascii=False, indent=2))
    elif output_format == "jsonl":
        for record in records:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    else:
        for index, record in enumerate(records):
            if index:
                print("--")
            print(record["document"])
            print(f"XPath: {record['xpath']}")
            print(f"Type: {record['logical_type']}")
            if record.get("page") is not None:
                print(f"Page: {record['page']}")
            context_values = record.get("context", {"before": [], "after": []})
            headings = context_values.get("headings", [])
            if headings:
                print("Section: " + " > ".join(headings))
            for context in context_values["before"]:
                print(f"- {context['xpath']} {context['text']}")
            print(_highlight(record["text"], record.get("matches", [])))
            for context in context_values["after"]:
                print(f"- {context['xpath']} {context['text']}")


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
