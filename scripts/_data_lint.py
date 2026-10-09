"""Private data lint for DoclingDocument JSON and DocLang (DCLX) files. Not part of the docling-core API.

Usage (the kind is chosen by the extension)::

    uv run python scripts/_data_lint.py FILE.json
    uv run python scripts/_data_lint.py FILE.dclx              # also FILE.dclg (plain DocLang markup) and FILE.xml
    uv run python scripts/_data_lint.py FILE.dclx --pair   # also FILE.json from the same folder, and both conversions

Options: ``--list-checks`` lists the checks, ``--only`` / ``--skip`` select checks by slug or dotted prefix
(``--only dclx.roundtrip``, ``--skip json.rules``), ``-q`` prints only failing checks (with ``--report files``), ``--report checks`` prints counts per check over all files
(``checks-with-files`` adds the files behind the fail and warn counts), ``--output json`` prints machine-readable
results; several files are allowed.
The exit code is 1 when a check fails. The ``check_*`` functions can also be imported.

Status
------
``ok``    the check passed
``fail``  a defect (non-zero exit code)
``warn``  a lint / suspicious pattern that may be legitimate (never affects the exit code)
``skip``  the check could not run (e.g. Schematron is not installed)
"""

from __future__ import annotations

import collections
import difflib
import re
import tempfile
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer
from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import ContentLayer, GroupItem, ListItem

_MAX_DETAILS = 12


@dataclass
class Result:
    check: str
    status: str  # ok | fail | warn | skip
    message: str = ""
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# slug -> (applies to, description). The slugs are what ``--only`` / ``--skip`` match (exactly or as a dotted prefix).
CHECKS: dict[str, tuple[str, str]] = {
    "json.load": ("json", "the JSON loads without warnings"),
    "json.rules": ("json", "DoclingDocument._validate_rules reports nothing"),
    "json.serialize": ("json", "the document serializes to DocLang"),
    "json.empty-list-wrapper": ("json", "empty list items that only wrap content (heuristic, warns)"),
    "dclx.xsd": ("dclx", "XSD validity of document.xml"),
    "dclx.schematron": ("dclx", "Schematron validity of document.xml (skipped without the backend)"),
    "dclx.deserialize": ("dclx", "the DocLang deserializes without warnings"),
    "dclx.rules": ("dclx", "model validation and _validate_rules on the deserialized document"),
    "dclx.roundtrip": ("dclx", "DCLX -> document -> DCLX gives the same DocLang"),
    "pair.json-to-dclx": ("pair", "the JSON serializes to the same DocLang as the DCLX"),
    "pair.dclx-to-json": (
        "pair",
        "diagnostic (warns): what differs between the JSON and the document deserialized from the DCLX; some differences are expected, as DocLang normalizes some aspects",
    ),
}


class Selection:
    """Which checks run: ``only`` (all when empty) minus ``skip``; a pattern matches a slug exactly or as a dotted prefix.

    A ``skip`` that refines an ``only`` is fine (``--only dclx --skip dclx.schematron``); an ``only`` pattern whose checks
    are all skipped is an error.
    """

    def __init__(self, only: list[str] | None = None, skip: list[str] | None = None) -> None:
        self.only, self.skip = list(only or []), list(skip or [])
        for pattern in self.only + self.skip:
            if not any(_matches(pattern, slug) for slug in CHECKS):
                raise SystemExit(f"unknown check: {pattern} (see --list-checks)")
        for pattern in self.only:  # contradictory instructions: everything --only asks for is excluded by --skip
            if not any(self.on(slug) for slug in CHECKS if _matches(pattern, slug)):
                raise SystemExit(f"--only {pattern} selects nothing: all of its checks are excluded by --skip")

    def on(self, slug: str) -> bool:
        if self.only and not any(_matches(p, slug) for p in self.only):
            return False
        return not any(_matches(p, slug) for p in self.skip)

    def any_on(self, *slugs: str) -> bool:
        return any(self.on(slug) for slug in slugs)


def _matches(pattern: str, slug: str) -> bool:
    return slug == pattern or slug.startswith(pattern + ".")


_ALL = Selection()


_CONTENT_RE = re.compile(r"(<content>.*?</content>)", re.DOTALL)


def _lines(text: str) -> list[str]:
    """DocLang as comparable lines: indentation outside ``<content>`` is normalized, a ``<content>`` element is kept
    verbatim as one entry, because it preserves whitespace (leading/trailing spaces and line breaks are significant)."""
    out: list[str] = []
    for part in _CONTENT_RE.split(text.strip()):
        if part.startswith("<content>"):
            out.append(part)
        else:
            out.extend(line.strip() for line in part.splitlines() if line.strip())
    return out


def _read_xml(path: Path) -> str:
    """Return the DocLang XML of a ``.dclx`` archive or of a plain markup file (``.dclg``, ``.xml``)."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return zf.read("document.xml").decode()
    return path.read_text()


def resolution_of(xml: str) -> tuple[int, int]:
    """The ``<default_resolution>`` grid of a DocLang document (512x512 when absent)."""
    m = re.search(r'default_resolution width="(\d+)" height="(\d+)"', xml)
    return (int(m[1]), int(m[2])) if m else (512, 512)


def serialize(doc: DoclingDocument, resolution: tuple[int, int]) -> str:
    return (
        DocLangDocSerializer(doc=doc, params=DocLangParams(xsize=resolution[0], ysize=resolution[1])).serialize().text
    )


def _warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return sorted({str(w.message)[:200] for w in caught})


def _load_json(path: Path) -> tuple[DoclingDocument | None, list[str], Result | None]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            doc = DoclingDocument.load_from_json(path)
        except Exception as e:
            return None, [], Result("json.load", "fail", f"{type(e).__name__}: {str(e)[:300]}")
    return doc, _warning_messages(caught), None


def _diff(a: str, b: str) -> list[str]:
    d = [
        line
        for line in difflib.unified_diff(_lines(a), _lines(b), lineterm="", n=0)
        if not line.startswith(("---", "+++", "@@"))
    ]
    return d


# --------------------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------------------
def empty_list_wrappers(doc: DoclingDocument) -> list[str]:
    """List items without text, marker or bbox that only wrap other content (suspected migration noise).

    A heuristic: a wrapper can be legitimate when no list item precedes it in its list.
    """
    out = []
    for item in doc.texts:
        if not isinstance(item, ListItem) or item.text or item.marker or item.prov or not item.children:
            continue
        parent = item.parent.resolve(doc) if item.parent else None
        refs = list(parent.children) if parent is not None else []
        idx = next((i for i, r in enumerate(refs) if r.cref == item.self_ref), -1)
        prev_is_item = idx > 0 and isinstance(refs[idx - 1].resolve(doc), ListItem)
        kinds = sorted({c.resolve(doc).label.value for c in item.children})
        out.append(f"{item.self_ref} wraps {'+'.join(kinds)}; preceded by a list item: {prev_is_item}")
    return out


def check_json(path: str | Path, *, resolution: tuple[int, int] | None = None, sel: Selection = _ALL) -> list[Result]:
    """Individual checks of a DoclingDocument JSON."""
    path = Path(path)
    if not sel.any_on(*(slug for slug, (kind, _) in CHECKS.items() if kind == "json")):
        return []
    results: list[Result] = []
    doc, load_warnings, err = _load_json(path)
    if err or doc is None:
        return [err or Result("json.load", "fail", "could not load")]
    if sel.on("json.load"):
        results.append(
            Result(
                "json.load", "fail" if load_warnings else "ok", "load warnings" if load_warnings else "", load_warnings
            )
        )
    if sel.on("json.rules"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            doc._validate_rules(raise_on_error=False)
        rules = _warning_messages(caught)
        results.append(Result("json.rules", "fail" if rules else "ok", "_validate_rules" if rules else "", rules))
    if sel.on("json.serialize"):
        try:
            serialize(doc, resolution or (512, 512))
            results.append(Result("json.serialize", "ok"))
        except Exception as e:
            results.append(Result("json.serialize", "fail", f"{type(e).__name__}: {str(e)[:300]}"))
    if sel.on("json.empty-list-wrapper"):
        wrappers = empty_list_wrappers(doc)
        results.append(
            Result(
                "json.empty-list-wrapper",
                "warn" if wrappers else "ok",
                f"{len(wrappers)} empty list item(s) that only wrap content (heuristic: legitimate if no list item precedes)"
                if wrappers
                else "",
                wrappers[:_MAX_DETAILS],
            )
        )
    return results


# --------------------------------------------------------------------------------------------------
# DCLX
# --------------------------------------------------------------------------------------------------
def _validate_xml(xml: str, sel: Selection) -> list[Result]:
    from doclang.validation import ValidationError, validate

    checks = [
        (slug, kwargs)
        for slug, kwargs in (("dclx.xsd", {"xsd_only": True}), ("dclx.schematron", {"schematron_only": True}))
        if sel.on(slug)
    ]
    if not checks:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "document.xml"
        p.write_text(xml)
        out = []
        for name, kwargs in checks:
            try:
                validate(p, allow_empty_namespace=True, **kwargs)
                out.append(Result(name, "ok"))
            except ValidationError as e:
                errs = [str(x)[:200] for x in list(e.xsd_errors) + list(e.schematron_errors)]
                out.append(Result(name, "fail", f"{len(errs)} error(s)", errs[:_MAX_DETAILS]))
            except Exception as e:
                out.append(Result(name, "skip", f"{type(e).__name__}: {str(e)[:200]}"))
        return out


def check_dclx(path: str | Path, *, sel: Selection = _ALL) -> list[Result]:
    """Individual checks of a DocLang document: schema validity, deserializability, round trip."""
    path = Path(path)
    xml = _read_xml(path)
    results = _validate_xml(xml, sel)
    res = resolution_of(xml)
    if not sel.any_on("dclx.deserialize", "dclx.rules", "dclx.roundtrip"):
        return results
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            doc = DocLangDocDeserializer().deserialize_str(xml)
        except Exception as e:
            results.append(Result("dclx.deserialize", "fail", f"{type(e).__name__}: {str(e)[:300]}"))
            return results
    w = _warning_messages(caught)
    if sel.on("dclx.deserialize"):
        results.append(Result("dclx.deserialize", "warn" if w else "ok", "warnings" if w else "", w))
    if sel.on("dclx.rules"):
        problems: list[str] = []
        try:  # the deserializer builds the document through the API, which does not run the model validators
            DoclingDocument.model_validate(doc.model_dump(mode="json"))
        except Exception as e:
            problems.append(f"model validation: {type(e).__name__}: {str(e)[:300]}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            doc._validate_rules(raise_on_error=False)
        problems += _warning_messages(caught)
        results.append(
            Result(
                "dclx.rules",
                "fail" if problems else "ok",
                "model validation and _validate_rules on the deserialized document" if problems else "",
                problems,
            )
        )
    if sel.on("dclx.roundtrip"):
        try:
            again = serialize(doc, res)
        except Exception as e:
            results.append(
                Result("dclx.roundtrip", "fail", f"re-serialization failed: {type(e).__name__}: {str(e)[:300]}")
            )
            return results
        d = _diff(xml, again)
        results.append(
            Result(
                "dclx.roundtrip",
                "fail" if d else "ok",
                f"re-serialization differs in {len(d)} line(s)" if d else "",
                d[:_MAX_DETAILS],
            )
        )
    return results


# --------------------------------------------------------------------------------------------------
# JSON <-> DCLX conversion comparison
# --------------------------------------------------------------------------------------------------
def _kind(item: Any) -> str:
    label = getattr(getattr(item, "label", None), "value", str(getattr(item, "label", "")))
    return ("group:" if isinstance(item, GroupItem) else "") + label


def _signature(doc: DoclingDocument, res: tuple[int, int]) -> dict[str, collections.Counter]:
    """Order-insensitive fingerprint of a document: kinds, (kind, parent kind), texts, layers, threads, grid bboxes."""
    sig: dict[str, collections.Counter] = {
        k: collections.Counter() for k in ("kinds", "parents", "texts", "layers", "threads", "bboxes")
    }
    for item, _ in doc.iterate_items(
        with_groups=True, traverse_pictures=True, included_content_layers=set(ContentLayer)
    ):
        kind = _kind(item)
        sig["kinds"][kind] += 1
        try:
            parent = _kind(item.parent.resolve(doc)) if item.parent else "-"
        except Exception:
            parent = "?"
        sig["parents"][f"{kind} in {parent}"] += 1
        sig["layers"][f"{kind} [{item.content_layer.value}]"] += 1
        text = (str(getattr(item, "marker", "") or "") + "|" + str(getattr(item, "text", "") or "")).split()
        if text and text != ["|"]:
            sig["texts"][f"{kind}: {' '.join(text)[:80]}"] += 1
        provs = getattr(item, "prov", None) or []
        if len(provs) > 1:
            sig["threads"][kind] += 1
        for prov in provs:
            page = doc.pages.get(prov.page_no)
            if page is None or not page.size.width:
                continue
            bbox = prov.bbox.to_top_left_origin(page.size.height)
            sig["bboxes"][
                tuple(
                    round(v)
                    for v in (
                        bbox.l / page.size.width * res[0],
                        bbox.t / page.size.height * res[1],
                        bbox.r / page.size.width * res[0],
                        bbox.b / page.size.height * res[1],
                    )
                )
            ] += 1
    return sig


def _bbox_diff(a: collections.Counter, b: collections.Counter) -> tuple[collections.Counter, collections.Counter]:
    """Multiset difference of grid bboxes with a +-1 tolerance (rounding)."""
    left, right = collections.Counter(a), collections.Counter(b)
    for box in list(left):
        for cand in list(right):
            if left[box] and right[cand] and all(abs(x - y) <= 1 for x, y in zip(box, cand)):
                n = min(left[box], right[cand])
                left[box] -= n
                right[cand] -= n
    return +left, +right


def conversion_diff(doc_json: DoclingDocument, doc_dclx: DoclingDocument, res: tuple[int, int]) -> list[str]:
    """What differs between the JSON document and the document deserialized from its DCLX (empty when equivalent)."""
    a, b = _signature(doc_json, res), _signature(doc_dclx, res)
    out: list[str] = []
    for key in ("kinds", "parents", "layers", "threads", "texts", "bboxes"):
        only_json, only_dclx = _bbox_diff(a[key], b[key]) if key == "bboxes" else (a[key] - b[key], b[key] - a[key])
        for side, counter in (("only in JSON", only_json), ("only after DCLX -> JSON", only_dclx)):
            out += [f"{key}: {side}: {k} x{n}" for k, n in sorted(counter.items(), key=str)]
    return out


# --------------------------------------------------------------------------------------------------
# Pair
# --------------------------------------------------------------------------------------------------
def check_pair(json_path: str | Path, dclx_path: str | Path, *, sel: Selection = _ALL) -> list[Result]:
    """Each side individually, plus JSON -> DCLX (same DocLang, a failure if not) and a DCLX -> JSON diagnostic (warns)."""
    json_path, dclx_path = Path(json_path), Path(dclx_path)
    xml = _read_xml(dclx_path)
    res = resolution_of(xml)
    results = [
        Result(f"json/{r.check.split('.', 1)[1]}", r.status, r.message, r.details)
        for r in check_json(json_path, resolution=res, sel=sel)
    ]
    results += [
        Result(f"dclx/{r.check.split('.', 1)[1]}", r.status, r.message, r.details)
        for r in check_dclx(dclx_path, sel=sel)
    ]
    if not sel.any_on("pair.json-to-dclx", "pair.dclx-to-json"):
        return results
    doc, _, err = _load_json(json_path)
    if err or doc is None:
        results += [
            Result(slug, "fail", "JSON does not load")
            for slug in ("pair.json-to-dclx", "pair.dclx-to-json")
            if sel.on(slug)
        ]
        return results
    if sel.on("pair.json-to-dclx"):
        try:
            d = _diff(xml, serialize(doc, res))
            results.append(
                Result(
                    "pair.json-to-dclx",
                    "fail" if d else "ok",
                    f"JSON serializes differently from the DCLX in {len(d)} line(s)" if d else "",
                    d[:_MAX_DETAILS],
                )
            )
        except Exception as e:
            results.append(Result("pair.json-to-dclx", "fail", f"{type(e).__name__}: {str(e)[:300]}"))
    if sel.on("pair.dclx-to-json"):
        try:
            back = DocLangDocDeserializer().deserialize_str(xml)
            diff = conversion_diff(doc, back, res)
            results.append(
                Result(
                    "pair.dclx-to-json",
                    "warn" if diff else "ok",
                    f"the document deserialized from the DCLX differs from the JSON in {len(diff)} way(s) "
                    "(may be legitimate, as DocLang normalizes some aspects)"
                    if diff
                    else "",
                    diff[:_MAX_DETAILS],
                )
            )
        except Exception as e:
            results.append(Result("pair.dclx-to-json", "fail", f"{type(e).__name__}: {str(e)[:300]}"))
    return results


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
_DOCLANG_SUFFIXES = (".dclx", ".dclg", ".xml")


def _base(path: Path) -> Path:
    """The path without its DocLang extension (``X.dclg.xml`` -> ``X``)."""
    name = path.name
    for ext in (".dclg.xml", *_DOCLANG_SUFFIXES):
        if name.endswith(ext):
            return path.with_name(name[: -len(ext)])
    return path.with_suffix("")


def _counterpart(path: Path) -> Path:
    """The same-stem file of the other kind in the same folder (``X.json`` <-> ``X.dclx`` / ``X.dclg`` / ``X.xml``)."""
    base = _base(path)
    candidates = [base.with_name(base.name + ext) for ext in (".dclx", ".dclg", ".dclg.xml", ".xml")]
    if path.suffix == ".json":
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise SystemExit(f"--pair: counterpart not found: {base}.dclx / .dclg / .dclg.xml / .xml")
        return found
    other = base.with_name(base.name + ".json")
    if not other.is_file():
        raise SystemExit(f"--pair: counterpart not found: {other}")
    return other


def _canonical(check: str) -> str:
    """The check slug (``json/load`` in pair mode is ``json.load``)."""
    return check.replace("/", ".", 1) if check.startswith(("json/", "dclx/")) else check


def _per_check(
    jobs: list[tuple[str, list[Result]]],
) -> tuple[dict[str, collections.Counter], dict[str, list[tuple[str, str]]]]:
    """Per check the number of files by status, and per check the files that fail or warn."""
    counts: dict[str, collections.Counter] = {}
    flagged: dict[str, list[tuple[str, str]]] = {}
    for label, results in jobs:
        for r in results:
            slug = _canonical(r.check)
            counts.setdefault(slug, collections.Counter())[r.status] += 1
            if r.status in ("fail", "warn"):
                flagged.setdefault(slug, []).append((r.status, label))
    order = {slug: i for i, slug in enumerate(CHECKS)}
    counts = dict(sorted(counts.items(), key=lambda kv: order.get(kv[0], len(order))))
    flagged = dict(sorted(flagged.items(), key=lambda kv: order.get(kv[0], len(order))))
    return counts, flagged


def _print_per_check(jobs: list[tuple[str, list[Result]]], list_files: bool) -> None:
    counts, flagged = _per_check(jobs)
    print(f"PER CHECK ({len(jobs)} file(s))")
    if not counts:
        print("no checks ran")
        return
    width = max(len(slug) for slug in counts)
    print(f"{'check':{width}}  {'ok':>5} {'warn':>5} {'fail':>5} {'skip':>5}")
    for slug, c in counts.items():
        print(f"{slug:{width}}  {c['ok']:>5} {c['warn']:>5} {c['fail']:>5} {c['skip']:>5}")
    if list_files:
        for slug, items in flagged.items():
            for status in ("fail", "warn"):
                labels = [label for st, label in items if st == status]
                if labels:
                    print(f"\n{status.upper()} {slug} ({len(labels)}):")
                    for label in labels:
                        print(f"  {label}")


def _list_checks() -> None:
    titles = {
        "json": "json   (run on .json files)",
        "dclx": "dclx   (run on .dclx / .dclg / .xml files)",
        "pair": "pair   (with --pair, on top of the json and dclx checks)",
    }
    for kind, title in titles.items():
        print(title)
        for slug, (k, description) in CHECKS.items():
            if k == kind:
                print(f"  {slug:32} {description}")


def _split(values: list[str] | None) -> list[str]:
    return [part for value in values or [] for part in value.split(",") if part]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Validate DoclingDocument JSON (.json) and DocLang (.dclx / .dclg / .xml) files, by extension."
    )
    ap.add_argument("files", nargs="*", help="FILE.json, FILE.dclx, FILE.dclg or FILE.xml")
    ap.add_argument(
        "-p",
        "--pair",
        action="store_true",
        help="also check the same-stem counterpart (FILE.json <-> FILE.dclx / .dclg / .xml) in the same folder, and the conversions both ways",
    )
    ap.add_argument(
        "--only",
        action="append",
        metavar="CHECK",
        help="run only these checks (slug or prefix, repeatable or comma-separated)",
    )
    ap.add_argument(
        "--skip",
        action="append",
        metavar="CHECK",
        help="skip these checks (slug or prefix, repeatable or comma-separated)",
    )
    ap.add_argument("--list-checks", action="store_true", help="list the available checks and exit")
    ap.add_argument(
        "--report",
        choices=["files", "checks", "checks-with-files"],
        default="files",
        help="files: the results of each file (default); checks: counts (ok / warn / fail / skip) per check over "
        "all files; checks-with-files: the same, plus the files behind each warn / fail count",
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="only print failing checks")
    ap.add_argument("--output", choices=["text", "json"], default="text")
    args = ap.parse_args(argv)

    if args.list_checks:
        _list_checks()
        return 0
    if not args.files:
        ap.error("no files given (or use --list-checks)")
    sel = Selection(_split(args.only), _split(args.skip))

    jobs: list[tuple[str, list[Result]]] = []
    seen: set[tuple[Path, Path]] = set()
    for name in args.files:
        path = Path(name)
        if path.suffix not in (".json", *_DOCLANG_SUFFIXES):
            raise SystemExit(f"unsupported extension (expected .json, .dclx, .dclg or .xml): {path}")
        if args.pair:
            other = _counterpart(path)
            json_path, dclx_path = (path, other) if path.suffix == ".json" else (other, path)
            if (json_path, dclx_path) not in seen:
                seen.add((json_path, dclx_path))
                jobs.append((f"{json_path.name} + {dclx_path.name}", check_pair(json_path, dclx_path, sel=sel)))
        elif path.suffix == ".json":
            jobs.append((path.name, check_json(path, sel=sel)))
        else:
            jobs.append((path.name, check_dclx(path, sel=sel)))

    failed = any(r.status == "fail" for _, results in jobs for r in results)
    per_check = args.report != "files"
    if args.output == "json":
        if per_check:
            counts, flagged = _per_check(jobs)
            out: dict[str, Any] = {"files": len(jobs), "checks": {slug: dict(c) for slug, c in counts.items()}}
            if args.report == "checks-with-files":
                out["flagged"] = {
                    slug: [{"status": st, "file": label} for st, label in items] for slug, items in flagged.items()
                }
        else:
            out = {label: [r.as_dict() for r in results] for label, results in jobs}
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        if per_check:
            _print_per_check(jobs, args.report == "checks-with-files")
        else:
            for label, results in jobs:
                if len(jobs) > 1 or args.pair:
                    print(f"== {label}")
                if not results:
                    print(f"no selected check applies to {label}")
                for r in results:
                    if args.quiet and r.status != "fail":
                        continue
                    print(f"{r.status.upper():5} {r.check}" + (f": {r.message}" if r.message else ""))
                    for line in r.details:
                        print(f"        {line}")
        warned = sum(r.status == "warn" for _, results in jobs for r in results)
        print("RESULT: " + ("FAILED" if failed else f"WARN ({warned} warning(s))" if warned else "OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
