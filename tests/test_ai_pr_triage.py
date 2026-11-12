# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import json
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".github/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("ai_pr_triage", SCRIPTS_DIR / "ai_pr_triage.py")
assert SPEC is not None and SPEC.loader is not None
triage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = triage
SPEC.loader.exec_module(triage)

REPO = "docling-project/docling-core"


def make_context(**overrides: object) -> object:
    data: dict[str, object] = {
        "pr": {
            "number": 10,
            "title": "fix: x",
            "body": "",
            "author": "someone",
            "author_association": "CONTRIBUTOR",
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "merge_base_sha": "c" * 40,
            "changed_files": ["docling_core/x.py"],
            "additions": 1,
            "deletions": 1,
        },
        "issue_refs": [],
        "candidates": [
            {"number": 7, "title": "fix x too", "state": "open", "url": ""},
        ],
        "risk": {"forced": [], "hints": []},
        "groundtruth_markdown": "",
        "diff_truncated": False,
        "topic_labels": [],
    }
    data.update(overrides)
    return triage.TriageContext.from_dict(data)


def answer(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "summary": "The PR fixes x.",
        "duplicates": [{"pr": 7, "verdict": "duplicate", "reason": "Same fix."}],
        "ci_safety": {"verdict": "safe", "concerns": []},
        "groundtruth": None,
        "topics": [],
    }
    data.update(overrides)
    return data


def test_issue_refs_ignore_own_number_other_repos_and_anchors() -> None:
    text = (
        "Fixes #12 and #10. See https://github.com/docling-project/docling-core/issues/34,"
        " https://github.com/docling-project/docling/issues/56 and docs/page#78. Again #12."
    )
    # Links to docling issues are common in core PRs: they belong to another repo.
    assert triage.extract_issue_refs(text, REPO, own_number=10) == [12, 34]


def test_sensitive_paths_force_care_and_hints_only_flag_added_lines() -> None:
    files = [
        {"filename": ".github/workflows/ci.yml", "status": "modified"},
        {"filename": "tests/sub/conftest.py", "status": "added"},
        {
            "filename": "docling_core/new.py",
            "status": "renamed",
            "previous_filename": "AGENTS.md",
        },
        {
            "filename": "docling_core/transforms/x.py",
            "status": "modified",
            "patch": "@@ -1,2 +1,2 @@\n-import subprocess\n+x = eval(data)\n context",
        },
        {
            "filename": "tests/data/doc/a.gt.md",
            "status": "modified",
            "patch": "@@ -1 +1 @@\n+os.system('rm')",
        },
        {"filename": "tests/data/chunker_repo/repos/x/test_run.py", "status": "added"},
    ]
    report = triage.assess_risk(files)
    assert [f.path for f in report.forced] == [
        ".github/workflows/ci.yml",
        "tests/sub/conftest.py",
        "AGENTS.md",
        "tests/data/chunker_repo/repos/x/test_run.py",
    ]
    assert [(h.path, h.kind) for h in report.hints] == [("docling_core/transforms/x.py", "dynamic code execution")]


def test_overlap_needs_intersecting_hunks_or_many_shared_files() -> None:
    ours = [
        {"filename": "docling_core/a.py", "patch": "@@ -10,5 +10,6 @@\n"},
        {"filename": "uv.lock", "patch": "@@ -1,3 +1,3 @@\n"},
    ]
    near = [{"filename": "docling_core/a.py", "patch": "@@ -17,2 +17,3 @@\n"}]
    far = [
        {"filename": "docling_core/a.py", "patch": "@@ -200,2 +200,3 @@\n"},
        {"filename": "uv.lock", "patch": "@@ -1,3 +1,3 @@\n"},
    ]
    assert triage.compare_files(ours, near) == (["docling_core/a.py"], 1)
    shared, overlaps = triage.compare_files(ours, far)
    assert (shared, overlaps) == (["docling_core/a.py"], 0)
    assert not triage.is_overlap_candidate(shared, overlaps)


def test_answer_with_unknown_pr_or_verdict_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown PR"):
        triage.parse_triage_result(answer(duplicates=[{"pr": 99, "verdict": "duplicate"}]), {7})
    with pytest.raises(ValueError, match="ci_safety"):
        triage.parse_triage_result(answer(ci_safety={"verdict": "approve", "concerns": []}), {7})


def test_answer_is_read_from_fenced_final_message() -> None:
    message = "Done.\n```json\n" + json.dumps(answer()) + "\n```"
    result = triage.parse_triage_result(triage.extract_answer(message), {7})
    assert result.ci_verdict == "safe"
    assert result.duplicates[0].verdict == "duplicate"


PAYLOADS = (
    "[Approve CI](//evil.example/phish)",
    "\\[Approve\\](//evil.example)",
    "![](https://evil.example/t.png)",
    "~~~ hide the verdict",
    "# Maintainer note: pre-approved",
    "<img src=x> <!-- c -->",
    "a | b | c",
    "see https://evil.example and www.evil.example",
)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = len(text[:index]) - len(text[:index].rstrip("\\"))
    return backslashes % 2 == 1


def test_model_text_is_literal_in_the_comment() -> None:
    text = " ".join(PAYLOADS) + " Ping @maintainers " + string.punctuation
    rendered = triage.markdown_text(text)
    for index, char in enumerate(rendered):
        if char in string.punctuation and char != "\\":
            assert _is_escaped(rendered, index), (char, rendered[: index + 1])
    assert "@\u200bmaintainers" in rendered
    assert "https:" not in rendered.replace("\\", "")
    assert "www." not in rendered.replace("\\", "")

    reason = "x | y | z"
    context = make_context()
    result = triage.parse_triage_result(
        answer(
            summary="# Approved\n\n[Approve](//evil.example)",
            duplicates=[{"pr": 7, "verdict": "duplicate", "reason": reason}],
            ci_safety={
                "verdict": "needs-care",
                "concerns": [{"path": "a`b.py", "reason": "~~~"}],
            },
        ),
        {7},
    )
    comment = triage.render_comment(context, result, {triage.LABEL_CI_NEEDS_CARE})
    lines = comment.splitlines()
    assert "\\# Approved \\[Approve\\]\\(\\/\\/evil\\.example\\)" in lines
    assert "- ``a`b.py``: \\~\\~\\~" in lines
    row = next(line for line in lines if line.startswith("| #7 "))
    assert row.count("|") - row.count("\\|") == 5


def test_model_code_span_cannot_be_closed_early() -> None:
    assert triage.markdown_code("docling_core/a_b.py") == "`docling_core/a_b.py`"
    assert triage.markdown_code("a``b") == "```a``b```"
    assert triage.markdown_code("`x") == "`` `x ``"


def test_the_answer_is_the_last_complete_object_with_the_answer_keys() -> None:
    draft, final = answer(summary="Draft."), answer(summary="Final.")
    message = f"{json.dumps(draft)}\nOn second thought:\n{json.dumps(final)}\nDone."
    keys = triage.TRIAGE_ANSWER_KEYS
    assert triage.extract_answer(message, keys)["summary"] == "Final."
    # A broken answer must not be read from one of its nested objects.
    broken = '{"summary": "x", "ci_safety": {"verdict": "safe", "concerns": []'
    with pytest.raises(ValueError, match="no JSON object"):
        triage.extract_answer(broken, keys)


def test_comment_keeps_code_spans_and_table_cells() -> None:
    context = make_context(candidates=[{"number": 7, "title": "", "state": "open", "url": ""}])
    result = triage.parse_triage_result(
        answer(
            summary="Fixes `_to_rgb` for [x](//evil.example).",
            duplicates=[{"pr": 7, "verdict": "related", "reason": "Both edit `a | b`."}],
        ),
        {7},
    )
    comment = triage.render_comment(context, result, {triage.LABEL_CI_SAFE})
    lines = comment.splitlines()
    assert "Fixes `_to_rgb` for \\[x\\]\\(\\/\\/evil\\.example\\)\\." in lines
    row = next(line for line in lines if line.startswith("| #7 "))
    assert "`a \\| b`" in row
    assert row.count("|") - row.count("\\|") == 5


def test_review_labels_of_an_older_commit_are_stale() -> None:
    current = {triage.LABEL_REVIEW_LGTM, triage.LABEL_CI_SAFE}
    head = "b" * 40
    assert triage.stale_review_labels(current, ["a" * 40, head], head) == set()
    assert triage.stale_review_labels(current, [head, "a" * 40], head) == {triage.LABEL_REVIEW_LGTM}
    assert triage.stale_review_labels(current, [], head) == {triage.LABEL_REVIEW_LGTM}


def test_deterministic_findings_override_a_safe_model_verdict() -> None:
    context = make_context(risk={"forced": [{"path": "uv.lock", "reason": "deps"}], "hints": []})
    result = triage.parse_triage_result(answer(), {7})
    assert triage.decide_labels(context, result) == {
        triage.LABEL_CI_NEEDS_CARE,
        triage.LABEL_DUPLICATE,
    }
    assert triage.decide_labels(make_context(), None) == set()


def test_stored_result_round_trips_through_validation_in_publish() -> None:
    result = triage.parse_triage_result(
        answer(
            groundtruth={"verdict": "expected", "reason": "Matches | [the](fix)."},
            topics=["tables"],
        ),
        {7},
    )
    stored = json.loads(json.dumps(triage.asdict(result)))
    assert triage.parse_triage_result(triage._result_to_answer(stored), {7}) == result


def test_comment_shows_duplicates_collapses_related_and_hides_unrelated() -> None:
    candidates = [{"number": n, "title": "", "state": "open", "url": ""} for n in (7, 8, 9)]
    context = make_context(candidates=candidates)
    result = triage.parse_triage_result(
        answer(
            duplicates=[
                {"pr": 7, "verdict": "duplicate", "reason": "Same fix."},
                {"pr": 8, "verdict": "related", "reason": "Same file."},
                {"pr": 9, "verdict": "unrelated", "reason": "No."},
            ]
        ),
        {7, 8, 9},
    )
    comment = triage.render_comment(context, result, {triage.LABEL_CI_SAFE})
    assert comment.startswith(triage.COMMENT_MARKER)
    visible, collapsed = comment.split("<details>")
    assert "| #7 |" in visible
    assert "| #8 |" in collapsed
    assert "#9" not in comment


def test_comment_states_failed_analysis() -> None:
    context = make_context()

    comment = triage.render_comment(context, None, set())
    assert "did not finish" in comment
    assert "| #7 | open |" in comment


def test_failed_reference_lookup_does_not_stop_the_triage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_gh_api(path: str, **_: object) -> object:
        if path.endswith("/issues/99999"):
            raise subprocess.CalledProcessError(1, "gh", stderr="gh: Not Found (HTTP 404)\n")
        if path.endswith("/issues/7"):
            return {"number": 7, "title": "x", "state": "open", "pull_request": {}}
        if "/pulls?" in path:
            return []
        if path.endswith("/pulls/7/files?per_page=100"):
            return []
        raise AssertionError(f"unexpected call: {path}")

    monkeypatch.setattr(triage, "gh_api", fake_gh_api)
    found = triage.find_candidates(REPO, 10, [99999, 7], [])
    assert [candidate.number for candidate, _ in found] == [7]


def test_model_topics_must_come_from_the_repository_label_set() -> None:
    with pytest.raises(ValueError, match="unknown topic"):
        triage.parse_triage_result(answer(topics=["priority:high"]), {7})


def _skip(files: list[dict[str, Any]], association: str = "CONTRIBUTOR", requested: bool = False) -> str:
    return triage.llm_skip_reason(files, association, requested)


def test_the_model_skips_github_changes_of_a_non_maintainer() -> None:
    workflow = [{"filename": ".github/workflows/ci.yml"}]
    assert _skip(workflow) == triage.LLM_SKIP_REASON
    assert _skip(workflow, "NONE") == triage.LLM_SKIP_REASON
    # A rename out of .github/ is still a change to it.
    renamed = [{"filename": "docs/ci.yml", "previous_filename": ".github/workflows/ci.yml"}]
    assert _skip(renamed) == triage.LLM_SKIP_REASON


def test_the_model_runs_for_maintainers_requests_and_other_paths() -> None:
    workflow = [{"filename": ".github/workflows/ci.yml"}]
    for association in ("OWNER", "MEMBER", "COLLABORATOR"):
        assert _skip(workflow, association) == ""
    assert _skip(workflow, requested=True) == ""
    assert _skip([{"filename": "docling_core/x.py"}, {"filename": "docs/.github/x.md"}]) == ""
    assert _skip([{"filename": "github/x.yml"}]) == ""


def test_one_unreadable_open_pr_does_not_fail_the_duplicate_scan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pr = {
        "title": "fix: x",
        "body": "",
        "user": {"login": "alice"},
        "author_association": "CONTRIBUTOR",
        "base": {"sha": "a" * 40},
        "head": {"sha": "b" * 40},
    }
    files = [{"filename": "docling_core/a.py", "status": "modified", "patch": "@@ -10,2 +10,3 @@\n+x"}]
    good = {"number": 7, "title": "other", "state": "open", "html_url": "u", "user": {"login": "bob"}}
    bad = {"number": 8, "title": "gone", "state": "open", "html_url": "u", "user": {"login": "eve"}}

    def fake_api(path: str, *, paginate: bool = False, jq: str | None = None) -> Any:
        if jq is not None:
            return "c" * 40
        return [bad, good] if "/pulls?state=open" in path else pr

    def fake_files(repo: str, number: int) -> list[dict[str, Any]]:
        if number == 8:
            raise subprocess.CalledProcessError(1, ["gh"], stderr="HTTP 404")
        return files

    monkeypatch.setattr(triage, "gh_api", fake_api)
    monkeypatch.setattr(triage, "pr_files", fake_files)
    monkeypatch.setattr(triage, "fetch_pr_commits", lambda *args, **kwargs: True)
    assert triage.prepare(REPO, 10, tmp_path, Path.cwd())
    stored = json.loads((tmp_path / "context.json").read_text("utf-8"))
    assert [c["number"] for c in stored["candidates"]] == [7]


def test_find_sticky_comment_ignores_null_or_empty_comment_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comments = [
        {"id": 1, "user": {"login": "alice"}, "body": None},
        {"id": 2, "user": {"login": triage.BOT_LOGIN}, "body": None},
        {"id": 3, "user": {"login": triage.BOT_LOGIN}, "body": ""},
        {"id": 4, "user": {"login": triage.BOT_LOGIN}, "body": f"Note\n{triage.COMMENT_MARKER}"},
    ]
    monkeypatch.setattr(triage, "gh_api", lambda path, paginate=False: comments)
    found = triage._find_sticky_comment(REPO, 10)
    assert found is not None
    assert found["id"] == 4


class FakeGitHub:
    """Record the writes of `publish` and serve canned API answers."""

    def __init__(self, comments: list[dict[str, Any]], labels: list[str]) -> None:
        self.comments = comments
        self.pr_labels = labels
        self.writes: list[tuple[str, str, Any]] = []

    def api(self, path: str, *, paginate: bool = False, jq: str | None = None) -> Any:
        if path.endswith("/comments?per_page=100"):
            return self.comments
        if "/labels?per_page=100" in path:
            return [{"name": name} for name in ("bug", "documentation", "dependencies")]
        if path.endswith("/reviews?per_page=100"):
            return []
        return {"head": {"sha": "b" * 40}, "labels": [{"name": name} for name in self.pr_labels]}

    def write(self, method: str, path: str, payload: dict[str, Any] | None = None) -> None:
        self.writes.append((method, path, payload))


def test_publish_without_a_context_posts_a_notice_and_removes_old_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeGitHub(comments=[], labels=["ai:ci-safe", "ai:review-lgtm", "bug"])
    monkeypatch.setattr(triage, "gh_api", fake.api)
    monkeypatch.setattr(triage, "gh_write", fake.write)
    triage.publish(REPO, 10, tmp_path / "missing.json", None)

    deleted = {path.rsplit("/", 1)[1] for method, path, _ in fake.writes if method == "DELETE"}
    assert deleted == {"ai:ci-safe", "ai:review-lgtm"}
    method, path, payload = fake.writes[-1]
    assert (method, path.endswith("/comments")) == ("POST", True)
    assert triage.COMMENT_MARKER in payload["body"]
    assert "did not run" in payload["body"]

    # The notice replaces the sticky comment of an earlier run.
    marker_comment = {"id": 5, "user": {"login": triage.BOT_LOGIN}, "body": triage.COMMENT_MARKER}
    later = FakeGitHub(comments=[marker_comment], labels=[])
    monkeypatch.setattr(triage, "gh_api", later.api)
    monkeypatch.setattr(triage, "gh_write", later.write)
    triage.publish(REPO, 10, tmp_path / "missing.json", None)
    assert later.writes[-1][0] == "PATCH"


def test_closed_pr_candidate_is_not_treated_as_duplicate_in_effective_verdicts() -> None:
    context = make_context(candidates=[{"number": 7, "title": "fix", "state": "closed", "url": ""}])
    result = triage.parse_triage_result(
        answer(duplicates=[{"pr": 7, "verdict": "duplicate", "reason": "same"}]),
        {7},
    )
    assert triage.effective_verdicts(context, result) == {7: "related"}
    assert triage.decide_labels(context, result) == {triage.LABEL_CI_SAFE}


@pytest.mark.parametrize(
    "path",
    [".skills/my-skill/SKILL.md", "CODEOWNERS"],
)
def test_agent_skills_and_codeowners_force_care(path: str) -> None:
    report = triage.assess_risk([{"filename": path, "status": "modified", "patch": "@@ -1 +1 @@\n+x"}])
    assert [finding.path for finding in report.forced] == [path]


def test_the_workflow_publishes_after_a_failed_prepare_and_has_no_reset_job() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/ai-pr-triage.yml").read_text("utf-8"))
    assert "reset-labels" not in workflow["jobs"]
    condition = workflow["jobs"]["publish"]["if"]
    assert "needs.context.outputs.run == 'true'" in condition
    assert "run_llm" not in condition
