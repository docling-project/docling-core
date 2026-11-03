# AI PR triage and first review

The `AI PR Triage` workflow (`.github/workflows/ai-pr-triage.yml`) runs on every
non-draft PR that is not from a bot. It gives maintainers an advisory triage
and a first review before they approve CI or review the code. It is a port of
the same workflow in [docling](https://github.com/docling-project/docling).

It never checks out or runs PR code. It reads the PR through the GitHub API,
applies deterministic rules, and asks Bob Shell in a read-only mode for an
assessment. It updates one sticky comment and sets labels. The workflow is
never a blocking check: every step uses `continue-on-error`, so an error (for
example a missing secret, an API outage, or an invalid model answer) shows as
a warning annotation on the run, and the PR check stays green.

## Labels

Triage labels:

- `ai:ci-safe`: no deterministic rule matched and the model found no concern.
- `ai:ci-needs-care`: a sensitive path changed, or the model reported a
  concern. Read the comment before you approve CI. Sensitive paths include
  `.github/**`, `conftest.py`, `pyproject.toml`, `uv.lock`, `Makefile`,
  `.pre-commit-config.yaml`, `scripts/**`, shell scripts, agent instruction
  files (`AGENTS.md`, `CLAUDE.md`, `.claude/**`, ...), and Python files below
  `tests/data/` (pytest collects `tests/` recursively). The full list is
  `SENSITIVE_PATHS` in `.github/scripts/ai_pr_triage.py`.
- `ai:possible-duplicate`: the model rated an open or merged PR as a
  duplicate. A closed PR that was not merged, or a small shared part, counts
  as related, not as a duplicate. Candidates come from shared issue references
  and overlapping diff hunks.

Review labels:

- `ai:review-lgtm`: the AI first review found no open `blocker`.
- `ai:review-changes`: the AI first review found a `blocker`. See its inline
  comments.

A review label describes the commit of the last AI review. A new push removes
all `ai:*` labels and runs the triage again.

The labels are advisory. They never approve a workflow run.

### Topic labels

The triage also adds topic labels from the existing repository labels:
`bug`, `enhancement`, `documentation`, `performance`, `dependencies`,
`security`, `ci`, `doclang`, `doctags`, `document-model`, `serialization`,
`tables`, `chunking`, `media`, `cli`. Rules map the conventional-commit title
(`fix` → `bug`, scope `(doclang)` → `doclang`) and the changed source paths to
labels, and the model can add up to three more. The rules are in
`.github/scripts/pr_topic_labels.py`. The workflow adds topic labels only on
the first triage of a PR, never creates a label, and never removes one, so
maintainers can correct them.

## The first review

The first review is a `COMMENT` review with inline comments. It never approves
or requests changes. It uses three severities: `blocker`, `question`, and
`suggestion` (defined in `.github/ai-triage/review-prompt.md`), and it reports
at most 5 findings. A `question` or a `suggestion` never sets
`ai:review-changes`. The review reads the PR discussion, so it does not repeat
points that a reviewer already made or that the author answered. It reads the
source of the installed dependencies (from the base branch lock file) before
it makes a claim about them.

The review focuses on what CI cannot check. For example, it does not report
structural schema changes, because `scripts/check_compat_projectors.py`
already enforces the `CURRENT_VERSION` rules in
[CONTRIBUTING.md](../CONTRIBUTING.md#breaking-changes-and-schema-bumps). It
looks for changes of meaning in the JSON format, lost content in serializer
round trips, and public API breaks that affect downstream packages.

The review does not run for PRs rated as duplicates of an open PR from another
author, for PRs with more than 60 source files or 2000 changed source lines
(`tests/data/` and `uv.lock` do not count), or for a commit that it already
reviewed. A PR gets at most two automatic AI reviews: the first review, and
one review of the next push. That review covers only the new commits, if the
history was not rewritten, and gives the status of each earlier finding:
`open`, `fixed`, or `answered`. Use `/ai review` for more reviews.

## Reference data

The triage comment summarizes changed files below `tests/data/`. The tests
write nearly every file there when `DOCLING_GEN_TEST_DATA=1` is set, so the
summary covers all of `tests/data/` except source code (the code-chunking
samples). It separates formatting-only and coordinate-only changes from text,
table, and structure changes, and reports changed images by size. This helps
with the two approvals that Mergify requires when `tests/` changes.

## `/ai` commands

After a maintainer (`OWNER`, `MEMBER`, or `COLLABORATOR`) comments on or
reviews a PR, a push runs only the triage, not the AI review. Maintainers can
ask for more with a PR comment that starts with one of these commands:

- `/ai review` (or `/ai`): a new, full AI review of the current commit. It
  also runs for a commit that was already reviewed and for a PR rated as a
  duplicate. The size limits still apply.
- `/ai triage`: run the triage again.

The bot adds a 👀 reaction to confirm the command. Commands from other users
are ignored.

Users with write access can also start the workflow by hand for one open PR,
without a comment. This is useful to catch up on PRs after an outage of the
workflow:

```bash
gh workflow run "AI PR Triage" -f pr=1234 -f mode=triage
```

`mode=triage` follows the rules of a push. `mode=review` also forces a full
review, like `/ai review`.

## Setup (repository admins)

The workflow needs:

- The secret `BOB_API_KEY`: a Bob API key with the Inference scope. Without
  it, the comment shows only the deterministic checks (sensitive paths,
  duplicate candidates, reference data summary, topic labels).
- The five `ai:*` labels. The workflow cannot add a label that does not exist:

  ```bash
  R=docling-project/docling-core
  gh label create "ai:ci-safe"            -R "$R" -c 0e8a16 -d "AI triage found no CI-safety concern (advisory)"
  gh label create "ai:ci-needs-care"      -R "$R" -c d93f0b -d "AI triage: check CI-safety findings before you approve CI"
  gh label create "ai:possible-duplicate" -R "$R" -c fbca04 -d "AI triage: another PR may fix the same problem"
  gh label create "ai:review-lgtm"        -R "$R" -c c2e0c6 -d "AI first review found no blocker or major issue (advisory)"
  gh label create "ai:review-changes"     -R "$R" -c f9d0c4 -d "AI first review suggests changes"
  ```

Optional repository variables: `BOB_TEAM_ID` (for `general` keys),
`BOB_TRIAGE_MAX_COST` (default `2`), and `BOB_REVIEW_MAX_COST` (default `3`).

## Files

| File | Role |
|---|---|
| `.github/workflows/ai-pr-triage.yml` | The workflow and its jobs |
| `.github/scripts/ai_pr_event.py` | Which PR and mode an event runs for |
| `.github/scripts/ai_pr_triage.py` | Triage: data collection, answer validation, labels, comment |
| `.github/scripts/ai_pr_review.py` | First review: data collection, validation, posting |
| `.github/scripts/groundtruth_diff.py` | Reference data summary |
| `.github/scripts/pr_topic_labels.py` | Topic label rules |
| `.github/ai-triage/*.md` | Prompts |
| `.github/ai-triage/custom_modes.yaml` | Read-only Bob Shell modes |
| `.github/actions/run-bobshell/action.yml` | Pinned Bob Shell install and run |

The scripts use only the Python standard library. Their tests are
`tests/test_ai_pr_*.py`, `tests/test_groundtruth_diff.py`, and
`tests/test_pr_topic_labels.py`.
