## Contributing In General
Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

For more details on the contributing guidelines head to the Docling Project [community repository](https://github.com/docling-project/community).

## Developing

### Usage of uv

We use [uv](https://docs.astral.sh/uv/) as package and project manager.

#### Installation

To install `uv`, check the documentation on [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

#### Create an environment and sync it

You can use the `uv sync` to create a project virtual environment (if it does not already exist) and sync
the project's dependencies with the environment.

```bash
uv sync --all-extras
```

#### Use a specific Python version (optional)

If you need to work with a specific version of Python, you can create a new virtual environment for that version
and run the sync command:

```bash
uv venv --python 3.12
uv sync --all-extras
```

More detailed options are described on the [Using Python environments](https://docs.astral.sh/uv/pip/environments/) documentation.

#### Add a new dependency

Simply use the `uv add` command. The `pyproject.toml` and `uv.lock` files will be updated.

```bash
uv add [OPTIONS] <PACKAGES|--requirements <REQUIREMENTS>>
```

### Code style guidelines

We use the following tools to enforce code style:

- [Ruff](https://docs.astral.sh/ruff/), to format and lint code
- [MyPy](https://mypy.readthedocs.io), as static type checker

A set of styling checks, as well as regression tests, are defined and managed through the [pre-commit](https://pre-commit.com/) framework. To ensure that those scripts run automatically before a commit is finalized, install `pre-commit` on your local repository:

```bash
uv run pre-commit install
```

To run the checks on-demand, type:

```bash
uv run pre-commit run --all-files
```

### Documentation

We use [JSON Schema for Humans](https://github.com/coveooss/json-schema-for-humans) to generate Markdown pages documenting the JSON schema of the Docling objects.

The documentation pages are stored in [docs](./docs/) folder and are updated at every commit, as part of the `pre-commit` check hooks.
To generate the documentation on-demand, run:

```bash
uv run python -m docling_core.utils.generate_docs docs
```

## Breaking changes and schema bumps

`DoclingDocument` carries an embedded **schema version** (`CURRENT_VERSION` in
[`constants.py`](docling_core/types/doc/common/constants.py)).
This version is independent of the library (`pyproject.toml`) version and must
only be bumped when the serialized JSON representation changes in a way that
could cause an older `docling-core` to fail parsing.

### Change classification

| Change | Schema version bump | Required companion work |
|---|---|---|
| New enum value on a serialized field | **Minor bump** | Downgrade projector mapping the new value to the nearest old equivalent |
| New optional field on an existing model | **Minor bump** | Downgrade projector stripping the field |
| New required field | **Minor bump** | Downgrade projector supplying a safe default |
| New `DocItem` subtype (new discriminator label) | **Minor bump** | Projector converting to nearest existing type or dropping the item |
| Field removed or renamed | **Major bump** | Migration guide; old clients must upgrade |
| Semantic change (same name, different meaning) | **Major bump** | Migration guide |
| Internal-only change (no JSON impact) | **No bump needed** | — |

> [!TIP]
> A minor bump signals backward-incompatible wire changes while staying
> within the same major "era".  Clients on the same major version can
> request a server-side downgrade that uses uses
> `docling_core.compat.project_to()` to satisfy the request.

### Step-by-step checklist per schema minor bump

Suppose you are bumping `CURRENT_VERSION` from `1.N.0` to `1.(N+1).0`:

1. **Implement the model change** in the relevant file under
   `docling_core/types/doc/`.
2. **Bump `CURRENT_VERSION`** in
   `docling_core/types/doc/common/constants.py` (e.g. `"1.10.0"` →
   `"1.11.0"`).
3. **Append a history entry** to `SCHEMA_VERSION_HISTORY` in the same
   file, recording the library version and a short description.
4. **Add a downgrade projector** to `docling_core/compat.py`:

   ```python
   @register_projector(from_minor=N+1, to_minor=N)
   def _project_1_{N+1}_to_1_{N}(data: dict) -> dict:
       """Describe what this projector does."""
       data = dict(data)
       # … transform data …
       data["version"] = f"1.{N}.0"
       return data
   ```

   The function receives the raw `dict` produced by
   `DoclingDocument.model_dump(mode="python")` and must return a modified
   `dict` that is valid against schema `1.N.0`.

5. **The JSON Schema snapshot is captured automatically** — you do not need
   to do anything manually.  `scripts/check_compat_projectors.py` (run by
   the pre-commit hook) detects that `docs/schemas/DoclingDocument_1_N.json`
   is missing and copies the current `docs/DoclingDocument.json` to create
   it.  The hook ordering guarantees this copy happens *before*
   `generate_docs` overwrites `docs/DoclingDocument.json` with the new
   schema.  Add the generated snapshot file to your commit.

6. **Write a test class** named exactly `TestProjector_1_{N+1}_to_1_{N}`
   in `test/test_compat.py`.  Add at least one non-trivial test that
   exercises the specific field(s) the projector handles.
7. Run `python scripts/check_compat_projectors.py` locally to confirm
   the coverage check passes.

### Automated enforcement

The following checks run automatically as pre-commit hooks and in CI:

| Check | Mechanism | What it catches |
|---|---|---|
| Projector count | `scripts/check_compat_projectors.py` + `test_projector_count_matches_expected` | Missing `@register_projector` function |
| All steps covered | Same | A step (N→N-1) with no registered function |
| History table | Same | Missing `SCHEMA_VERSION_HISTORY` entry |
| Unannounced structural break | `scripts/check_compat_projectors.py` check 9 | New enum value, new field on strict model, new required field, or new union subtype added **without** bumping `CURRENT_VERSION` |
| Projector output shape | `test_every_projector_produces_valid_output` | Projector that crashes, drops required keys, or sets wrong version |
| Test class present | `test_every_projector_has_a_test_class` | Projector with no `TestProjector_1_N_to_1_{N-1}` class |

> [!WARNING]
> A PR that bumps `CURRENT_VERSION` without a matching projector *and* test class **will fail CI**.
> A PR that introduces a structural breaking change without bumping `CURRENT_VERSION` **will also fail CI**.

> [!NOTE]
> **What the schema diff cannot detect** — serialization-semantic changes
> (e.g. a changed `model_serializer`, a different roundtrip encoding for an
> existing field, reordered keys) leave the JSON Schema structurally identical
> so the diff is empty.  These changes are **invisible to static analysis**.
> They are the contributor's responsibility to identify and announce via a
> version bump.  The classification table above lists them as "Semantic change
> (same name, different meaning) → Major bump".  For a minor-bump
> serialization-semantic change (e.g. a new encoding that old clients can still
> parse) no automated check is needed — bump the version, provide the
> projector, write the test class; the existing checks will enforce those.

The `test_every_projector_produces_valid_output` test is parameterized
over every registered projector.  It runs a minimal document dict through
the projector and checks: the function does not raise; the output is a
`dict`; the `version` key is set to the correct target version; `schema_name`
is preserved; all required top-level keys are present.

> [!NOTE]
> **Why not `DoclingDocument.model_validate()`?** The contract of a projector
> is "produce a dict valid for the *target* SDK" — an older SDK not available
> at test time.  Validating with the *current* SDK would be incorrect: if a
> future schema change removes a field, the current model's `extra="forbid"`
> would reject a projected dict that contains the old field, producing a false
> failure.  Conversely, the current model accepts a superset of what old
> clients accept, so it would also produce false passes.  Full semantic
> correctness — "does this parse under the old SDK?" — is the responsibility
> of the hand-written per-projector test class.

### Using projectors on the server side

This is an implementation example for a server providing `DoclingDocument`
objects to a client application with version compatibility.

```python
from docling_core.compat import project_to
from docling_core.types.doc.document import DoclingDocument

doc: DoclingDocument = converter.convert(pdf)

# The client advertises its schema version in a request header.
client_version = request.headers.get("Accept-Schema-Version")
if client_version:
    doc = project_to(doc, target_version=client_version)

return doc.model_dump_json()
```
