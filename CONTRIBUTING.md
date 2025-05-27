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
uv sync
```

#### Use a specific Python version (optional)

If you need to work with a specific version of Python, you can create a new virtual environment for that version
and run the sync command:

```bash
uv venv --python 3.12
uv sync
```

More detailed options are described on the [Using Python environments](https://docs.astral.sh/uv/pip/environments/) documentation.

#### Add a new dependency

Simply use the `uv add` command. The `pyproject.toml` and `uv.lock` files will be updated.

```bash
uv add [OPTIONS] <PACKAGES|--requirements <REQUIREMENTS>>
```

### Code sytle guidelines

We use the following tools to enforce code style:

- isort, to sort imports
- Black, to format code
- Flake8, to lint code
- autoflake, to remove unused variables and imports
- [MyPy](https://mypy.readthedocs.io), as static type checker

A set of styling checks, as well as regression tests, are defined and managed through the [pre-commit](https://pre-commit.com/) framework. To ensure that those scripts run automatically before a commit is finalized, install `pre-commit` on your local repository:

```bash
uv run pre-commit install
```

To run the checks on-demand, type:

```bash
uv run pre-commit run --all-files
```

Note: Checks like `Black` and `isort` will _fail_ if they modify files. This is because `pre-commit` doesn't like to see files modified by their hooks. In these cases, `git add` the modified files and `git commit` again.


### Documentation

We use [JSON Schema for Humans](https://github.com/coveooss/json-schema-for-humans) to generate Markdown pages documenting the JSON schema of the Docling objects.

The documentation pages are stored in [docs](./docs/) folder and are updated at every commit, as part of the `pre-commit` check hooks.
To generate the documentation on-demand, run:

```bash
uv run python -m docling_core.utils.generate_docs docs
```
