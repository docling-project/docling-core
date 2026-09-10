#!/bin/bash

set -e  # trigger failure on error - do not remove!
set -x  # display command on output

# Build each package into its own dist subdirectory so the PyPI publish action
# can upload them independently (a single `dist/` makes the second publish step
# re-upload the first package's files and fail on `skip-existing: false`).

echo "Building docling-core package..."
uv build --package docling-core --out-dir dist/docling-core

echo "Building dclq package..."
uv build --package dclq --out-dir dist/dclq

echo "Build complete."
echo "docling-core artifacts:"
ls -lh dist/docling-core/
echo "dclq artifacts:"
ls -lh dist/dclq/
