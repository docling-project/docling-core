#!/usr/bin/env python3
"""Assert that dlgrep is releasable in lockstep with docling-core.

dlgrep and docling-core ship from the same commit, at the same version, and
dlgrep pins docling-core exactly. If any of the three drifts apart, the
published dlgrep wheel is uninstallable (its pin names a docling-core version
that was never released) - so fail loudly here instead.

Run by pre-commit and by .github/workflows/checks.yml.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
PIN_RE = re.compile(r'"docling-core==([^"]+)"')


def main() -> int:
    core = VERSION_RE.search((ROOT / "pyproject.toml").read_text())
    dlgrep_toml = (ROOT / "packages" / "dlgrep" / "pyproject.toml").read_text()
    dlgrep = VERSION_RE.search(dlgrep_toml)
    pin = PIN_RE.search(dlgrep_toml)

    if not (core and dlgrep and pin):
        print("check_lockstep: could not read versions - did a pyproject.toml layout change?")
        return 1

    errors = []
    if dlgrep.group(1) != core.group(1):
        errors.append(f"dlgrep version {dlgrep.group(1)} != docling-core version {core.group(1)}")
    if pin.group(1) != core.group(1):
        errors.append(f"dlgrep pins docling-core=={pin.group(1)} != docling-core version {core.group(1)}")

    for error in errors:
        print(f"check_lockstep: {error}")
    if errors:
        print("check_lockstep: bump both via .github/scripts/release.sh, do not edit versions by hand")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
