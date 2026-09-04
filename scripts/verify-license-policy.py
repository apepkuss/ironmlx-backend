#!/usr/bin/env python3
"""Enforce the repository's machine-checkable third-party license policy."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "third-party-inventory.json"
LICENSES = ROOT / "THIRD_PARTY_LICENSES"
FORBIDDEN = re.compile(
    r"(?i)(?<![A-Z])(AGPL|LGPL|GPL|SSPL|BUSL|ELASTIC-2\.0)(?![A-Z])"
)


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_expression(value: object, owner: str) -> None:
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} has no license expression")
    if FORBIDDEN.search(value):
        fail(f"{owner} uses a policy-forbidden license expression: {value}")


def check_license_file(name: object, owner: str) -> None:
    if not isinstance(name, str) or not name or Path(name).name != name:
        fail(f"{owner} has an invalid license file name: {name!r}")
    path = LICENSES / name
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"{owner} references a missing or empty license file: {name}")


def main() -> None:
    if not INVENTORY.is_file():
        fail(f"third-party inventory is missing: {INVENTORY}")
    if not LICENSES.is_dir():
        fail(f"third-party license directory is missing: {LICENSES}")

    try:
        inventory = json.loads(INVENTORY.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read third-party inventory: {exc}")

    for crate in inventory.get("rust", {}).get("crates", []):
        owner = f"Rust crate {crate.get('name', '<unknown>')}"
        check_expression(crate.get("license_expression"), owner)
        files = crate.get("license_files", [])
        if not files:
            fail(f"{owner} has no license text mapping")
        for name in files:
            check_license_file(name, owner)

    for section, key in (
        ("native", "dependencies"),
        ("swift", "external_packages"),
        ("bundled_assets", "assets"),
    ):
        for item in inventory.get(section, {}).get(key, []):
            owner = f"{section} component {item.get('component', '<unknown>')}"
            check_expression(item.get("license"), owner)
            check_license_file(item.get("license_file"), owner)

    metadata = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    cargo = json.loads(metadata.stdout)
    workspace_ids = set(cargo["workspace_members"])
    for package in cargo["packages"]:
        if package["id"] in workspace_ids:
            license_value = package.get("license")
            if license_value != "Apache-2.0":
                fail(
                    f"workspace crate {package['name']} must declare Apache-2.0, "
                    f"found {license_value!r}"
                )

    print("License policy passed: workspace Apache-2.0 and third-party materials are allowed")


if __name__ == "__main__":
    main()
