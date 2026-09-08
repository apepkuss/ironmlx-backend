#!/usr/bin/env python3
"""Bind a release tag, clean checkout, and optional App metadata."""

import argparse
import pathlib
import plistlib
import re
import subprocess
import sys


def verify(repo, tag, app=None, *, candidate=False):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    pattern = r"v([0-9]+\.[0-9]+\.[0-9]+)"
    if candidate:
        pattern += r"-rc\.[1-9][0-9]*"
    match = re.fullmatch(pattern, tag)
    require(match is not None,
            "candidate tag must be vX.Y.Z-rc.N (N >= 1)" if candidate
            else "release tag must be vX.Y.Z")
    version = (repo / "VERSION").read_text().strip()
    require(match.group(1) == version, "release tag does not match VERSION")
    commit = git("rev-parse", "HEAD")
    tag_commit = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    require(tag_commit == commit, "release tag does not point to HEAD")
    require(not git("status", "--porcelain=v1", "--untracked-files=all"),
            "release checkout must be clean (including untracked files)")
    with (repo / "ironmlx-app/Packaging/Info.plist").open("rb") as handle:
        source_info = plistlib.load(handle)
    require(source_info.get("CFBundleShortVersionString") == version,
            "source App version does not match VERSION")
    build = source_info.get("CFBundleVersion", "")
    require(isinstance(build, str) and re.fullmatch(r"[1-9][0-9]*", build),
            "source App build number must be a positive integer")
    if app is not None:
        with (app / "Contents/Info.plist").open("rb") as handle:
            info = plistlib.load(handle)
        expected = {
            "CFBundleIdentifier": "com.ironmlx.app",
            "CFBundleShortVersionString": version,
            "CFBundleVersion": build,
            "IronMLXSourceCommit": commit,
            "IronMLXSourceTreeState": "clean",
        }
        for key, value in expected.items():
            require(info.get(key) == value, f"Bundle {key} must equal {value}")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true",
                        help="validate an RC tag only; never enables stable publication")
    parser.add_argument("tag", help="existing stable tag, or RC tag with --candidate")
    parser.add_argument("app", nargs="?", type=pathlib.Path, help="also verify this App Bundle")
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parent.parent
    try:
        commit = verify(repo, args.tag, args.app, candidate=args.candidate)
    except (ValueError, OSError, plistlib.InvalidFileException, subprocess.CalledProcessError) as error:
        print(f"error: release identity verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Release identity passed: {args.tag} at {commit}" +
          (" with matching clean Bundle metadata" if args.app else " (source only)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
