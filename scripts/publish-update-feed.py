#!/usr/bin/env python3
"""Publish a signed feed only after its exact update archive is publicly released."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from urllib.parse import quote


def check(condition, message):
    if not condition:
        raise ValueError(message)


def api(repository, route, body=None, method=None, missing=False):
    command = ["gh", "api", f"repos/{repository}/{route}"]
    if body is not None:
        command += ["--method", method or "POST", "--input", "-"]
    result = subprocess.run(command, input=json.dumps(body) if body is not None else None,
                            text=True, capture_output=True)
    if missing and result.returncode and "(HTTP 404)" in result.stderr:
        return None
    if result.returncode:
        raise RuntimeError(f"GitHub API failed for {route}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def newer(previous, current):
    if previous is None:
        return True
    check(previous["channel"] == current["channel"], "feed channel mismatch")
    if previous == current:
        return False  # Safe retry after the feed was already published.
    check(current["build"] > previous["build"], "update build number must increase; refusing rollback or replacement")
    return True


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--branch", default="updates")
    args = parser.parse_args()
    check(re.fullmatch(r"[\w.-]+/[\w.-]+", args.repository), "invalid repository")
    check(re.fullmatch(r"[\w.-]+", args.branch), "invalid feed branch")
    repo = args.repository
    data = json.loads((args.output / "update.json").read_text())
    subprocess.run([str(Path(__file__).resolve().parent / "release-legal-gate.sh")], check=True)
    channel = data["channel"]
    check(channel in ("stable", "release-candidate"), "invalid channel")
    tag = data["tag"]
    pattern = r"v[0-9]+\.[0-9]+\.[0-9]+" + (r"-rc\.[1-9][0-9]*" if channel == "release-candidate" else "")
    check(re.fullmatch(pattern, tag), "tag/channel mismatch")
    identity = ["python3", str(Path(__file__).resolve().parent / "verify-release-identity.py")]
    if channel == "release-candidate":
        identity.append("--candidate")
    subprocess.run([*identity, tag], check=True)
    check(data["archive"] == f"IronMLX-{tag}-update.zip" and data["feed"] == f"{channel}.xml", "unexpected payload names")
    expected_feed = f"https://raw.githubusercontent.com/{repo}/{args.branch}/{channel}.xml"
    check(data["feed_url"] == expected_feed, "Bundle feed URL does not match deployment destination")
    for key in ("archive", "feed"):
        check(sha(args.output / data[key]) == data[key + "_sha256"], f"{key} checksum mismatch")
    enclosure = ET.parse(args.output / data["feed"]).find("./channel/item/enclosure")
    expected_url = f"https://github.com/{repo}/releases/download/{quote(tag)}/{quote(data['archive'])}"
    check(enclosure is not None and enclosure.get("url") == expected_url, "feed archive URL mismatch")
    release = api(repo, f"releases/tags/{tag}")
    check(not release["draft"] and release["prerelease"] == (channel == "release-candidate"), "release visibility/channel mismatch")
    check(api(repo, f"commits/{tag}")["sha"] == data["source_commit"], "release tag moved")
    with tempfile.TemporaryDirectory(prefix="ironmlx-download-update-") as temporary:
        subprocess.run(["gh", "release", "download", tag, "--repo", repo,
                        "--pattern", data["archive"], "--dir", temporary], check=True)
        check(sha(Path(temporary) / data["archive"]) == data["archive_sha256"], "published update archive differs")
    ref = api(repo, f"git/ref/heads/{args.branch}", missing=True)
    previous = None
    parent = None
    tree_body = {}
    if ref:
        parent = ref["object"]["sha"]
        tree_body["base_tree"] = api(repo, f"git/commits/{parent}")["tree"]["sha"]
        old = api(repo, f"contents/{channel}.json?ref={parent}", missing=True)
        if old:
            previous = json.loads(base64.b64decode(old["content"]))
    if not newer(previous, data):
        print("Update feed already published with identical metadata")
        return
    tree_body["tree"] = [
        dict(path=f"{channel}.xml", mode="100644", type="blob", content=(args.output / data["feed"]).read_text()),
        dict(path=f"{channel}.json", mode="100644", type="blob", content=json.dumps(data, indent=2) + "\n"),
    ]
    tree = api(repo, "git/trees", tree_body)
    commit = api(repo, "git/commits", dict(message=f"chore: publish {tag} update feed", tree=tree["sha"],
                                         parents=[parent] if parent else []))
    if parent:
        api(repo, f"git/refs/heads/{args.branch}", dict(sha=commit["sha"], force=False), method="PATCH")
    else:
        api(repo, "git/refs", dict(ref=f"refs/heads/{args.branch}", sha=commit["sha"]))
    published = api(repo, f"contents/{channel}.xml?ref={commit['sha']}")
    check(hashlib.sha256(base64.b64decode(published["content"])).hexdigest() == data["feed_sha256"], "published feed differs")
    print(f"Published verified update feed: {expected_feed}")


if __name__ == "__main__":
    main()
