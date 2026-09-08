#!/usr/bin/env python3
"""Create an App-only Sparkle archive and signed, isolated-channel appcast."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import plistlib
import re
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import quote

SPARKLE = "http://www.andymatuschak.org/xml-namespaces/sparkle"
ET.register_namespace("sparkle", SPARKLE)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate(info, tag):
    match = re.fullmatch(r"v([0-9]+\.[0-9]+\.[0-9]+)(-rc\.[1-9][0-9]*)?", tag)
    check(match is not None, "invalid release tag")
    channel = "release-candidate" if match.group(2) else "stable"
    check(info.get("IronMLXUpdateChannel") == channel, "App update channel does not match tag")
    check(info.get("CFBundleShortVersionString") == match.group(1), "App version does not match tag")
    check(re.fullmatch(r"[1-9][0-9]*", str(info.get("CFBundleVersion", ""))), "invalid build number")
    check(info.get("CFBundleIdentifier") == "com.ironmlx.app", "wrong App identifier")
    check(info.get("SURequireSignedFeed") is True and info.get("SUVerifyUpdateBeforeExtraction") is True,
          "App must require signed feed and pre-extraction verification")
    return channel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("tag")
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--sign-tool", required=True, type=Path)
    args = parser.parse_args()
    check(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository), "invalid GitHub repository")
    with (args.app / "Contents/Info.plist").open("rb") as handle:
        info = plistlib.load(handle)
    channel = validate(info, args.tag)
    # Never generate a payload that the installed App's pinned public key cannot verify.
    public_key = subprocess.check_output([
        "swift", str(Path(__file__).with_name("update-key-public.swift")), str(args.key_file)
    ], text=True).strip()
    check(public_key == info.get("SUPublicEDKey"), "signing key does not match Bundle public key")
    check(not args.output.exists() or not any(args.output.iterdir()), "update output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"IronMLX-{args.tag}-update.zip"
    subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(args.app), str(archive)], check=True)
    tool = [str(args.sign_tool), "--ed-key-file", str(args.key_file)]
    signature = subprocess.check_output([*tool, "-p", str(archive)], text=True).strip()
    subprocess.run([*tool, "--verify", str(archive), signature], check=True)
    url = f"https://github.com/{args.repository}/releases/download/{quote(args.tag)}/{quote(archive.name)}"
    rss = ET.Element("rss", {"version": "2.0"})
    feed_channel = ET.SubElement(rss, "channel")
    ET.SubElement(feed_channel, "title").text = f"IronMLX {channel}"
    item = ET.SubElement(feed_channel, "item")
    ET.SubElement(item, "title").text = f"IronMLX {args.tag}"
    ET.SubElement(item, "pubDate").text = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    ET.SubElement(item, f"{{{SPARKLE}}}version").text = info["CFBundleVersion"]
    ET.SubElement(item, f"{{{SPARKLE}}}shortVersionString").text = args.tag.removeprefix("v")
    ET.SubElement(item, f"{{{SPARKLE}}}minimumSystemVersion").text = info["LSMinimumSystemVersion"]
    if channel == "release-candidate":
        ET.SubElement(item, f"{{{SPARKLE}}}channel").text = channel
    ET.SubElement(item, "enclosure", {"url": url, "length": str(archive.stat().st_size),
                                    "type": "application/octet-stream", f"{{{SPARKLE}}}edSignature": signature})
    feed = args.output / f"{channel}.xml"
    ET.ElementTree(rss).write(feed, encoding="utf-8", xml_declaration=True)
    subprocess.run([*tool, str(feed)], check=True)
    subprocess.run([*tool, "--verify", str(feed)], check=True)
    metadata = dict(channel=channel, tag=args.tag, build=int(info["CFBundleVersion"]),
                    source_commit=info["IronMLXSourceCommit"], feed_url=info["SUFeedURL"],
                    archive=archive.name, archive_sha256=sha(archive), feed=feed.name, feed_sha256=sha(feed))
    (args.output / "update.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Signed update payload prepared: {args.output}")


if __name__ == "__main__":
    main()
