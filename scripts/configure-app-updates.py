#!/usr/bin/env python3
"""Validate update configuration and optionally apply it before App signing."""
import argparse
import base64
import plistlib
from pathlib import Path
from urllib.parse import urlsplit


def validate(channel, feed, key):
    if channel == "disabled":
        if feed or key:
            raise ValueError("disabled updates cannot specify a feed or key")
        return
    if channel not in ("development", "stable", "release-candidate"):
        raise ValueError("unsupported update channel")
    url = urlsplit(feed)
    if url.scheme != "https" or not url.hostname or url.username is not None or url.password is not None or url.fragment:
        raise ValueError("update feed requires HTTPS without credentials or fragment")
    loopback = url.hostname.lower() in ("localhost", "127.0.0.1", "::1")
    if loopback != (channel == "development"):
        raise ValueError("development requires loopback; public channels require a remote host")
    if len(base64.b64decode(key, validate=True)) != 32:
        raise ValueError("update public EdDSA key must decode to 32 bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channel")
    parser.add_argument("feed")
    parser.add_argument("key")
    parser.add_argument("--plist", type=Path)
    args = parser.parse_args()
    validate(args.channel, args.feed, args.key)
    if args.plist:
        with args.plist.open("rb") as handle:
            info = plistlib.load(handle)
        for name in ("IronMLXUpdateChannel", "SUFeedURL", "SUPublicEDKey", "SUEnableAutomaticChecks",
                     "SUAutomaticallyUpdate", "SUEnableSystemProfiling", "SURequireSignedFeed",
                     "SUVerifyUpdateBeforeExtraction"):
            info.pop(name, None)
        if args.channel != "disabled":
            info.update(IronMLXUpdateChannel=args.channel, SUFeedURL=args.feed, SUPublicEDKey=args.key,
                        SUEnableAutomaticChecks=True, SUAutomaticallyUpdate=True,
                        SUEnableSystemProfiling=False, SURequireSignedFeed=True,
                        SUVerifyUpdateBeforeExtraction=True)
        args.plist.write_bytes(plistlib.dumps(info))


if __name__ == "__main__":
    main()
