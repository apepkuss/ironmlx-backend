"""Public update policy and real Sparkle signing tests, without publishing."""
import importlib.util
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

SCRIPTS = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


configuration = module("configure-app-updates")
payload = module("package-app-update")
publish = module("publish-update-feed")


class PolicyTests(unittest.TestCase):
    def test_channels_and_signed_key_requirements(self):
        key = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        for channel in ("stable", "release-candidate"):
            configuration.validate(channel, f"https://example.com/{channel}.xml", key)
            for url in ("http://example.com/feed", "https://127.0.0.1/feed", "https://user:pass@example.com/feed"):
                with self.assertRaises(ValueError):
                    configuration.validate(channel, url, key)
        with self.assertRaises(ValueError):
            configuration.validate("development", "https://example.com/feed", key)
        with self.assertRaises(ValueError):
            configuration.validate("disabled", "https://example.com/feed", key)

    def test_build_monotonicity_and_idempotent_retry(self):
        old = dict(channel="stable", build=10, tag="v0.1.0")
        self.assertFalse(publish.newer(old, old.copy()))
        self.assertTrue(publish.newer(old, dict(channel="stable", build=11, tag="v0.1.1")))
        for new in (dict(channel="stable", build=9), dict(channel="stable", build=10),
                    dict(channel="release-candidate", build=11)):
            with self.assertRaises(ValueError):
                publish.newer(old, new)

    def test_payload_rejects_channel_and_version_mismatch(self):
        info = dict(IronMLXUpdateChannel="stable", CFBundleShortVersionString="0.1.0",
                    CFBundleVersion="2", CFBundleIdentifier="com.ironmlx.app",
                    SURequireSignedFeed=True, SUVerifyUpdateBeforeExtraction=True)
        self.assertEqual(payload.validate(info, "v0.1.0"), "stable")
        for tag in ("v0.1.0-rc.1", "v0.2.0", "v0.1.0-rc.0"):
            with self.assertRaises(ValueError):
                payload.validate(info, tag)


SIGN = next((path for path in (
    SCRIPTS.parent / "ironmlx-app/.build/artifacts/sparkle/Sparkle/bin/sign_update",
    SCRIPTS.parent / ".build/app-bundle/swift-build/artifacts/sparkle/Sparkle/bin/sign_update",
) if path.exists()), Path("/missing-sign-update"))


@unittest.skipUnless(SIGN.exists() and shutil.which("swift"), "requires pinned Sparkle macOS tools")
class SigningTests(unittest.TestCase):
    def test_archive_feed_signatures_and_key_binding(self):
        with tempfile.TemporaryDirectory(prefix="ironmlx-update-signing-") as temporary:
            root = Path(temporary)
            private = root / "key"
            public = subprocess.check_output(["swift", str(SCRIPTS / "generate-development-update-key.swift"), str(private)], text=True).strip()
            app = root / "IronMLX.app"
            (app / "Contents").mkdir(parents=True)
            for channel, tag in (("stable", "v0.1.0"), ("release-candidate", "v0.1.0-rc.1")):
                info = dict(IronMLXUpdateChannel=channel, CFBundleShortVersionString="0.1.0",
                            CFBundleVersion="2", CFBundleIdentifier="com.ironmlx.app",
                            SUPublicEDKey=public, SURequireSignedFeed=True, SUVerifyUpdateBeforeExtraction=True,
                            SUFeedURL=f"https://raw.githubusercontent.com/test/repo/updates/{channel}.xml",
                            IronMLXSourceCommit="a" * 40, LSMinimumSystemVersion="26.2")
                (app / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
                output = root / channel
                command = ["python3", str(SCRIPTS / "package-app-update.py"), str(app), tag, str(output),
                           "--repository", "test/repo", "--key-file", str(private), "--sign-tool", str(SIGN)]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads((output / "update.json").read_text())
                feed = output / metadata["feed"]
                item = ET.parse(feed).find("./channel/item")
                self.assertEqual(item.find(f"{{{payload.SPARKLE}}}version").text, "2")
                signature = item.find("enclosure").get(f"{{{payload.SPARKLE}}}edSignature")
                archive = output / metadata["archive"]
                archive.write_bytes(archive.read_bytes() + b"tampered")
                verify = [str(SIGN), "--ed-key-file", str(private), "--verify"]
                self.assertNotEqual(subprocess.run([*verify, str(archive), signature], capture_output=True).returncode, 0)
                feed.write_text(feed.read_text().replace("<title>", "<title>TAMPERED"))
                self.assertNotEqual(subprocess.run([*verify, str(feed)], capture_output=True).returncode, 0)
                info["SUPublicEDKey"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
                (app / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
                command[4] = str(root / f"wrong-key-{channel}")
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("signing key does not match", result.stderr)
