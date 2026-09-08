"""Exercise release identity against real Git refs and plist fixtures."""

import importlib.util
import pathlib
import plistlib
import subprocess
import tempfile
import shutil
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "verify-release-identity.py"
spec = importlib.util.spec_from_file_location("release_identity", SCRIPT)
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


class ReleaseIdentityTests(unittest.TestCase):
    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = pathlib.Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "tag.gpgsign", "false")
        (self.repo / "VERSION").write_text("0.1.0\n")
        (self.repo / ".gitignore").write_text("/dist/\n")
        self.source = self.repo / "ironmlx-app/Packaging/Info.plist"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(plistlib.dumps({
            "CFBundleShortVersionString": "0.1.0", "CFBundleVersion": "1",
        }))
        self.git("add", ".")
        self.git("commit", "-qm", "baseline")
        self.commit = self.git("rev-parse", "HEAD")
        self.git("tag", "v0.1.0")
        self.app = self.repo / "dist/IronMLX.app"
        (self.app / "Contents").mkdir(parents=True)
        self.info = {
            "CFBundleIdentifier": "com.ironmlx.app",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "1",
            "IronMLXSourceCommit": self.commit,
            "IronMLXSourceTreeState": "clean",
        }
        self.write_info()

    def write_info(self):
        (self.app / "Contents/Info.plist").write_bytes(plistlib.dumps(self.info))

    def test_matching_source_and_bundle(self):
        self.assertEqual(identity.verify(self.repo, "v0.1.0"), self.commit)
        self.assertEqual(identity.verify(self.repo, "v0.1.0", self.app), self.commit)

    def test_annotated_tag_with_other_tags_on_same_commit(self):
        self.git("tag", "-d", "v0.1.0")
        self.git("tag", "-a", "v0.1.0", "-m", "release")
        self.git("tag", "other-tag")
        self.assertEqual(identity.verify(self.repo, "v0.1.0", self.app), self.commit)

    def test_wrong_version_and_preview_tag(self):
        for tag in ("v0.2.0", "v0.1.0-rc.1", "dev", "--help"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                identity.verify(self.repo, tag, self.app)

    def test_rc_packager_dispatches_only_matching_clean_candidates(self):
        scripts = self.repo / "scripts"
        scripts.mkdir()
        shutil.copy2(SCRIPT, scripts / SCRIPT.name)
        shutil.copy2(SCRIPT.parent / "package-release-candidate.sh",
                     scripts / "package-release-candidate.sh")
        dispatch = scripts / "package-development-preview.sh"
        dispatch.write_text('#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$@"\n')
        dispatch.chmod(0o755)
        self.git("add", ".")
        self.git("commit", "-qm", "install candidate packager")
        self.commit = self.git("rev-parse", "HEAD")
        self.info["IronMLXSourceCommit"] = self.commit
        self.write_info()
        self.git("tag", "v0.1.0-rc.1")
        command = ["bash", str(scripts / "package-release-candidate.sh"), "v0.1.0-rc.1"]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.endswith(
            f"v0.1.0-rc.1\n{self.commit}\nrelease-candidate\nvalidate\n"))
        self.info["IronMLXSourceTreeState"] = "dirty"
        self.write_info()
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("release-candidate", result.stdout)

    def test_material_validation_does_not_authorize_publication(self):
        scripts = self.repo / "scripts"
        scripts.mkdir()
        for name in ("verify-distribution-materials.sh", "release-legal-gate.sh",
                     "release-config.sh"):
            shutil.copy2(SCRIPT.parent / name, scripts / name)
        sbom_check = scripts / "verify-sbom.sh"
        sbom_check.write_text("#!/usr/bin/env bash\nexit 0\n")
        sbom_check.chmod(0o755)
        for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
                     "third-party-inventory.json", "docs/model-license-boundary.md",
                     "SBOM.cdx.json", "THIRD_PARTY_LICENSES/test.txt"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        validate = ["bash", str(scripts / "verify-distribution-materials.sh")]
        self.assertEqual(subprocess.run(validate, capture_output=True).returncode, 0)
        result = subprocess.run(["bash", str(scripts / "release-legal-gate.sh")],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("public distribution is disabled", result.stderr)
        (self.repo / "NOTICE").unlink()
        self.assertNotEqual(subprocess.run(validate, capture_output=True).returncode, 0)

    def test_candidate_accepts_matching_rc_and_stable_mode_rejects_it(self):
        for number in (1, 12):
            tag = f"v0.1.0-rc.{number}"
            self.git("tag", tag)
            self.assertEqual(identity.verify(self.repo, tag, self.app, candidate=True),
                             self.commit)
            with self.assertRaises(ValueError):
                identity.verify(self.repo, tag, self.app)

    def test_candidate_rejects_wrong_version_and_invalid_suffix(self):
        for tag in ("v0.2.0-rc.1", "v0.1.0", "v0.1.0-rc.0", "v0.1.0-rc.01",
                    "v0.1.0-beta.1", "v0.1.0-rc.1+build"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                identity.verify(self.repo, tag, self.app, candidate=True)

    def test_candidate_retains_clean_bundle_and_head_checks(self):
        tag = "v0.1.0-rc.1"
        self.git("tag", tag)
        for key, value in (("IronMLXSourceTreeState", "dirty"),
                           ("IronMLXSourceCommit", "0" * 40),
                           ("CFBundleShortVersionString", "0.1.0-rc.1")):
            original = self.info[key]
            self.info[key] = value
            self.write_info()
            with self.assertRaisesRegex(ValueError, key):
                identity.verify(self.repo, tag, self.app, candidate=True)
            self.info[key] = original
        self.write_info()
        dirty = self.repo / "untracked"
        dirty.write_text("dirty")
        with self.assertRaisesRegex(ValueError, "checkout must be clean"):
            identity.verify(self.repo, tag, self.app, candidate=True)
        dirty.unlink()
        self.git("commit", "--allow-empty", "-qm", "next")
        with self.assertRaisesRegex(ValueError, "does not point to HEAD"):
            identity.verify(self.repo, tag, self.app, candidate=True)

    def test_missing_tag_cannot_resolve_same_named_branch(self):
        self.git("tag", "-d", "v0.1.0")
        self.git("branch", "v0.1.0")
        with self.assertRaises(subprocess.CalledProcessError):
            identity.verify(self.repo, "v0.1.0", self.app)

    def test_tag_at_old_commit(self):
        self.git("commit", "--allow-empty", "-qm", "next")
        with self.assertRaisesRegex(ValueError, "does not point to HEAD"):
            identity.verify(self.repo, "v0.1.0", self.app)

    def test_dirty_tracked_staged_and_untracked_checkout(self):
        path = self.repo / "unexpected"
        for stage in (False, True):
            path.write_text("unexpected")
            if stage:
                self.git("add", "unexpected")
            with self.assertRaisesRegex(ValueError, "checkout must be clean"):
                identity.verify(self.repo, "v0.1.0", self.app)
            if stage:
                self.git("reset", "-q", "HEAD", "unexpected")
            path.unlink()
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "checkout must be clean"):
            identity.verify(self.repo, "v0.1.0", self.app)

    def test_bundle_metadata_mismatches_and_missing_fields(self):
        for key in list(self.info):
            original = self.info[key]
            for value in ("wrong", None):
                with self.subTest(key=key, value=value):
                    if value is None:
                        self.info.pop(key)
                    else:
                        self.info[key] = value
                    self.write_info()
                    with self.assertRaisesRegex(ValueError, key):
                        identity.verify(self.repo, "v0.1.0", self.app)
                    self.info[key] = original

    def test_source_version_mismatch(self):
        self.source.write_bytes(plistlib.dumps({
            "CFBundleShortVersionString": "0.2.0", "CFBundleVersion": "1",
        }))
        self.git("add", ".")
        self.git("commit", "-qm", "mismatched source")
        self.git("tag", "-f", "v0.1.0")
        with self.assertRaisesRegex(ValueError, "source App version"):
            identity.verify(self.repo, "v0.1.0")

    def test_malformed_bundle(self):
        (self.app / "Contents/Info.plist").write_bytes(b"invalid")
        with self.assertRaises(plistlib.InvalidFileException):
            identity.verify(self.repo, "v0.1.0", self.app)


if __name__ == "__main__":
    unittest.main()
