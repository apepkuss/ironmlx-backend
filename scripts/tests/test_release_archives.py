"""Real ZIP/DMG content round trips; no signing credentials or publication."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "release-archives.py"
spec = importlib.util.spec_from_file_location("archives", SCRIPT)
archives = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archives)


@unittest.skipUnless(shutil.which("hdiutil") and shutil.which("ditto"), "requires macOS archive tools")
class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="ironmlx-archive-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.repo = Path(cls.temp.name) / "repo"
        cls.repo.mkdir()
        for name in archives.MATERIALS:
            path = archives.material_source(cls.repo, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        (cls.repo / "THIRD_PARTY_LICENSES").mkdir()
        (cls.repo / "THIRD_PARTY_LICENSES/license.txt").write_text("license fixture")
        (cls.repo / "scripts").mkdir()
        shutil.copy2(SCRIPT.parent / "verify-model-distribution-boundary.sh", cls.repo / "scripts")
        cls.app = cls.repo / "Reference.app"
        (cls.app / "Contents/MacOS").mkdir(parents=True)
        executable = cls.app / "Contents/MacOS/test"
        executable.write_text("fixture executable")
        executable.chmod(0o755)
        (cls.app / "Contents/Info.plist").write_text("fixture metadata")
        (cls.app / "Contents/current").symlink_to("MacOS")
        cls.assets = cls.repo / "assets"
        cls.package = "IronMLX-0.1.0"
        archives.assemble(cls.repo, cls.app, cls.assets, cls.package)

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="ironmlx-archive-case-")
        self.addCleanup(self.scratch.cleanup)
        self.output = Path(self.scratch.name) / "assets"
        shutil.copytree(self.assets, self.output)

    def test_round_trip_and_no_output_overwrite(self):
        archives.verify(self.repo, self.app, self.output, self.package)
        with self.assertRaisesRegex(ValueError, "must be empty"):
            archives.assemble(self.repo, self.app, self.output, self.package)

    def test_incomplete_manifest(self):
        manifest = self.output / "SHA256SUMS"
        manifest.write_text("\n".join(manifest.read_text().splitlines()[1:]) + "\n")
        with self.assertRaisesRegex(ValueError, "coverage"):
            archives.verify(self.repo, self.app, self.output, self.package)

    def test_modified_material_even_with_new_checksum(self):
        (self.output / "NOTICE").write_text("changed")
        (self.output / "SHA256SUMS").write_text(archives.checksums(self.repo, self.output, self.package))
        with self.assertRaisesRegex(ValueError, "material mismatch"):
            archives.verify(self.repo, self.app, self.output, self.package)

    def test_missing_material(self):
        (self.output / "LICENSE").unlink()
        with self.assertRaises(FileNotFoundError):
            archives.verify(self.repo, self.app, self.output, self.package)

    def replace_zip_entry(self, name, content):
        path = self.output / f"{self.package}.zip"
        replacement = path.with_suffix(".tmp")
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(replacement, "w") as target:
            for entry in source.infolist():
                target.writestr(entry, content if entry.filename == name else source.read(entry))
        replacement.replace(path)
        (self.output / "SHA256SUMS").write_text(archives.checksums(self.repo, self.output, self.package))

    def test_changed_bundle_identity_even_with_new_checksum(self):
        self.replace_zip_entry(f"{self.package}/IronMLX.app/Contents/Info.plist", b"wrong source/version")
        with self.assertRaisesRegex(ValueError, "App differs"):
            archives.verify(self.repo, self.app, self.output, self.package)

    def test_unsafe_zip_path(self):
        with zipfile.ZipFile(self.output / f"{self.package}.zip", "a") as archive:
            archive.writestr("../escape", "unsafe")
        (self.output / "SHA256SUMS").write_text(archives.checksums(self.repo, self.output, self.package))
        with self.assertRaisesRegex(ValueError, "unsafe ZIP"):
            archives.verify(self.repo, self.app, self.output, self.package)
