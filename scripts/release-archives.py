#!/usr/bin/env python3
"""Assemble or verify release archives without granting distribution authorization.

Requires macOS ditto/hdiutil. The reference App is never modified. Stable
identity, signing, notarization and publication authorization are caller gates.
"""
import argparse
import hashlib
import os
from pathlib import Path
import plistlib
import posixpath
import shutil
import stat
import subprocess
import tempfile
import zipfile

MATERIALS = ("LICENSE", "NOTICE", "SBOM.cdx.json", "THIRD_PARTY_NOTICES.md",
             "third-party-inventory.json", "model-license-boundary.md")


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True, stdout=subprocess.DEVNULL)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inventory(root):
    result = {}
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(base) / name
            rel = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if path.is_symlink():
                target = os.readlink(path)
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"escaping symlink: {path}")
                result[rel] = ("link", target)
            elif path.is_file():
                result[rel] = ("file", digest(path), stat.S_IMODE(mode) & 0o111)
            elif path.is_dir():
                result[rel] = ("directory",)
            else:
                raise ValueError(f"unsupported archive entry: {path}")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def material_source(repo, name):
    return repo / ("docs/model-license-boundary.md" if name == "model-license-boundary.md" else name)


def check_materials(repo, root):
    for name in MATERIALS:
        require((root / name).is_file() and not (root / name).is_symlink(), f"missing material: {name}")
        require(digest(root / name) == digest(material_source(repo, name)), f"material mismatch: {name}")
    require(inventory(root / "THIRD_PARTY_LICENSES") == inventory(repo / "THIRD_PARTY_LICENSES")
            and (root / "THIRD_PARTY_LICENSES").is_dir(), "license directory mismatch")


def asset_names(repo, package):
    return sorted([f"{package}.zip", f"{package}.dmg", *MATERIALS] + [
        f"THIRD_PARTY_LICENSES/{path.relative_to(repo / 'THIRD_PARTY_LICENSES').as_posix()}"
        for path in (repo / "THIRD_PARTY_LICENSES").rglob("*") if path.is_file()])


def checksums(repo, output, package):
    return "".join(f"{digest(output / name)}  {name}\n" for name in asset_names(repo, package))


def verify(repo, app, output, package):
    # Compare the entire manifest, not only the entries an archive supplied.
    require((output / "SHA256SUMS").read_text() == checksums(repo, output, package),
            "SHA256SUMS mismatch or incomplete coverage")
    check_materials(repo, output)
    require({p.name for p in output.iterdir()} == {
        f"{package}.zip", f"{package}.dmg", "SHA256SUMS", *MATERIALS, "THIRD_PARTY_LICENSES"},
        "unexpected standalone assets")
    expected = inventory(app)

    def check_root(root):
        check_materials(repo, root)
        require({p.name for p in root.iterdir()} == {*MATERIALS, "THIRD_PARTY_LICENSES", "IronMLX.app"},
                "unexpected or missing archive root entries")
        require(inventory(root / "IronMLX.app") == expected, "archived App differs from reference App")
        run(repo / "scripts/verify-model-distribution-boundary.sh", root)

    zip_path = output / f"{package}.zip"
    run(repo / "scripts/verify-model-distribution-boundary.sh", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for entry in archive.infolist():
            path = Path(entry.filename)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe ZIP path")
            require(path.parts and path.parts[0] in (package, "__MACOSX"), "unexpected ZIP root")
            if stat.S_ISLNK(entry.external_attr >> 16):
                target = archive.read(entry).decode("utf-8")
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(entry.filename), target))
                require(not target.startswith("/") and resolved.startswith(package + "/"),
                        "escaping ZIP symlink")
    with tempfile.TemporaryDirectory(prefix="ironmlx-archive-verify-") as temporary:
        temp = Path(temporary)
        unpack = temp / "zip"
        unpack.mkdir()
        run("ditto", "-x", "-k", zip_path, unpack)
        check_root(unpack / package)
        mount = temp / "dmg"
        mount.mkdir()
        run("hdiutil", "attach", output / f"{package}.dmg", "-readonly", "-nobrowse",
            "-mountpoint", mount, "-quiet")
        try:
            check_root(mount)
        finally:
            run("hdiutil", "detach", mount, "-quiet")


def assemble(repo, app, output, package):
    require(not output.exists() or not any(output.iterdir()), "output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for name in MATERIALS:
        shutil.copy2(material_source(repo, name), output / name)
    shutil.copytree(repo / "THIRD_PARTY_LICENSES", output / "THIRD_PARTY_LICENSES")
    with tempfile.TemporaryDirectory(prefix="ironmlx-archive-stage-") as temporary:
        root = Path(temporary) / package
        root.mkdir()
        run("ditto", app, root / "IronMLX.app")
        for name in (*MATERIALS, "THIRD_PARTY_LICENSES"):
            run("ditto", output / name, root / name)
        run("ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", root, output / f"{package}.zip")
        run("hdiutil", "create", "-volname", package, "-srcfolder", root,
            "-format", "UDZO", output / f"{package}.dmg")
    (output / "SHA256SUMS").write_text(checksums(repo, output, package))
    verify(repo, app, output, package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("assemble", "verify"))
    parser.add_argument("app", type=Path, help="reference App already checked by the caller")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    app, output = args.app.resolve(), args.output.resolve()
    require(not output.is_relative_to(app) and not app.is_relative_to(output), "App and output must not overlap")
    version = (repo / "VERSION").read_text().strip()
    with (app / "Contents/Info.plist").open("rb") as handle:
        info = plistlib.load(handle)
    require(info.get("CFBundleShortVersionString") == version, "reference App version mismatch")
    require(info.get("CFBundleIdentifier") == "com.ironmlx.app", "reference App identifier mismatch")
    run(repo / "scripts/verify-distribution-materials.sh")
    operation = assemble if args.action == "assemble" else verify
    operation(repo, app, output, f"IronMLX-{version}")
    print(f"Archive {args.action} passed (content only, not distribution approval): {output}")


if __name__ == "__main__":
    main()
