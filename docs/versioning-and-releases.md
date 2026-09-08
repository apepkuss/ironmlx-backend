# Versioning and release process

## One product version

The repository-root `VERSION` file is canonical. Rust workspace packages, CLI,
`healthz`, App `CFBundleShortVersionString`, and release tags must agree.
`CFBundleVersion` is a monotonically increasing positive integer.

Do not edit versions file by file. Run:

```bash
scripts/bump-version.sh 0.2.0
```

The script updates `VERSION`, workspace package versions, internal explicit
dependencies, `Cargo.lock`, and the App plist. It increments the App build
number by default; pass an explicit number when needed:

```bash
scripts/bump-version.sh 0.2.0 7
```

Commit all generated changes and run:

```bash
scripts/verify-version-consistency.sh
```

CI also verifies that every workspace crate declares `publish = false`, so
IronMLX cannot accidentally publish to crates.io.

## Tags and release notes

Stable tags use `vX.Y.Z` and must match `VERSION`. DMG, App About, CLI
`--version`, `healthz.version`, the release tag, and release notes must use the
same product version. Development previews use the separate
`preview-YYYYMMDD-shortSHA` namespace.

## Current hard gate

`release-legal-gate.sh` runs during packaging and in the GitHub preview workflow.
`IRONMLX_PUBLIC_DISTRIBUTION_READY=false` currently makes public binary release
fail. After P0-8B, an authorized reviewer may enable it only when notices,
inventory, license texts, SBOM, and final legal review are complete.

The gate requires the project `LICENSE`, `NOTICE`, and deterministic
`SBOM.cdx.json` to be present in the release materials. It does not require or
imply a particular first-party open-source license;
that policy is a separate release decision. See [Third-party materials](third-party-materials.md)
for the locked inventory process.

## Stable release identity

Before packaging, run the identity gate with the exact existing release tag:

```bash
python3 scripts/verify-release-identity.py v0.1.0
python3 scripts/verify-release-identity.py v0.1.0 dist/IronMLX.app
```

The source check requires the tag under `refs/tags/` to resolve to HEAD, match
`VERSION`, and use a clean checkout, including non-ignored untracked files.
Lightweight and annotated tags are supported. The optional App check additionally
requires matching product version, build number, source commit, and a `clean`
source-tree marker. The stable packager always performs both checks; its third
argument is the release tag (default: `v` plus `VERSION`). Both automatic and
manual stable workflows pass the selected tag explicitly.

This gate validates identity metadata, not cryptographic build provenance or
signing/notarization. Those remain separate release gates. Local development
builds and their static Bundle checks continue to allow dirty source trees.

RC identity validation is explicitly separate from stable packaging:

```bash
python3 scripts/verify-release-identity.py --candidate v0.1.0-rc.1 dist/IronMLX.app
```

Candidate mode accepts only `vX.Y.Z-rc.N` with a positive, non-zero-prefixed N.
It compares the base `X.Y.Z` to the App version and `VERSION`, retaining all
clean-checkout, tag/HEAD, build-number and Bundle-source checks. Stable packaging
and publication never enable this mode and continue to reject RC tags.

## RC packaging and publication

The `Release Candidate` workflow handles existing `vX.Y.Z-rc.N` tags, via tag
push or manual dispatch with `release_tag`. The workflow must be available on
the default branch for manual dispatch, and the selected tag must contain the
RC scripts. Existing tags are never moved or recreated; an existing Release is
not overwritten. After committing release changes, use a new RC tag on that
commit rather than moving an earlier RC tag.

Build a clean candidate and validate its archives locally (no publication authorization required):

```bash
scripts/package-release-candidate.sh v0.1.0-rc.2 validate
```

The command consumes `dist/IronMLX.app` from that exact clean tagged commit.
It reuses the development archive engine and writes verified assets under
`.build/development-preview-release/assets`. Both DMG and ZIP carry the RC tag
and `ADHOC-NOT-NOTARIZED` suffix; the App is named `IronMLX Release Candidate.app`
and declares `release-candidate`. The existing `DEVELOPMENT-PREVIEW-NOTICE.txt`
and `PREVIEW-BUILD-METADATA.json` filenames are shared with the preview engine;
their contents identify the RC tag and channel. Source identity is checked
before packaging and in both extracted archives.

The workflow builds the pinned MLX and Release App, verifies archives and
checksums, and defaults to validation only. Tag pushes also validate only. Manual dispatch
with `publish=true` enables a separate write-permission job that creates a GitHub
**Prerelease**, never Latest, after enforcing the distribution authorization gate.
Validation always checks material completeness and SBOM consistency. Installers
are uploaded as Actions artifacts only when distribution is authorized; otherwise
only the validation result is retained in the job summary. Local packaging accepts
`validate` (default) or `publish`; the latter enforces authorization before packaging.
Neither mode signs with Developer ID or notarizes. Normal Gatekeeper
installation acceptance and production automatic updates are not established
by this channel. Stable publication excludes prerelease tags.
