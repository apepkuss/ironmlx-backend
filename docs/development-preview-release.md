# IronMLX GitHub Actions development previews

[简体中文](zh-CN/development-preview-release.md)

The development-preview workflow is retained as a future distribution path,
but public preview packaging and publication are currently disabled. Every
future preview will be built from the immutable commit selected by manually
dispatching the workflow on the remote `dev` branch.

> **未使用 Developer ID 签名、未经 Apple 公证，仅供开发验证。**

The preview workflow must never be represented as a stable release. Developer
ID signing, Apple notarization, stapling, formal target-machine acceptance, and
stable GitHub Releases remain disabled until the required Apple credentials and
real acceptance environment are available.

## Pull request CI

`.github/workflows/ci.yml` runs automatically for pull requests targeting
`main` or `dev`, pushes to `main` or `dev`, and pushes to the approved typed
branch prefixes (`feat/**`, `fix/**`, `test/**`, `refactor/**`, `perf/**`,
`build/**`, `ci/**`, `style/**`, `chore/**`, and `revert/**`). `docs/**` pushes
are intentionally excluded from the expensive full build. It uses GitHub-hosted
`macos-26` and has only read access to repository contents. It does not reference
a GitHub Environment or any secret.

Push and pull-request runs from the same repository and head branch share one
concurrency group, so a newer run cancels the duplicate older run. Before any
expensive build, CI validates every commit introduced by the event against the
Conventional Commits subject shape.

The job:

1. checks out the immutable IronMLX event commit;
2. validates the event's commit range as Conventional Commits;
3. installs the pinned `cargo-about 0.9.1` inventory tool and creates a clean
   detached checkout of the MLX commit declared by
   `scripts/release-config.sh`;
4. audits the locked Rust dependency graph with the pinned `cargo-audit 0.22.2`,
   enforces the repository license policy, and scans the tracked source tree
   with the pinned `gitleaks 8.30.1` binary;
5. verifies the tracked dependency inventory and license texts against the
   actual native and Rust Release inputs, then builds the self-contained App;
6. runs stable and pinned-nightly Rust formatting checks, all-feature workspace
   Clippy with warnings denied, a locked Release build, and the serial full
   workspace Rust test suite;
7. runs Swift tests in Release mode;
8. verifies the App, helpers, and Metal library as arm64 with `minos=26.2`,
   system-only dynamic dependencies, and no embedded developer paths.

## Current legal-material gate

`scripts/release-legal-gate.sh` runs before packaging. It fails while
`IRONMLX_PUBLIC_DISTRIBUTION_READY` is false. P0-8A now supplies reproducible
third-party Notices, license texts, and an engineering inventory; P0-8B must
review those materials, approve the generated CycloneDX SBOM, and explicitly
authorize distribution before the flag can be enabled.

## Manual preview publication after the gate is approved

After that separate approval, run `Development Preview Release` from the
GitHub Actions UI:

1. select the `dev` branch;
2. acknowledge that the result is ad-hoc signed and not notarized;
3. dispatch the workflow.

The event's `GITHUB_SHA` is frozen as the source commit. The workflow checks out
that exact SHA and builds MLX from its separate pinned SHA. It creates a tag in
the following namespace:

```text
preview-YYYYMMDD-abcdef0
```

The date is evaluated in `Asia/Tokyo`. A second publication for the same commit
on the same date fails instead of moving or overwriting the existing preview
tag. Preview tags never use a semantic-version namespace reserved for a future
stable release.

The GitHub Release is always a prerelease. Its title and notes contain the
required warning. The downloadable files include the product version and use
the `ADHOC-NOT-NOTARIZED` suffix, and both archives contain:

- `IronMLX Development Preview.app`;
- `DEVELOPMENT-PREVIEW-NOTICE.txt`;
- `PREVIEW-BUILD-METADATA.json`;
- `LICENSE`, `NOTICE`, `SBOM.cdx.json`, `THIRD_PARTY_NOTICES.md`,
  `third-party-inventory.json`, and `THIRD_PARTY_LICENSES/`;
- `model-license-boundary.md`.

The App's `Info.plist`, bundled notice, and metadata repeat that Developer ID
signing and Apple notarization are disabled. Metadata identifies both the
non-official MLX fork and its upstream base revision. `SHA256SUMS` covers the
DMG, ZIP, notices, inventory, model-rights boundary, every license file, metadata,
and release notes. The packaging and verification scripts also reject common
model-weight files in the App, ZIP, and DMG.

## Local packaging check

After the legal-material gate is approved and `dist/IronMLX.app` is built,
package a preview with:

```bash
scripts/package-development-preview.sh \
  preview-20260802-a166fd1 \
  a166fd1127b84d44249b72881202f8863de966e3
```

The packager invokes `scripts/verify-development-preview.sh`, extracts the ZIP,
mounts the DMG, re-runs the static App Bundle gate on both copies, verifies the
ad-hoc identity and absent Team ID, and validates all SHA-256 entries.

## Deferred stable-release gates

The following are intentionally not implemented or enabled by the preview
workflow:

- Developer ID Application signing and protected signing secrets;
- hardened-runtime production signing;
- Apple `notarytool` submission and stapling;
- Gatekeeper acceptance of a notarized artifact;
- real MLX/Metal inference on the minimum supported target machine;
- stable semantic-version tags and stable GitHub Releases.

These gates require a separate design and authorization after Developer ID and
notarization credentials become available. Fixture or GitHub-hosted runner
checks must not be described as formal release acceptance.

`ironmlx-model-migrate` is not bundled as an App helper and is outside both the
development-preview and future release acceptance boundary. Whether its source
is retained is outside this workflow task.
