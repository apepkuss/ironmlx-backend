# Contributing to IronMLX

[简体中文](docs/zh-CN/contributing.md)

## Before contributing

Read the [installation and build guide](docs/installation.md),
[supported-model matrix](docs/supported-models.md), and
[security boundary](docs/security-boundary.md). Do not include model weights,
tokens, prompts, private certificates, or unredacted diagnostic data in a
patch or issue.

## License and contribution terms

IronMLX original source is licensed under the Apache License, Version 2.0. By
submitting a contribution, you certify that you have the right to submit it
and agree that the contribution may be distributed under Apache-2.0 with the
project. Preserve applicable copyright and license notices. Third-party
dependencies, bundled assets, and model weights retain their own terms; this
project does not relicense model weights.

## Developer Certificate of Origin

Contributions use the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).
Sign each commit with your name and email using:

```text
git commit -s
```

The resulting `Signed-off-by:` line certifies the DCO statement. If an
employer or another party owns rights in the contribution, obtain the required
permission before submitting it.

## Changes and verification

Keep changes focused, explain behavior changes, and update the English and
Simplified Chinese user documentation when public behavior changes. Run the
relevant Rust, Swift, App Bundle, SBOM, license-policy, dependency-audit,
secret-scan, and model-boundary checks described in the README before opening a
pull request. Never weaken a release or security gate to make a check pass.

For vulnerability reports, use [SECURITY.md](SECURITY.md) instead of a public
issue.
