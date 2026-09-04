# IronMLX

[简体中文](README.zh-CN.md)

IronMLX is a local large-language-model inference App and service runtime for
Apple Silicon. It packages a Rust inference engine, MLX/Metal runtime, model
management Dashboard, and OpenAI/Anthropic-compatible HTTP APIs into a
self-contained macOS App.

Current product version: **0.1.0**

## Requirements

- Apple Silicon (`arm64`); Intel Macs are not supported;
- macOS 26.2 or later;
- Source builds require full Xcode, Rust 1.94, and the pinned MLX source.

## Capabilities

- Local model search, immutable-snapshot downloads, resume, and integrity checks;
- Multi-model loading, unloading, pinning, TTL, and memory protection;
- OpenAI Chat Completions/Responses and Anthropic Messages APIs with client-side
  function-call protocols;
- Streaming, continuous batching, paged KV/prefix cache, MTP, and Prompt Lookup;
- Qwen3.8 reasoning/tools, matching MTP, and isolated DFlash2 text execution;
- Text and controlled base64 image input;
- Local redacted diagnostic export with no prompt, credential, or network upload;
- Loopback by default, with optional LAN mode using HTTPS and API keys.

## Quick start

Third-party inventories, notices, license texts, and the deterministic CycloneDX
SBOM are generated for engineering review, but public binary distribution remains
blocked until P0-8B legal review and explicit authorization are complete. Build
from a trusted checkout:

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about
scripts/checkout-release-mlx.sh /tmp/ironmlx-mlx-source
MLX_SRC=/tmp/ironmlx-mlx-source scripts/build-app-bundle.sh
open dist/IronMLX.app
```

See [Installation and build](docs/installation.md) for details, or read the
[Simplified Chinese translation](docs/zh-CN/installation.md).

## Documentation

- [Supported model matrix](docs/supported-models.md) · [简体中文](docs/zh-CN/supported-models.md)
- [Model rights boundary](docs/model-license-boundary.md) · [简体中文](docs/zh-CN/model-license-boundary.md)
- [HTTP API](docs/api.md) · [简体中文](docs/zh-CN/api.md)
- [API compatibility matrix](docs/api-compatibility-matrix.md)
- [DFlash2 server and CLI](docs/dflash2-server-api.md)
- [Hermes Agent integration](docs/hermes-agent.md) · [简体中文](docs/zh-CN/hermes-agent.md)
- [oh-my-pi integration](docs/oh-my-pi.md) · [简体中文](docs/zh-CN/oh-my-pi.md)
- [Privacy and network boundary](docs/privacy.md) · [简体中文](docs/zh-CN/privacy.md)
- [Data locations and uninstall](docs/storage-and-uninstall.md) · [简体中文](docs/zh-CN/storage-and-uninstall.md)
- [Troubleshooting](docs/troubleshooting.md) · [简体中文](docs/zh-CN/troubleshooting.md)
- [Known issues](docs/known-issues.md) · [简体中文](docs/zh-CN/known-issues.md)
- [0.1.0 release notes](docs/release-notes/0.1.0.md) · [简体中文](docs/zh-CN/release-notes/0.1.0.md)
- [Versioning and releases](docs/versioning-and-releases.md) · [简体中文](docs/zh-CN/versioning-and-releases.md)
- [Third-party materials](docs/third-party-materials.md) · [简体中文](docs/zh-CN/third-party-materials.md)
- [Security boundary](docs/security-boundary.md) · [简体中文](docs/zh-CN/security-boundary.md)
- [Development previews](docs/development-preview-release.md) · [简体中文](docs/zh-CN/development-preview-release.md)
- [Diagnostic export](docs/diagnostic-bundle.md) · [简体中文](docs/zh-CN/diagnostic-bundle.md)
- [Security reporting](SECURITY.md) · [中文](docs/zh-CN/security.md)
- [Support](SUPPORT.md) · [中文](docs/zh-CN/support.md)
- [Contributing](CONTRIBUTING.md) · [中文](docs/zh-CN/contributing.md)

## Development verification

```bash
scripts/verify-version-consistency.sh
scripts/verify-sbom.sh
scripts/verify-third-party-materials.sh
python3 scripts/verify-license-policy.py
cargo audit
GITLEAKS_BIN=/path/to/gitleaks-8.30.1 scripts/verify-secrets.sh
cargo fmt --all -- --check
cargo +nightly fmt --all -- --check
cargo +nightly clippy --locked --all-features --workspace -- -D warnings
cargo build --locked --release
cargo test --locked --all-features --workspace -- --test-threads=1
swift test --package-path ironmlx-app --configuration release --no-parallel
```

For a built App Bundle:

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
scripts/verify-model-distribution-boundary.sh dist/IronMLX.app
```

## License

IronMLX original source code is licensed under the Apache License, Version 2.0;
see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party dependencies and
bundled assets remain under their respective licenses, as listed in
`THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/`. Model weights are not
licensed or redistributed by IronMLX; users are responsible for the terms of
the upstream model repository.
