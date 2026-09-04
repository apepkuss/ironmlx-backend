# IronMLX GitHub Actions 开发预览

[English](../development-preview-release.md)

开发预览 workflow 保留为未来分发路径，但当前公开预览打包和发布仍已禁用。未来
预览必须通过远程 `dev` 分支手动 dispatch，并从选定的不可变 commit 构建。

> **未使用 Developer ID 签名、未经 Apple 公证，仅供开发验证。**

预览绝不能被描述为稳定版本。Developer ID 签名、公证、stapling、正式目标机验收
和 stable GitHub Release，须等 Apple 凭据和真实验收环境可用后再启用。

## Pull request CI

`.github/workflows/ci.yml` 会对目标为 `main`/`dev` 的 PR、推送到这两个分支以及批准
的功能分支前缀自动运行。CI 使用 `macos-26` runner，仅有仓库读取权限，不引用
GitHub Environment 或 secret，并在昂贵构建前校验 Conventional Commits。

任务会检出事件 commit，安装锁定的 `cargo-about`，检出发布配置指定的 MLX commit，
使用锁定的 `cargo-audit 0.22.2` 扫描 Rust 依赖、执行许可证策略检查，并使用锁定的
`gitleaks 8.30.1` 扫描已跟踪源代码；随后验证第三方材料，构建自包含 App，运行 Rust
格式/Clippy/Release、全 workspace 串行测试和 Swift Release 测试，并验证 App、helper
与 Metal library 的 arm64、macOS 26.2 和无开发路径属性。

## 当前法律材料门禁

打包前执行 `scripts/release-legal-gate.sh`。当
`IRONMLX_PUBLIC_DISTRIBUTION_READY=false` 时会失败。P0-8B 必须完成第三方材料审查、
复核第三方材料、批准已生成的 CycloneDX SBOM 并取得明确分发授权后，才能启用该标志。

## 手动发布预览

门禁获批后，在 GitHub Actions 中选择 `dev` 分支并 dispatch `Development Preview Release`。
workflow 冻结 `GITHUB_SHA`，从独立的 MLX pinned SHA 构建，并创建
`preview-YYYYMMDD-abcdef0` 命名空间中的 tag。GitHub Release 始终标记为 prerelease。

DMG 和 ZIP 包含 App、预览警告、构建元数据、项目 LICENSE/NOTICE/SBOM、第三方
notices/inventory/licenses，以及
`model-license-boundary.md`。`SHA256SUMS` 覆盖这些材料、归档和 release notes；验证器
会拒绝 App、ZIP 或 DMG 中的常见模型权重文件。

## 本地检查

```bash
scripts/package-development-preview.sh \
  preview-20260802-a166fd1 \
  a166fd1127b84d44249b72881202f8863de966e3
```

打包器会解包 ZIP、挂载 DMG，在两份 App 上重新运行静态门禁，验证 ad-hoc 身份、无
Team ID、所有 SHA-256，以及模型分发边界。

## 延后的稳定发布门禁

Developer ID Application 签名、Hardened Runtime、公证/stapling、Gatekeeper 验收、
最低目标机真实推理和 stable semantic-version tag 均需在取得 Apple 凭据后单独设计和
授权。fixture 或 GitHub runner 检查不能被描述为正式发布验收。
