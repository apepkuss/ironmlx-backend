# 版本与发布流程

## 单一产品版本

仓库根目录 `VERSION` 是产品版本的规范输入。Rust workspace、CLI、`healthz`、
App `CFBundleShortVersionString` 与发布 tag 必须保持一致。`CFBundleVersion` 是单调
递增的正整数构建号。

不要手动逐文件改版本。发布或跳版时运行：

```bash
scripts/bump-version.sh 0.2.0
```

脚本会更新 `VERSION`、workspace package version、内部显式依赖版本、Cargo.lock
与 App plist；版本变化时默认把 App build number 加一。需要指定构建号时：

```bash
scripts/bump-version.sh 0.2.0 7
```

完成后必须提交脚本生成的全部改动，并运行：

```bash
scripts/verify-version-consistency.sh
```

CI 会执行同一检查，并验证每个 workspace crate 都声明 `publish = false`。因此
IronMLX 不能通过 `cargo publish` 意外发布到 crates.io。

## Tag 与发布说明

未来 stable tag 使用 `vX.Y.Z`，并必须与 `VERSION` 一致。DMG、App About、CLI
`--version`、`healthz.version`、release tag 和 release notes 应引用同一产品版本。

当前 development preview 使用独立的 `preview-YYYYMMDD-shortSHA` 命名空间，
不会冒充 stable semantic-version tag。

## 当前发布硬门禁

`scripts/release-legal-gate.sh` 在打包和 GitHub preview workflow 中执行。当前
`IRONMLX_PUBLIC_DISTRIBUTION_READY=false`，因此 public binary 分发必然失败。

P0-8B 完成后，只有同时满足以下条件才能由单独评审显式开启：

- `THIRD_PARTY_NOTICES.md` 存在且非空；
- `third-party-inventory.json` 存在且非空；
- `THIRD_PARTY_LICENSES/` 至少包含一份非空第三方许可证文本；
- `SBOM.cdx.json` 存在且非空；
- 材料已按最终闭源 App 的依赖与分发方式完成法律/合规复核；
- `scripts/release-config.sh` 中的 distribution-ready 标志经授权改为 `true`。

项目 `LICENSE`、`NOTICE` 与确定性生成的 `SBOM.cdx.json` 也必须进入发布材料。门禁不要求或暗示采用任何第一方开源许可证；第一方授权与版权策略由未来发布
决策单独确定。

第三方材料由 P0-8A 的锁定工程流程生成，更新与验证方式见
[第三方依赖与许可证材料](third-party-materials.md)。这些材料存在并不等于完成
法律判断，也不会自动解除 public distribution 门禁。

## 正式发布产物身份

正式打包前，使用明确的现有发布 tag 检查源码及 App：

```bash
python3 scripts/verify-release-identity.py v0.1.0
python3 scripts/verify-release-identity.py v0.1.0 dist/IronMLX.app
```

源码检查要求 `refs/tags/` 下的 tag 指向 HEAD、与 `VERSION` 一致，且工作区
clean，包括未忽略的未跟踪文件。支持 lightweight 和 annotated tag。提供 App
参数时，还要求产品版本、build number、来源提交一致，来源状态为 `clean`。
正式打包脚本始终执行两项检查；第三个参数为发布 tag，默认是 `v` 加 `VERSION`。
自动和手动触发的正式工作流均显式传入选定 tag。

此门禁检查身份元数据，不证明密码学构建来源，也不替代签名和公证检查。
本地开发构建及其静态 Bundle 检查仍允许 dirty 源码。

RC 身份验收使用独立的显式模式：

```bash
python3 scripts/verify-release-identity.py --candidate v0.1.0-rc.1 dist/IronMLX.app
```

候选模式只接受 `vX.Y.Z-rc.N`，N 必须为无前导零的正整数。基础版本 `X.Y.Z`
必须与 App 版本和 `VERSION` 一致，仍执行 clean、tag/HEAD、build number 和
Bundle 来源检查。正式打包和发布不启用此模式，继续拒绝 RC tag。

## RC 打包与发布

`Release Candidate` 工作流接受现有 `vX.Y.Z-rc.N` tag，可通过推送 tag 或手动填写
`release_tag` 触发。手动触发要求工作流已进入默认分支；选定 tag 必须包含 RC
脚本。不会移动或重建 tag，也不覆盖已有 Release。发布改动提交后应创建新的 RC
tag，不能移动早先的 RC tag。

对同一 clean tag 构建的 `dist/IronMLX.app` 执行本地归档验证：

```bash
scripts/package-release-candidate.sh v0.1.0-rc.2 validate
```

产物复用预览归档引擎，位于 `.build/development-preview-release/assets`。
DMG/ZIP 文件名含 RC tag 和 `ADHOC-NOT-NOTARIZED`，App 名称为
`IronMLX Release Candidate.app`，分发通道为 `release-candidate`。沿用的
`DEVELOPMENT-PREVIEW-NOTICE.txt` 与 `PREVIEW-BUILD-METADATA.json` 文件名属于
共享归档格式，其内容明确记录 RC tag 和通道。打包前及两种归档解开后均核对身份。

工作流构建固定 MLX 和 Release App，校验归档及 SHA-256。手动触发默认仅验证，
推送 RC tag 也仅验证。只有手动选择 `publish=true`，才进入独立的写权限任务，
强制核对分发授权后创建 GitHub **Prerelease**，不标为 Latest。

验证模式始终检查材料完整性和 SBOM 一致性；分发未授权时不上传安装包 artifact，
只在任务摘要中记录验证结果。授权后才可保留可下载的 Actions artifact。
本地打包支持 `validate`（默认）与 `publish`；后者在打包前强制检查分发授权。
本通道不执行 Developer ID
签名或公证，也不证明正常 Gatekeeper 安装或生产自动更新已经通过。正式发布通道
排除候选 tag。

## 正式归档布局与独立内容验证

正式输出目录统一包含 `IronMLX-X.Y.Z.dmg`、`IronMLX-X.Y.Z.zip`、`SHA256SUMS`、
独立法律材料和 `THIRD_PARTY_LICENSES/`。ZIP 内以 `IronMLX-X.Y.Z/` 为根目录，
DMG 卷根目录放置相同内容：`IronMLX.app` 和所有法律材料。输出目录必须不存在或
为空；脚本不自动删除旧产物。

无需 release tag 或 Developer ID 即可验证归档机制：

```bash
python3 scripts/release-archives.py assemble dist/IronMLX.app .build/archive-check
python3 scripts/release-archives.py verify dist/IronMLX.app .build/archive-check
```

检查涵盖当前材料/SBOM、产品版本和标识、完整校验和清单、实际 ZIP 解压与 DMG
只读挂载，以及 App 每个文件、执行位和符号链接与参考 Bundle 的一致性，包括
版本/来源元数据及内嵌法律材料。这不证明参考 Bundle 来自当前 clean 提交或已签名。
正式打包入口仍先执行身份、clean、分发授权、静态 Bundle、签名与 Gatekeeper 门禁。
仅内容验证产生的文件不能作为已批准的正式版发布。

公网 Sparkle 通道配置及 feed 发布流程见[自动更新](automatic-updates.md)。
