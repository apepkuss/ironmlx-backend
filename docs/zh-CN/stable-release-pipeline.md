# 稳定版发布流水线

`Stable Release` 从现有 `vX.Y.Z` tag 的干净完整 checkout 构建，校验 tag、
`VERSION`、App 版本与 build。发布 job 检出已验证的精确提交，下载候选 ZIP 后再次
校验 tag 和 Bundle 身份。MLX 使用 `release-config.sh` 固定提交，构建安装
Rust 1.94.0 与 cargo-about 0.9.1。

## 无 Apple 凭据的验证

tag push 和手动 `publish=false` 只构建、验证，不读取签名 Secret，不创建 Release
或更新 feed。使用仓库变量 `IRONMLX_UPDATE_PUBLIC_ED_KEY` 配置 stable HTTPS 通道，
构建完整 Release App，生成 ZIP/DMG，并解压、挂载检查内容。此模式仍要求公钥已配置，
不代表 Developer ID、公证或 Gatekeeper 验收通过。

## 正式发布配置

手动 `publish=true` 才进入发布流程。候选 App 用 ZIP 传递，保留权限、框架软链接和
签名；传递前必须通过现有 `release-legal-gate.sh`。本次实现不修改公开分发授权开关。

发布 job 使用 GitHub `stable-release` environment，生产使用前应配置审核人及允许
部署的 refs。该 environment 需要以下配置；Sparkle 私钥也可沿用仓库 Secret。

| 类型 | 名称 | 内容 |
|---|---|---|
| Secret | `IRONMLX_DEVELOPER_ID_P12_BASE64` | 含 Developer ID Application 证书及私钥的 PKCS#12，Base64 编码 |
| Secret | `IRONMLX_DEVELOPER_ID_P12_PASSWORD` | 非空 PKCS#12 导出密码 |
| Variable | `IRONMLX_SIGNING_IDENTITY` | 完整 `Developer ID Application: Name (TEAMID)` |
| Variable | `IRONMLX_APPLE_TEAM_ID` | 证书 Team ID |
| Secret | `IRONMLX_NOTARY_KEY_ID` | App Store Connect 团队 API Key ID |
| Secret | `IRONMLX_NOTARY_ISSUER_ID` | 团队 API Issuer UUID |
| Secret | `IRONMLX_NOTARY_PRIVATE_KEY` | API Key 的 `.p8` 内容 |
| Secret | `IRONMLX_UPDATE_PRIVATE_ED_KEY` | 与构建公钥匹配的 Sparkle Ed25519 seed |

公钥变量必须保留在仓库级，因为构建 job 不使用生产 environment。本次只实现流程，
不生成或配置 Apple 凭据。

## 执行顺序与失败处理

1. 在临时 keychain 导入证书及公证凭据；签名前确定全部 Bundle 元数据。
2. 从内到外签署 Sparkle、Rust helpers 和 App，启用 hardened runtime、可信时间戳。
   Downloader 使用明确的 sandbox/network entitlements，不添加 JIT 或禁用库验证例外。
3. 提交 App ZIP 公证，必须收到 `Accepted`，再 staple、校验票据及 Gatekeeper。
   plist 中的 `stapled` 是签名前封存的声明，不能替代实际票据验证。
4. 从已 staple 的 App 打包 ZIP/DMG；对最终 DMG 签名、公证、staple，再更新
   `SHA256SUMS` 并复验归档内容。
5. 签署更新 ZIP/XML；生成覆盖全部上传文件的 `RELEASE-SHA256SUMS`。
6. 创建草稿 Release，上传后核对完整资产集合，逐项下载并比对哈希；再次校验远端
   tag，才将草稿公开。公开后再次下载复验，再发布更新 feed。

构建、签名、公证失败时停止；临时证书、私钥、keychain 在退出时清理，工作流还有
`always()` 清理。公证结果保存在 `.build/notarization`，不作为公开发布资产。
上传或草稿校验失败时保留草稿供检查，不自动公开。普通重跑拒绝已存在的 Release，
不会覆盖资产或自动删除草稿，需检查失败原因后显式处理。

若 Release 已公开但 feed 失败，使用其原始更新 ZIP/XML/`update.json` 重试
`publish-update-feed.py`，不重新构建或覆盖 Release；新 feed 成功前旧 feed 保持可用。

本地测试模拟 Apple/GitHub 操作，只证明执行顺序、拒绝路径与清理。真实公证、
Gatekeeper、公网下载及完整 App 升级验收，需要凭据与分发授权就绪后执行。

参考：[Apple 公证流程](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)、
[Sparkle 手动签名](https://sparkle-project.org/documentation/sandboxing/)。
