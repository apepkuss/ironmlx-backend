# 自动更新

使用固定的 Sparkle 2.9.6。`stable` 与 `release-candidate` 使用独立的签名 HTTPS
更新源；`development` 仅接受 loopback HTTPS，本地普通构建默认关闭更新。

## 版本与通道

- App 产品版本保持 `VERSION` 的 `X.Y.Z`，RC 后缀放在 tag 和 feed 显示版本中。
- Sparkle 按正整数 `CFBundleVersion` 判断新旧，每次发布更新都必须递增，包括
  同一产品版本的多个 RC。可用 `scripts/bump-version.sh X.Y.Z N` 显式指定更高 N。
- 更新源为 `https://raw.githubusercontent.com/<owner>/<repo>/updates/stable.xml`
  和 `https://raw.githubusercontent.com/<owner>/<repo>/updates/release-candidate.xml`。
  发布脚本维护独立 `updates` 分支，不修改源码分支或 release tag。
- RC 条目额外带有 Sparkle `release-candidate` 标记，正式客户端不订阅该标记。
  RC 转正式版需要主动安装/切换，不通过 RC feed 自动转入正式通道。

## 长期签名配置

更新使用的 Ed25519 key 与 Apple Developer ID 无关。现有工具可生成所需的
32 字节 seed；工具名称中的 development 不限制密钥用途：

```bash
swift scripts/generate-development-update-key.swift /absolute/private/path/update-key
```

目标文件须不存在，生成后权限为 0600，标准输出只有公钥。将公钥配置为 GitHub
仓库 variable `IRONMLX_UPDATE_PUBLIC_ED_KEY`，私钥文件内容配置为 secret
`IRONMLX_UPDATE_PRIVATE_ED_KEY`。私钥保存在仓库外；发布后换 key 需要单独迁移方案。
脚本会检查私钥派生的公钥与 App 内嵌公钥一致。源码测试只生成临时 key，不创建
长期密钥或修改 GitHub 设置。

正式 App 必须在构建、签名前配置 `IRONMLX_UPDATE_CHANNEL=stable`、对应
`IRONMLX_UPDATE_FEED_URL` 和 `IRONMLX_UPDATE_PUBLIC_ED_KEY`。RC 工作流自动填写
RC 通道和 URL。仅验证模式缺少长期公钥时使用临时 key，并禁止上传安装包，即使
其他分发材料已获授权；实际发布必须同时具备长期公钥和私钥 secret。

## 发布顺序

先通过身份、归档及分发门禁，再从实际 App 生成独立更新 ZIP，签署 ZIP 与 XML，
验证签名并记录 `update.json`。更新 ZIP、XML 和元数据随 Release 上传后，脚本
下载公开 ZIP 核对 SHA-256、tag 来源和 Release 状态，最后才更新 feed。
用户手动安装用的 ZIP 不直接充当更新 ZIP。当前不生成差分更新。

Feed 拒绝 build number 倒退或同编号替换；相同元数据可安全重试。发布分支使用
非强制更新，发生并发冲突时失败而不覆盖另一通道。若 Release 已创建而 feed
更新失败，复用已发布的 ZIP/XML/`update.json` 重试 `publish-update-feed.py`，
不要重新生成或覆盖 Release。首次成功发布之前 feed 可以不存在，此时更新失败
不得影响已安装 App。

RC 明确选择发布时执行上述流程。正式版工作流也已接入更新产物和 feed 步骤，
但仍依赖后续正式构建/签名/公证流水线提供合格 Bundle；本项不取消这些门禁。

## 验证边界

`test_app_updates.py` 覆盖通道/版本规则、递增编号、真实 Sparkle 签名、错误 key
及篡改拒绝。`validate-update-installation.py` 将生产更新管理器编译到隔离测试 App，
执行本机 HTTPS 升级与重启，不启动 IronMLX 模型或读取其配置。测试会临时在登录
钥匙串信任 localhost TLS 证书，结束后移除。这不等于生产模型恢复、M1 Pro 或
公证包升级已经验收。

完整 App 现有退出流程会取消下载，并等待 `backend.stopForAppQuit()` 后退出。
最终仍需在两台目标机器上验证生产 helper、活动请求和真实配置/模型保留。
