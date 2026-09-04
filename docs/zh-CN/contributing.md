# 参与 IronMLX 开发

[English](../../CONTRIBUTING.md)

## 开始前

请先阅读[安装与构建](installation.md)、[支持模型矩阵](supported-models.md)和[安全边界](security-boundary.md)。不要在代码、issue 或诊断数据中包含模型权重、token、prompt、私有证书或未脱敏信息。

## 许可证与贡献条款

IronMLX 原创源代码采用 Apache License 2.0。提交贡献即表示你确认有权提交，并同意该贡献可随项目按 Apache-2.0 分发。应保留适用的版权和许可证声明。第三方依赖、随 App 打包的资产和模型权重仍受各自条款约束；本项目不重新授权模型权重。

## Developer Certificate of Origin

贡献采用 [Developer Certificate of Origin（DCO）](https://developercertificate.org/)。提交时使用：

```text
git commit -s
```

生成的 `Signed-off-by:` 行表示你确认 DCO 声明。如果贡献的权利归雇主或其他主体所有，请在提交前取得必要授权。

## 修改与验证

保持改动聚焦，说明行为变化；公开行为变化时同步更新英文和简体中文文档。创建 Pull Request 前，运行 README 中适用的 Rust、Swift、App Bundle、SBOM、许可证策略、依赖审计、secret 扫描和模型边界检查。不要为了通过检查而削弱发布或安全门禁。

安全漏洞请使用 [SECURITY.md](../../SECURITY.md) 私密报告，不要创建公开 issue。
