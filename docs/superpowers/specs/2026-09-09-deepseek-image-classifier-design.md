# DeepSeek图片分类工具 — 独立新项目设计文档

- 日期：2026-09-09
- 状态：已确认
- 项目：`C:\Users\HP\projects\deepseek-image-classifier`
- 前身：`image-classifier-gui` v2（main@4926285，保持原样不动）

## 1. 目标

保留原工具（Kimi/DeepSeek 双服务商版）不变，另建一个**独立新工具**：只提供
DeepSeek 视觉模型（固定内置模型名），其余功能（DPAPI 加密、取消修复、判重、
预览清理、分类/关键词/提示词/RPM）与原版一致。

## 2. 项目身份

- 本地目录：`C:\Users\HP\projects\deepseek-image-classifier`
- GitHub 仓库：`github.com/zyc771/deepseek-image-classifier`（PRIVATE）
- 产品名：DeepSeek图片分类工具（窗口标题 / exe 名 / 桌面快捷方式同名）

## 3. 改造点

| 项 | 说明 |
|---|---|
| 服务商 UI | 删除「API 服务商」下拉与「模型」输入框；`providers.py` 注册表保留（Kimi 隐藏扩展点） |
| 模型 | 固定 `get_provider("deepseek")["default_model"]`（`deepseek-v4-flash-vision-exp`），不再持久化 model |
| 配置 | QSettings 改用 `DeepSeekImageClassifier/Config`；只存 `keys/deepseek`（DPAPI）；旧 `api_key_b64` 迁移至此 |
| 一次性迁移 | 首次运行：新配置无 `keys/deepseek` 且旧配置 `KimiClassifier/Config` 有 → 解密旧 blob → 加密写入新配置（之后独立，不再读旧配置） |
| 接口 | `run_tab.set_config` 精简 7 参（api_key, src, out, categories, prompt, keywords, rpm）；`Classifier(service, api_key, model, ...)` 接口保留，调用方固定传 `"deepseek"` |
| 其余 | classifier.py、secure.py、providers.py、build 逻辑、判重/取消/预览清理全部保持 |

## 4. 测试

- 保留：test_providers（注册表含 Kimi）、test_classifier、test_secure
- 更新：test_run_tab_config_tuple（7 参）、test_config_tab_smoke（独立命名空间、
  无下拉/模型框、api_key_b64→keys/deepseek 迁移、旧配置一次性迁移）

## 5. 验证

1. 全量 pytest 通过
2. offscreen 冒烟（UI 构建、无下拉/模型框、迁移逻辑）
3. `python build.py` 打包 → `dist/DeepSeek图片分类工具.exe`
4. 桌面新建「DeepSeek图片分类工具.lnk」；原 Kimi 快捷方式不动
5. `gh repo create deepseek-image-classifier --private --push`
6. 用户真机验证（DeepSeek 预览一张）
