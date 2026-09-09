# Kimi 图片分类工具 v2 — 接入 DeepSeek 视觉模型 设计文档

- 日期：2026-09-09
- 状态：已确认
- 项目：`C:\Users\HP\projects\image-classifier-gui`

## 1. 背景与目标

现有工具通过 Kimi (Moonshot) API 对图片分类并按类归档。用户希望将视觉模型换成
DeepSeek（`deepseek-v4-flash-vision-exp`，OpenAI 兼容、支持 base64 data URL 传图），
同时保留 Kimi 作为可一键切换的备选，并顺手修复已确认的四个代码问题。

## 2. 范围

### 包含

- 双服务商切换（DeepSeek 官方 / Kimi 官方），默认 DeepSeek
- Windows DPAPI 加密存储 API Key（非 Windows 回退 base64）
- 取消/重开逻辑修复（不再阻塞主线程、防止双线程并存）
- 预览测试线程清理
- 判重策略从"仅文件名"改为"(文件名, 大小, 修改时间)"
- 旧 QSettings 配置迁移（旧 `api_key_b64` → Kimi 的 key）
- 重新打包 exe（桌面快捷方式指向不变）

### 不包含（YAGNI）

- 自定义任意 base_url / 第三方中转配置（注册表模式已预留，但本轮不做 UI）
- 子文件夹递归扫描、webp/bmp 支持、输出重命名规则等未确认需求
- 自动化测试框架搭建（项目现有 0 测试，本轮以手动验证为准）

## 3. 架构

### 3.1 Provider 注册表 — 新增 `app/providers.py`

```python
PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "default_model": "deepseek-v4-flash-vision-exp",
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
        "default_model": "kimi-k2.6",
    },
}

def get_provider(service: str) -> dict: ...
def default_service() -> str: ...          # "deepseek"
def provider_choices() -> list[dict]: ...  # UI 下拉用
```

职责：查询端点/默认模型；无网络、无状态。

### 3.2 安全存储 — 新增 `app/secure.py`

- `encrypt_secret(plain: str) -> str` / `decrypt_secret(blob: str) -> str`
- Windows：ctypes 调 `CryptProtectData` / `CryptUnprotectData`（DPAPI，仅当前用户可解密），
  结果 base64 编码后存 QSettings；零第三方依赖，PyInstaller 兼容
- 非 Windows：回退 base64 明文（与旧行为一致），并在回退时不报错

### 3.3 分类线程 — 修改 `app/classifier.py`

- 构造签名改为 `Classifier(service, api_key, model, source_dir, output_dir,
  categories, global_prompt, category_keywords=None, rpm=30)`
- `_classify_one` 内端点取自 `get_provider(self._service)["endpoint"]`，请求体结构不变
- 判重：`_filter_done` 收集输出目录各子目录文件的 `(name, size, mtime)` 集合；
  源文件 `(name, size, mtime)` 全等才算已处理；同名不同内容 → 重新分类并覆盖
- 取消响应：请求循环各步之间检查 `_cancelled`；线程退出后照常 emit `finished`

### 3.4 运行页 — 修改 `app/run_tab.py`

- `set_config` 增加 `service` 参数（`main_window.py` 同步传入）
- `_cancel`：调用 `classifier.cancel()` 后**不 wait 阻塞**；按钮进入"取消中…"禁用态；
  由 `finished` 信号统一走 `_on_finished` 重置按钮
- `_start`：若 `self._classifier` 仍 `isRunning()` 则弹提示拒绝启动
- 取消后 `finished` 的 summary 含 `cancelled: true` 标记，日志提示"已取消"

### 3.5 配置页 — 修改 `app/config_tab.py`

- 顶部新增 "API 服务商" 下拉框（选项来自 `provider_choices()`）
- 单个 API Key 输入框：切换服务商时保存当前 key 到该服务商、加载目标服务商 key
- 模型输入框按服务商记忆（`model/deepseek`、`model/kimi`），切换加载记忆值，空则填默认模型
- 预览测试：`preview_done` 后对 QThread 执行 `quit() → wait() → deleteLater()` 统一清理；
  预览测试始终使用**当前选中服务商 + 当前 Key 输入框 + 当前模型输入框**的即时值发起
- 保存结构（QSettings `KimiClassifier/Config`）：
  - `service`：当前服务商 id
  - `keys/<service>`：DPAPI 加密后的 key
  - `model/<service>`：该服务商模型名
  - 旧键 `api_key_b64` 首次运行时迁移为 `keys/kimi` 并删除旧键

## 4. 数据流

```
配置页（下拉选服务商 → key/模型 联动）
  → 切运行页 save_settings + get_*（含 service）
  → RunTab.set_config(service, ...)
  → 开始 → Classifier(service, key, model, ...)
  → requests.post(PROVIDERS[service].endpoint, base64 data URL + prompt)
  → 解析 "分类||置信度||关键词" → copy2 到 输出目录/分类/ → 统计
```

两家服务商共享同一响应解析逻辑；`_parse_response` 不改。

## 5. 错误处理

- DPAPI 解密失败（key 损坏/非本机用户）：捕获异常，回退尝试 base64 明文读取，均失败则清空该 key 输入并提示重新输入
- 取消后未发 finished：`finished` 为兜底重置信号；若线程在 `requests` 60s 超时中，取消最多延迟 60s 生效，UI 保持"取消中…"不闪退
- 分类失败、429 重试、限速逻辑保持现状

## 6. 验证（手动验收）

1. `python main.py` 启动，配置页切到 DeepSeek，填入 key
2. 单张预览测试：任意 jpg 走通 → 验证 base64 传图格式（唯一有外部风险的点）
3. 回切 Kimi 再预览一张 → 验证兼容没破坏
4. 用 2-3 张小图 + 输出目录已有同名的图各跑一轮：验证判重（同名同内容→跳过；同名不同内容→重分类并覆盖）
5. 分类进行中点"取消"→ 观察按钮进入"取消中…"且不卡界面 → 结束后可正常再"开始"
6. 连续多次单张预览 → 进程内存/线程不增长
7. 重启应用 → 配置与 key 均保留，注册表中 key 为加密 blob（非 base64 明文）
8. `python build.py` 重新打包 → 双击 dist exe 重复步骤 1-7 抽验

## 7. 交付物

- 新增：`app/providers.py`、`app/secure.py`、本设计文档
- 修改：`app/classifier.py`、`app/config_tab.py`、`app/run_tab.py`、`app/main_window.py`
- 重建：`dist/Kimi图片分类工具.exe`（桌面快捷方式指向不变）
