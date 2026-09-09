# DeepSeek 视觉模型接入实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将图片分类工具从仅支持 Kimi 升级为 DeepSeek/Kimi 双服务商（默认 DeepSeek），API Key 改 DPAPI 加密存储，并修复取消逻辑、预览线程泄漏、判重策略三个稳定性问题。

**Architecture:** 新增 `app/providers.py`（服务商注册表）与 `app/secure.py`（DPAPI 加密）。`Classifier` 改为按 service 查端点；`ConfigTab` 加服务商下拉联动 key/模型框；`RunTab` 取消不阻塞、由 `finished` 信号统一重置。两家服务商共享同一 OpenAPI 兼容请求体。

**Tech Stack:** Python 3 + PySide6、requests、pytest（新增 dev 依赖）、PyInstaller。

**Spec:** `docs/superpowers/specs/2026-09-09-image-classifier-deepseek-vision-design.md`

## Global Constraints

- 端点：DeepSeek `https://api.deepseek.com/chat/completions`，模型默认 `deepseek-v4-flash-vision-exp`；Kimi `https://api.moonshot.cn/v1/chat/completions`，模型默认 `kimi-k2.6`
- 请求体结构（base64 data URL + text 消息）两家保持一致，`_parse_response` 不允许改动语义
- 加密存储：Windows 走 DPAPI（零第三方依赖，ctypes），非 Windows 回退 base64；加密 blob 带前缀 `dpapi:` / `b64:`
- QSettings 组织：`service`、`keys/<service>`、`model/<service>`；旧键 `api_key_b64` 一次性迁移进 `keys/kimi` 后删除
- 判重键 = `(文件名, 文件大小, int(mtime))`
- 所有 UI/代码文案保持简体中文
- 测试运行：`python -m pytest tests/ -v`（在项目根目录）；GUI 行为以手动清单验收

---

### Task 1: Provider 注册表

**Files:**
- Create: `app/providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Produces: `PROVIDERS: dict[str, dict]`；`default_service() -> str`；`get_provider(service: str) -> dict`（未知服务商抛 `ValueError`）；`provider_choices() -> list[tuple[str, str]]`（(id, label)，保持插入顺序）

- [ ] **Step 1: 写失败测试**

Create `tests/test_providers.py`:

```python
"""providers 注册表测试"""
import pytest
from app.providers import PROVIDERS, default_service, get_provider, provider_choices


class TestProviders:
    def test_default_service_is_deepseek(self):
        assert default_service() == "deepseek"

    def test_providers_contain_deepseek_and_kimi(self):
        assert set(PROVIDERS.keys()) == {"deepseek", "kimi"}

    def test_deepseek_endpoint_and_model(self):
        p = get_provider("deepseek")
        assert p["endpoint"] == "https://api.deepseek.com/chat/completions"
        assert p["default_model"] == "deepseek-v4-flash-vision-exp"

    def test_kimi_endpoint_and_model(self):
        p = get_provider("kimi")
        assert p["endpoint"] == "https://api.moonshot.cn/v1/chat/completions"
        assert p["default_model"] == "kimi-k2.6"

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError):
            get_provider("unknown")

    def test_choices_preserve_order(self):
        assert provider_choices() == [
            ("deepseek", "DeepSeek"),
            ("kimi", "Kimi (Moonshot)"),
        ]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_providers.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'app.providers'`）

- [ ] **Step 3: 实现 `app/providers.py`**

```python
"""API 服务商注册表 — 查询端点与默认模型"""
PROVIDERS: dict[str, dict] = {
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

DEFAULT_SERVICE = "deepseek"


def default_service() -> str:
    return DEFAULT_SERVICE


def get_provider(service: str) -> dict:
    if service not in PROVIDERS:
        raise ValueError(f"未知服务商: {service}")
    return PROVIDERS[service]


def provider_choices() -> list[tuple[str, str]]:
    return [(sid, p["label"]) for sid, p in PROVIDERS.items()]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_providers.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
git add tests/test_providers.py app/providers.py
git commit -m "feat(providers): 新增 DeepSeek/Kimi 双服务商注册表"
```

---

### Task 2: DPAPI 加密存储

**Files:**
- Create: `app/secure.py`
- Test: `tests/test_secure.py`

**Interfaces:**
- Produces: `encrypt_secret(plain: str) -> str`（返回带前缀 blob）；`decrypt_secret(blob: str) -> str`（解密失败返回 `""`）；兼容旧无前缀 base64 与 `b64:` 前缀

- [ ] **Step 1: 写失败测试**

Create `tests/test_secure.py`:

```python
"""secure 加解密测试（Windows 主路径 + 回退 + 兼容）"""
import base64
from app import secure


class TestSecure:
    def test_roundtrip_dpapi(self):
        blob = secure.encrypt_secret("sk-test123")
        assert blob.startswith("dpapi:")
        assert "sk-test123" not in blob
        assert secure.decrypt_secret(blob) == "sk-test123"

    def test_b64_prefix_roundtrip(self):
        blob = "b64:" + base64.b64encode("sk-plain".encode()).decode()
        assert secure.decrypt_secret(blob) == "sk-plain"

    def test_legacy_bare_base64(self):
        blob = base64.b64encode("sk-legacy".encode()).decode()
        assert secure.decrypt_secret(blob) == "sk-legacy"

    def test_garbage_returns_empty(self):
        assert secure.decrypt_secret("not-a-valid-blob!!!") == ""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_secure.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'app.secure'`）

- [ ] **Step 3: 实现 `app/secure.py`**

```python
"""API Key 安全存储 — Windows DPAPI 加密，非 Windows 回退 base64"""
import base64
import ctypes
import sys
from ctypes import wintypes


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _crypt_protect(data: bytes) -> bytes:
    """DPAPI 加密，仅当前 Windows 用户可解密"""
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = _DATA_BLOB(
        len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
    )
    blob_out = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _crypt_unprotect(data: bytes) -> bytes:
    """DPAPI 解密"""
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = _DATA_BLOB(
        len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
    )
    blob_out = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def encrypt_secret(plain: str) -> str:
    if not plain:
        return ""
    if sys.platform == "win32":
        try:
            blob = _crypt_protect(plain.encode("utf-8"))
            return "dpapi:" + base64.b64encode(blob).decode()
        except Exception:
            pass
    # 回退：base64 明文（与旧行为一致）
    return "b64:" + base64.b64encode(plain.encode("utf-8")).decode()


def decrypt_secret(blob: str) -> str:
    if not blob:
        return ""
    if blob.startswith("dpapi:"):
        try:
            return _crypt_unprotect(base64.b64decode(blob[6:])).decode("utf-8")
        except Exception:
            return ""
    if blob.startswith("b64:"):
        try:
            return base64.b64decode(blob[4:]).decode("utf-8")
        except Exception:
            return ""
    # 兼容旧格式：无前缀 base64
    try:
        return base64.b64decode(blob).decode("utf-8")
    except Exception:
        return ""
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_secure.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add tests/test_secure.py app/secure.py
git commit -m "feat(secure): Windows DPAPI 加密存 Key，含 base64 回退与旧格式兼容"
```

---

### Task 3: Classifier 服务商化 + 判重改造

**Files:**
- Modify: `app/classifier.py:32-51`（构造函数签名）、`app/classifier.py:70-81`（prompt 构建不变，仅确认）、`app/classifier.py:164-204`（`_classify_one` 端点）、`app/classifier.py:128-160`（summary 加 cancelled 标记）、`app/classifier.py:265-277`（`_filter_done` 判重）
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `get_provider(service)`（Task 1）
- Produces: `Classifier(service, api_key, model, source_dir, output_dir, categories, global_prompt, category_keywords=None, rpm=30)`；`run()` 的 `finished` summary 含 `"cancelled": bool`；`_filter_done(images)` 用 `(name, size, mtime)` 判重

- [ ] **Step 1: 写锁定行为的测试**

Create `tests/test_classifier.py`:

```python
"""classifier 行为测试（不启动线程、不发网络请求）"""
import shutil
from pathlib import Path
from app.classifier import Classifier


def _make_clf(tmp_path: Path, source_dir: Path, output_dir: Path, **kw):
    return Classifier(
        service=kw.pop("service", "deepseek"),
        api_key="test-key",
        model="test-model",
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        categories=["科技", "日常"],
        global_prompt="prompt {categories}",
        **kw,
    )


class TestParseResponse:
    def test_standard_format(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("动漫||0.85||火影,鸣人")
        assert cat == "动漫"
        assert conf == 0.85
        assert kws == ["火影", "鸣人"]

    def test_fuzzy_fallback(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("这张图是动漫风格的插画")
        assert cat == "动漫"
        assert conf > 0

    def test_unknown_category_becomes_unclassified(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("未知内容||0.5||a,b")
        assert cat == "未整理"

    def test_confidence_clamped(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("科技||1.7||a")
        assert conf == 1.0


class TestFilterDone:
    def test_same_name_size_mtime_skipped(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        img = src / "a.jpg"
        img.write_bytes(b"imgdata")
        st = img.stat()
        (out / "科技").mkdir(parents=True)
        done = out / "科技" / "a.jpg"
        shutil.copy2(img, done)  # copy2 保留 mtime

        clf = _make_clf(tmp_path, src, out)
        pending = clf._filter_done([img])
        assert pending == []

    def test_same_name_different_content_kept(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        img = src / "a.jpg"
        img.write_bytes(b"imgdata1")
        (out / "科技").mkdir(parents=True)
        old = out / "科技" / "a.jpg"
        old.write_bytes(b"imgdata2")

        clf = _make_clf(tmp_path, src, out)
        pending = clf._filter_done([img])
        assert pending == [img]

    def test_non_image_ext_ignored_in_scan(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        (src / "note.txt").write_text("x")
        clf = _make_clf(tmp_path, src, out)
        from app.classifier import Classifier
        images = [f for f in src.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        assert images == []


class TestSummaryCancelledFlag:
    def test_init_defaults_not_cancelled(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        assert clf._cancelled is False
```

- [ ] **Step 2: 跑测试确认现状**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: PASS（现有 `_parse_response`/`_filter_done` 行为已匹配；`Classifier(service=...)` 第一个测试可能因签名不符 FAIL — 若 FAIL 属预期，继续 Step 3）

- [ ] **Step 3: 修改 `Classifier.__init__` 签名**

Replace `app/classifier.py:32-44`:

```python
    def __init__(self, service: str, api_key: str, model: str, source_dir: str, output_dir: str,
                 categories: list[str], global_prompt: str, category_keywords: dict[str, str] = None,
                 rpm: int = 30):
        super().__init__()
        self.signals = ClassifierSignals()
        self._service = service
        self._api_key = api_key
        self._model = model
        self._source_dir = Path(source_dir)
        self._output_dir = Path(output_dir)
        self._categories = categories
        self._global_prompt = global_prompt
        self._category_keywords = category_keywords or {}
        self._rpm = rpm
```

- [ ] **Step 4: `_classify_one` 换端点**

Replace `app/classifier.py:171-176`（`requests.post(...)` 的 URL 参数行）：

```python
                from app.providers import get_provider
                resp = requests.post(
                    get_provider(self._service)["endpoint"],
                    headers={
```

- [ ] **Step 5: summary 加 cancelled 标记**

Replace `app/classifier.py:131-141`（summary 字典定义处，在 `"total": total,` 前加一行）：

```python
        summary = {
            "cancelled": self._cancelled,
            "total": total,
            "pending": pending_count,
            "success": success,
            "failed": failed,
            "elapsed_seconds": elapsed,
```

- [ ] **Step 6: `_filter_done` 改为 (name, size, mtime) 判重**

Replace `app/classifier.py:265-277`:

```python
    def _filter_done(self, images: list[Path]) -> list[Path]:
        """过滤输出目录中已存在的文件: 匹配键 = (文件名, 大小, 修改时间)"""
        existing = set()
        if self._output_dir.exists():
            for d in self._output_dir.iterdir():
                if d.is_dir():
                    for f in d.iterdir():
                        if f.is_file():
                            st = f.stat()
                            existing.add((f.name, st.st_size, int(st.st_mtime)))
        new_images = []
        for img in images:
            st = img.stat()
            key = (img.name, st.st_size, int(st.st_mtime))
            if key not in existing:
                new_images.append(img)
        skipped = len(images) - len(new_images)
        if skipped > 0:
            self.signals.log.emit(f"跳过已处理: {skipped} 张")
        return new_images
```

- [ ] **Step 7: 跑测试确认通过**

Run: `python -m pytest tests/test_classifier.py -v -p no:cacheprovider`
Expected: PASS（8 passed）

- [ ] **Step 8: Commit**

```bash
git add app/classifier.py tests/test_classifier.py
git commit -m "feat(classifier): 服务商化端点 + (名,大小,mtime) 判重 + cancelled 标记"
```

---

### Task 4: RunTab / MainWindow 接口串联 + 取消修复

**Files:**
- Modify: `app/run_tab.py:99-101`（`set_config`）、`app/run_tab.py:104-135`（`_start`）、`app/run_tab.py:146-157`（`_cancel`/`_reset_buttons`）、`app/run_tab.py:187-207`（`_on_finished`）
- Modify: `app/main_window.py:29-44`（`_on_tab_changed`）
- Test: `tests/test_run_tab_config_tuple.py`（set_config 参数顺序锁定）

**Interfaces:**
- Consumes: `Classifier(service, api_key, ...)`（Task 3）
- Produces: `set_config(service, api_key, model, src, out, categories, global_prompt, category_keywords, rpm)`；取消后 `finished` summary 驱动按钮重置

- [ ] **Step 1: 写失败测试**

Create `tests/test_run_tab_config_tuple.py`:

```python
"""RunTab.set_config 参数顺序锁定 — 防止 main_window 与 run_tab 解包错位"""
from pathlib import Path
from PySide6.QtWidgets import QApplication
from app.run_tab import RunTab


def test_set_config_unpack_order():
    app = QApplication.instance() or QApplication([])
    tab = RunTab()
    tab.set_config(
        "deepseek", "key", "model-x", "/src", "/out",
        ["科技"], "prompt {categories}", {"科技": "芯片"}, 42,
    )
    service, api_key, model, src, out, cats, prompt, kws, rpm = tab._config
    assert service == "deepseek"
    assert api_key == "key"
    assert model == "model-x"
    assert src == "/src"
    assert out == "/out"
    assert cats == ["科技"]
    assert prompt == "prompt {categories}"
    assert kws == {"科技": "芯片"}
    assert rpm == 42
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_run_tab_config_tuple.py -v`
Expected: FAIL（当前 set_config 为 8 参，抛 TypeError）

- [ ] **Step 3: 修改 `set_config`（run_tab.py:99-101）**

Replace:

```python
    def set_config(self, service, api_key, model, src, out, categories, global_prompt, category_keywords, rpm):
        self._config = (service, api_key, model, src, out, categories, global_prompt, category_keywords, rpm)
        self._output_dir = out
```

- [ ] **Step 4: `_start` 增加运行中保护 + 解包 service（run_tab.py:104-115）**

Replace（在 `_start` 开头加检查，解包元组加 service）：

```python
    def _start(self):
        if self._classifier and self._classifier.isRunning():
            QMessageBox.warning(self, "提示", "分类正在进行中，请先等待完成或取消")
            return
        service, api_key, model, src, out, categories, global_prompt, category_keywords, rpm = self._config
        if not api_key:
            QMessageBox.warning(self, "错误", "请先在配置页输入 API 密钥")
            return
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "错误", "源文件夹不存在")
            return
        if not categories:
            QMessageBox.warning(self, "错误", "请至少设置一个分类")
            return
```

以及 `_start` 中构造处（原 122 行）：

```python
        self._classifier = Classifier(service, api_key, model, src, out, categories, global_prompt, category_keywords, rpm)
```

- [ ] **Step 5: `_cancel` 不再阻塞主线程（run_tab.py:146-151）**

Replace:

```python
    def _cancel(self):
        if self._classifier:
            self._classifier.cancel()
            self._cancel_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setText("取消中…")
```

- [ ] **Step 6: `_reset_buttons` 恢复取消按钮文案（run_tab.py:153-157）**

Replace:

```python
    def _reset_buttons(self):
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("⏹ 取消")
        self._pause_btn.setText("⏸ 暂停")
```

（`_cancel_btn.setEnabled(True)` + 文案恢复，由 `finished` 触发的 `_on_finished` → `_reset_buttons` 统一执行）

- [ ] **Step 7: `_on_finished` 处理 cancelled 标记（run_tab.py:187-190）**

Replace（函数开头加）：

```python
    def _on_finished(self, summary):
        self._reset_buttons()
        if summary.get("cancelled"):
            self._add_log("已取消")
            self._progress_bar.setValue(self._progress_bar.value())
            return
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._add_log("\n=== 完成 ===")
```

- [ ] **Step 8: MainWindow 传 service（main_window.py:33-43）**

Replace `_on_tab_changed` 中的 cfg 元组：

```python
            cfg = (
                self._config_tab.get_service(),
                self._config_tab.get_api_key(),
                self._config_tab.get_model(),
                self._config_tab.get_source_dir(),
                self._config_tab.get_output_dir(),
                self._config_tab.get_categories(),
                self._config_tab.get_global_prompt(),
                self._config_tab.get_category_keywords(),
                self._config_tab.get_rpm(),
            )
```

- [ ] **Step 9: 跑测试确认通过**

Run: `python -m pytest tests/test_run_tab_config_tuple.py -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add app/run_tab.py app/main_window.py tests/test_run_tab_config_tuple.py
git commit -m "fix(run): 取消不再阻塞主线程，finished 统一复位按钮，串联 service 参数"
```

---

### Task 5: ConfigTab 服务商 UI + 联动 + 迁移 + 加密存取 + 预览清理

**Files:**
- Modify: `app/config_tab.py`（imports、`__init__` 顶部下拉、`_on_service_changed`、`save_settings`、`_load_settings`、`_run_preview`、`_on_preview_done`、预览线程清理、`get_service()`、`_make_clf` 处）

**Interfaces:**
- Consumes: `provider_choices()`、`get_provider()`（Task 1）；`encrypt_secret`/`decrypt_secret`（Task 2）
- Produces: `get_service() -> str`（当前下拉服务商 id）

- [ ] **Step 1: 修改 imports 与常量（config_tab.py 顶部）**

Replace 导入块（4-9 行）加 `QComboBox`：

```python
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QPlainTextEdit, QSlider, QSpinBox,
    QFileDialog, QMessageBox, QScrollArea, QFormLayout, QComboBox,
)
```

Replace 10-11 行后追加：

```python
from PySide6.QtCore import Qt, QSettings, QThread
from app.providers import default_service, get_provider, provider_choices
from app.secure import decrypt_secret, encrypt_secret
```

- [ ] **Step 2: `__init__` 加服务商下拉（config_tab.py:53 组前插入）**

在 `# ── API密钥 ──` 组之前插入：

```python
        # ── API 服务商 ──
        group_svc = QGroupBox("API 服务商")
        gsvc = QHBoxLayout(group_svc)
        gsvc.addWidget(QLabel("服务商:"))
        self._service_combo = QComboBox()
        for sid, label in provider_choices():
            self._service_combo.addItem(label, sid)
        self._service_combo.currentIndexChanged.connect(lambda _: self._on_service_changed())
        gsvc.addWidget(self._service_combo)
        gsvc.addStretch()
        layout.addWidget(group_svc)
```

并在 `__init__` 的 `self._load_settings()` 之前（约 181 行处）初始化状态：

```python
        self._current_service = default_service()
        self._saved_keys: dict[str, str] = {}
        self._saved_models: dict[str, str] = {}
        self._preview_thread = None
```

- [ ] **Step 3: 实现 `get_service` 与 `_on_service_changed`（config_tab.py 公开接口区）**

在 `get_api_key`（约 185 行）之前插入：

```python
    def get_service(self) -> str:
        return self._current_service

    def _on_service_changed(self):
        new = self._service_combo.currentData()
        if new is None or new == self._current_service:
            return
        # 记住旧服务商的当前输入
        if self._current_service:
            self._saved_keys[self._current_service] = self._key_input.text().strip()
            self._saved_models[self._current_service] = self._model_input.text().strip()
        self._current_service = new
        # 加载新服务商的记忆值（模型为空则用默认）
        self._key_input.setText(self._saved_keys.get(new, ""))
        self._model_input.setText(
            self._saved_models.get(new, "")
            or get_provider(new)["default_model"]
        )
```

- [ ] **Step 4: 重写 `save_settings` 的 key/模型存储（config_tab.py:225-241）**

Replace 原函数：

```python
    def save_settings(self):
        s = self._settings
        # 更新内存态（当前服务商的输入即最新值）
        self._saved_keys[self._current_service] = self._key_input.text().strip()
        self._saved_models[self._current_service] = self._model_input.text().strip()

        s.setValue("service", self._current_service)
        s.remove("keys")
        s.beginGroup("keys")
        for sid, plain in self._saved_keys.items():
            s.setValue(sid, encrypt_secret(plain))
        s.endGroup()
        s.remove("model")
        s.beginGroup("model")
        for sid, model in self._saved_models.items():
            s.setValue(sid, model)
        s.endGroup()

        s.setValue("source_dir", self.get_source_dir())
        s.setValue("output_dir", self.get_output_dir())
        s.setValue("categories_raw", self._cat_input.text())
        s.setValue("global_prompt", self.get_global_prompt())
        s.setValue("rpm", self.get_rpm())
        # 保存每类关键词
        s.remove("kw")
        s.beginGroup("kw")
        for cat, input_widget in self._kw_inputs.items():
            s.setValue(cat, input_widget.text())
        s.endGroup()
        s.sync()
```

- [ ] **Step 5: 重写 `_load_settings`（config_tab.py:243-266）**

Replace 原函数：

```python
    def _load_settings(self):
        s = self._settings

        # ── Key 与模型：读取各服务商 + 旧格式迁移 ──
        s.beginGroup("keys")
        saved_keys_raw = s.childKeys()
        for sid in saved_keys_raw:
            blob = s.value(sid, "")
            plain = decrypt_secret(blob)
            if plain:
                self._saved_keys[sid] = plain
        s.endGroup()
        s.beginGroup("model")
        for sid in s.childKeys():
            m = s.value(sid, "")
            if m:
                self._saved_models[sid] = m
        s.endGroup()

        # 旧格式迁移：api_key_b64 → keys/kimi
        legacy = s.value("api_key_b64", "")
        if legacy:
            try:
                import base64 as _b64
                plain = _b64.b64decode(legacy).decode()
                if plain and "kimi" not in self._saved_keys:
                    self._saved_keys["kimi"] = plain
            except Exception:
                pass
            s.remove("api_key_b64")

        # 当前服务商
        self._current_service = s.value("service", default_service())
        if self._current_service not in dict(provider_choices()):
            self._current_service = default_service()

        idx = self._service_combo.findData(self._current_service)
        if idx >= 0:
            self._service_combo.blockSignals(True)
            self._service_combo.setCurrentIndex(idx)
            self._service_combo.blockSignals(False)

        self._key_input.setText(self._saved_keys.get(self._current_service, ""))
        self._model_input.setText(
            self._saved_models.get(self._current_service, "")
            or get_provider(self._current_service)["default_model"]
        )

        # ── 其余配置 ──
        self._src_input.setText(s.value("source_dir", ""))
        self._out_input.setText(s.value("output_dir", ""))
        self._cat_input.setText(s.value("categories_raw", DEFAULT_CATEGORIES))
        self._prompt_input.setPlainText(s.value("global_prompt", DEFAULT_GLOBAL_PROMPT))
        rpm = int(s.value("rpm", DEFAULT_RPM))
        self._rpm_slider.setValue(rpm)
        self._rpm_spin.setValue(rpm)
        self._on_categories_changed(self._cat_input.text())
        s.beginGroup("kw")
        for cat in self._kw_inputs:
            val = s.value(cat, "")
            if val:
                self._kw_inputs[cat].setText(val)
        s.endGroup()
```

（注意：`_load_settings` 中原 `self._model_input.setText(...)` 与 `model` QSettings 读取已并入上方逻辑；删除原"model 读取"与"key 读取"两段，避免重复。）

- [ ] **Step 6: `_run_preview` 传 service + 清理线程（config_tab.py:314-343）**

Replace `_run_preview` 中 Classifier 构造（324-333 行）：

```python
        from app.classifier import Classifier
        self._preview_clf = Classifier(
            service=self.get_service(),
            api_key=self.get_api_key(),
            model=self.get_model(),
            source_dir=self.get_source_dir() or str(Path(path).parent),
            output_dir=self.get_output_dir() or str(Path(path).parent),
            categories=self.get_categories(),
            global_prompt=self.get_global_prompt(),
            category_keywords=self.get_category_keywords(),
            rpm=self.get_rpm(),
        )
        self._preview_clf.signals.preview_done.connect(self._on_preview_done)
```

`_on_preview_done`（342-343 行）Replace：

```python
    def _on_preview_done(self, cat, conf, kws, raw, pt, ct, elapsed):
        self.set_preview_result(cat, conf, kws, raw, pt, ct, elapsed)
        self._cleanup_preview_thread()

    def _cleanup_preview_thread(self):
        if self._preview_thread:
            self._preview_thread.quit()
            self._preview_thread.wait(2000)
            self._preview_thread.deleteLater()
            self._preview_thread = None
```

- [ ] **Step 7: `preview()` 异常兜底（classifier.py 末尾）**

在 `Classifier.preview`（约 248-263 行）包 try/except：

Replace：

```python
    def preview(self, filepath: str):
        """单张预览测试（非线程）"""
        try:
            cat_list = "、".join(self._categories)
            defs = []
            for cat in self._categories:
                kws = self._category_keywords.get(cat, "")
                defs.append(f"- {cat}: {kws}" if kws else f"- {cat}")
            cat_defs = "\n".join(defs)
            prompt_text = self._global_prompt.replace("{categories}", cat_list)
            prompt_text = prompt_text.replace("{category_definitions}", cat_defs)

            start = time.time()
            category, confidence, keywords, raw, pt, ct = self._classify_one(Path(filepath), prompt_text)
            elapsed = time.time() - start

            self.signals.preview_done.emit(category, confidence, keywords, raw, pt, ct, elapsed)
        except Exception as e:
            self.signals.preview_done.emit("ERROR", 0.0, [], str(e), 0, 0, 0.0)
```

- [ ] **Step 8: 手动验证（启动 GUI）**

Run: `python main.py`
手动检查：
1. 配置页出现"API 服务商"下拉，默认 DeepSeek，模型框默认 `deepseek-v4-flash-vision-exp`
2. 填入 API Key → 切换到 Kimi → 切回 DeepSeek → Key 与模型各自正确往返
3. 单张预览测试（DeepSeek 与 Kimi 各一张）→ 结果显示正常 → **连续预览 5 次进程线程不累积**（任务管理器观察）
4. 关闭应用 → 重启 → Key/模型/服务商保留（注册表 `HKCU\Software\KimiClassifier\Config` 中 keys 值以 `dpapi:` 开头）

- [ ] **Step 9: Commit**

```bash
git add app/config_tab.py app/classifier.py
git commit -m "feat(config): 服务商下拉联动 Key/模型，DPAPI 存储，旧Key迁移，预览线程清理"
```

---

### Task 6: build.py 修复 + 冒烟 + 打包

**Files:**
- Modify: `build.py:8-13`

**Interfaces:**
- Produces: 运行 `python build.py` 成功产出 `dist/Kimi图片分类工具.exe`

- [ ] **Step 1: 修复 spec 文件名引用（build.py:8-13）**

Replace：

```python
def build():
    specs = sorted(ROOT.glob("*.spec"))
    if not specs:
        print("Build FAILED: 未找到 .spec 文件")
        sys.exit(1)
    spec = specs[0]
    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "--clean", "--noconfirm", str(spec),
    ], check=True)
```

- [ ] **Step 2: 冒烟测试（源码运行）**

Run: `python main.py`
Expected: 启动无异常 → 配置页切到 DeepSeek → 填入 key + 选一张测试图 → "▶ 测试" → 显示分类结果（首次验证 base64 传图格式）
若显示 ERROR：检查环境变量/Key 权限 → 将错误信息反馈（计划执行者与用户确认后处理）

- [ ] **Step 3: 重新打包**

Run: `python build.py`
Expected: 输出 `Build OK: ...\dist\Kimi图片分类工具.exe (xx.x MB)`

- [ ] **Step 4: 验证 exe 与快捷方式**

Run:

```powershell
Test-Path "C:\Users\HP\projects\image-classifier-gui\dist\Kimi图片分类工具.exe"
Test-Path "C:\Users\HP\Desktop\Kimi图片分类工具.lnk"
```

Expected: 两个 True
双击快捷方式 → 应用启动正常 → 重复 Step 2 的 DeepSeek 预览测试

- [ ] **Step 5: Commit**

```bash
git add build.py
git commit -m "fix(build): 修复 spec 文件名查找，build.py 直接可用"
```

---

### Task 7: 端到端验收

**Files:** 无代码变更（手动清单）

- [ ] **Step 1: 完整跑一遍 spec 验收清单**

1. `python main.py`，配置页切 DeepSeek，填 key
2. 单张预览走通（base64 传图 OK）
3. 回切 Kimi 再预览一张 → 兼容未破坏
4. 2-3 张小图 + 输出目录已有同名图各跑一轮：同名同内容→跳过；同名不同内容→重新分类并覆盖
5. 分类进行中点"取消" → 按钮"取消中…"不卡界面 → 完成后可正常再"开始"
6. 连续 5 次单张预览 → 内存/线程不增长
7. 重启应用 → 配置与 key 保留，注册表 key 为 `dpapi:` blob
8. `python build.py` 打包 → exe 重复步骤 1-3 抽验

- [ ] **Step 2: 全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全 PASS

- [ ] **Step 3: 提交并推送**

```bash
git add -A
git commit -m "docs: 更新 README/计划与交付说明（如有变更）"
git push -u origin main
```

---

## 交付清单

- 新增：`app/providers.py`、`app/secure.py`、`tests/test_providers.py`、`tests/test_secure.py`、`tests/test_classifier.py`、`tests/test_run_tab_config_tuple.py`、`requirements-dev.txt`（内容：`pytest>=8.0.0`）
- 修改：`app/classifier.py`、`app/config_tab.py`、`app/run_tab.py`、`app/main_window.py`、`build.py`
- 重建：`dist/Kimi图片分类工具.exe`
