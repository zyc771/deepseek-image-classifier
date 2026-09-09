"""配置页 — API密钥、文件夹、分类、提示词、频率、单张测试"""
import base64
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QPlainTextEdit, QSlider, QSpinBox,
    QFileDialog, QMessageBox, QScrollArea, QFormLayout, QComboBox,
)
from PySide6.QtCore import Qt, QSettings, QThread
from app.providers import default_service, get_provider, provider_choices
from app.secure import decrypt_secret, encrypt_secret

DEFAULT_CATEGORIES = "科技;日常;学习;体育;军事;动漫;历史;地理;政治;游戏;经济"
DEFAULT_KEYWORDS = {
    "科技": "电脑;手机;芯片;AI;数码产品",
    "日常": "美食;宠物;聊天记录;自拍;家庭",
    "学习": "书本;教室;考试;笔记;学生",
    "体育": "运动;球赛;运动员;健身;跑步",
    "军事": "军装;武器;坦克;阅兵;装备",
    "动漫": "二次元;动漫角色;漫画;日系画风",
    "历史": "古代;老照片;文物;历史事件",
    "地理": "地图;地球;山川;自然风光",
    "政治": "领导人;会议;国旗;政府;外交",
    "游戏": "游戏界面;电竞;手柄;网游",
    "经济": "金钱;股票;K线;商业;人民币",
}
DEFAULT_GLOBAL_PROMPT = (
    "你是一个图片分类助手，所有图片本质上是搞笑/幽默内容。\n"
    "根据图片内容从以下分类中选择最匹配的一个。\n\n"
    "分类标准：\n{category_definitions}\n\n"
    "回复格式：分类名||置信度(0到1之间的数字)||关键词1,关键词2,关键词3"
)
DEFAULT_MODEL = "moonshot-v1-8k-vision-preview"
DEFAULT_RPM = 30


def _parse_categories(text: str) -> list[str]:
    """解析分号分隔的分类列表"""
    text = text.replace("；", ";")
    return [c.strip() for c in text.split(";") if c.strip()]


class ConfigTab(QWidget):
    def __init__(self, parent=None, settings: QSettings = None):
        super().__init__(parent)
        self._settings = settings or QSettings("KimiClassifier", "Config")
        self._kw_inputs: dict[str, QLineEdit] = {}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)

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

        # ── API密钥 ──
        group_key = QGroupBox("API 密钥")
        gk = QHBoxLayout(group_key)
        self._key_input = QLineEdit()
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("输入 Kimi API Key (sk-...开头)")
        gk.addWidget(self._key_input)
        self._key_toggle = QPushButton("显示")
        self._key_toggle.setCheckable(True)
        self._key_toggle.toggled.connect(self._toggle_key_visibility)
        gk.addWidget(self._key_toggle)
        layout.addWidget(group_key)

        # ── 文件夹 ──
        group_dir = QGroupBox("文件夹")
        gd = QVBoxLayout(group_dir)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("源文件夹:"))
        self._src_input = QLineEdit()
        self._src_input.setPlaceholderText("选择或拖入文件夹路径")
        self._src_input.setAcceptDrops(True)
        self._src_input.dragEnterEvent = self._make_drag_enter()
        self._src_input.dropEvent = self._make_drop(self._src_input)
        row1.addWidget(self._src_input)
        btn_src = QPushButton("浏览...")
        btn_src.clicked.connect(self._browse_src)
        row1.addWidget(btn_src)
        gd.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("输出目录:"))
        self._out_input = QLineEdit()
        self._out_input.setPlaceholderText("选择输出目录")
        self._out_input.setAcceptDrops(True)
        self._out_input.dragEnterEvent = self._make_drag_enter()
        self._out_input.dropEvent = self._make_drop(self._out_input)
        row2.addWidget(self._out_input)
        btn_out = QPushButton("浏览...")
        btn_out.clicked.connect(self._browse_out)
        row2.addWidget(btn_out)
        gd.addLayout(row2)
        layout.addWidget(group_dir)

        # ── 分类列表 ──
        group_cat = QGroupBox("分类列表（用分号隔开，中英文均可）")
        gc = QVBoxLayout(group_cat)
        self._cat_input = QLineEdit()
        self._cat_input.setPlaceholderText("科技;日常;学习;体育;...")
        self._cat_input.textChanged.connect(self._on_categories_changed)
        gc.addWidget(self._cat_input)
        btn_reset_cat = QPushButton("恢复默认11类")
        btn_reset_cat.clicked.connect(self._reset_categories)
        gc.addWidget(btn_reset_cat)
        layout.addWidget(group_cat)

        # ── 各分类关键词 ──
        group_kw = QGroupBox("各分类关键词（分号隔开）")
        self._kw_layout = QFormLayout()
        self._kw_container = QWidget()
        self._kw_container.setLayout(self._kw_layout)
        gkw = QVBoxLayout(group_kw)
        gkw.addWidget(self._kw_container)
        layout.addWidget(group_kw)

        # ── 总提示词 ──
        group_prompt = QGroupBox(
            "总提示词模板\n"
            "  {categories} → 分类列表（如 科技、日常、学习）\n"
            "  {category_definitions} → 分类定义块（如 - 科技: 电脑;手机;芯片）"
        )
        gp = QVBoxLayout(group_prompt)
        self._prompt_input = QPlainTextEdit()
        self._prompt_input.setMaximumHeight(160)
        gp.addWidget(self._prompt_input)
        btn_reset_prompt = QPushButton("恢复默认提示词")
        btn_reset_prompt.clicked.connect(lambda: self._prompt_input.setPlainText(DEFAULT_GLOBAL_PROMPT))
        gp.addWidget(btn_reset_prompt)
        layout.addWidget(group_prompt)

        # ── 模型与频率 ──
        group_model = QGroupBox("模型与频率")
        gm = QVBoxLayout(group_model)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("模型:"))
        self._model_input = QLineEdit(DEFAULT_MODEL)
        row3.addWidget(self._model_input)
        gm.addLayout(row3)
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("请求频率:"))
        self._rpm_slider = QSlider(Qt.Orientation.Horizontal)
        self._rpm_slider.setRange(1, 60)
        self._rpm_slider.setValue(DEFAULT_RPM)
        self._rpm_slider.valueChanged.connect(lambda v: self._rpm_spin.setValue(v))
        row4.addWidget(self._rpm_slider)
        self._rpm_spin = QSpinBox()
        self._rpm_spin.setRange(1, 60)
        self._rpm_spin.setValue(DEFAULT_RPM)
        self._rpm_spin.valueChanged.connect(lambda v: self._rpm_slider.setValue(v))
        self._rpm_label = QLabel("30 张/分钟")
        self._rpm_spin.valueChanged.connect(lambda v: self._rpm_label.setText(f"{v} 张/分钟"))
        row4.addWidget(self._rpm_spin)
        row4.addWidget(self._rpm_label)
        gm.addLayout(row4)
        layout.addWidget(group_model)

        # ── 单张测试 ──
        group_preview = QGroupBox("单张预览测试")
        gprev = QVBoxLayout(group_preview)
        row5 = QHBoxLayout()
        self._preview_path = QLineEdit()
        self._preview_path.setPlaceholderText("选择一张测试图片")
        row5.addWidget(self._preview_path)
        btn_pick = QPushButton("选择图片")
        btn_pick.clicked.connect(self._browse_preview)
        row5.addWidget(btn_pick)
        btn_test = QPushButton("▶ 测试")
        btn_test.clicked.connect(self._run_preview)
        row5.addWidget(btn_test)
        gprev.addLayout(row5)
        self._preview_result = QLabel("结果: (未测试)")
        self._preview_result.setWordWrap(True)
        gprev.addWidget(self._preview_result)
        layout.addWidget(group_preview)

        layout.addStretch()
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll)

        self._current_service = default_service()
        self._saved_keys: dict[str, str] = {}
        self._saved_models: dict[str, str] = {}
        self._preview_thread = None

        self._load_settings()  # 此初始化顺序: 先建控件再读配置
        self._on_categories_changed(self._cat_input.text())

    # ── 公开接口 ──
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
    def get_api_key(self) -> str:
        return self._key_input.text().strip()

    def get_source_dir(self) -> str:
        return self._src_input.text().strip()

    def get_output_dir(self) -> str:
        return self._out_input.text().strip()

    def get_categories(self) -> list[str]:
        return _parse_categories(self._cat_input.text())

    def get_category_keywords(self) -> dict[str, str]:
        result = {}
        for cat, input_widget in self._kw_inputs.items():
            val = input_widget.text().strip()
            if val:
                result[cat] = val
        return result

    def get_global_prompt(self) -> str:
        return self._prompt_input.toPlainText()

    def get_model(self) -> str:
        return self._model_input.text().strip()

    def get_rpm(self) -> int:
        return self._rpm_spin.value()

    def set_preview_result(self, category, confidence, keywords, raw, pt, ct, elapsed):
        kw_str = ", ".join(keywords) if keywords else "无"
        text = (
            f"分类: {category}  |  置信度: {confidence:.2f}  |  耗时: {elapsed:.1f}s\n"
            f"关键词: {kw_str}\n"
            f"Token: 输入{pt} / 输出{ct}\n"
            f"原始返回: {raw[:200]}"
        )
        self._preview_result.setText(text)

    # ── 配置持久化 ──
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

    def _load_settings(self):
        s = self._settings

        # ── Key 与模型：读取各服务商 + 旧格式迁移 ──
        s.beginGroup("keys")
        for sid in s.childKeys():
            plain = decrypt_secret(s.value(sid, ""))
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

    # ── 分类变化时重建关键词输入 ──
    def _on_categories_changed(self, text: str):
        new_cats = _parse_categories(text)
        old_cats = set(self._kw_inputs.keys())
        new_set = set(new_cats)

        # 移除不再存在的
        for cat in old_cats - new_set:
            self._kw_layout.removeRow(self._kw_inputs[cat])
            del self._kw_inputs[cat]

        # 添加新的
        for cat in new_cats:
            if cat not in self._kw_inputs:
                label = QLabel(f"{cat}:")
                input_widget = QLineEdit()
                input_widget.setPlaceholderText(f"关键词;关键词;...")
                if cat in DEFAULT_KEYWORDS:
                    input_widget.setText(DEFAULT_KEYWORDS[cat])
                self._kw_layout.addRow(label, input_widget)
                self._kw_inputs[cat] = input_widget

    def _reset_categories(self):
        self._cat_input.setText(DEFAULT_CATEGORIES)

    # ── 内部 ──
    def _toggle_key_visibility(self, checked):
        self._key_input.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)
        self._key_toggle.setText("隐藏" if checked else "显示")

    def _browse_src(self):
        d = QFileDialog.getExistingDirectory(self, "选择源文件夹")
        if d:
            self._src_input.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self._out_input.setText(d)

    def _browse_preview(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择测试图片", "",
                                           "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if f:
            self._preview_path.setText(f)

    def _run_preview(self):
        path = self._preview_path.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "错误", "请先选择一张测试图片")
            return
        if not self.get_api_key():
            QMessageBox.warning(self, "错误", "请先输入 API 密钥")
            return

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
        self._preview_clf.signals.log.connect(lambda msg: self._preview_result.setText(msg))

        self._preview_thread = QThread()
        self._preview_clf.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(lambda: self._preview_clf.preview(path))
        self._preview_thread.start()

    def _on_preview_done(self, cat, conf, kws, raw, pt, ct, elapsed):
        self.set_preview_result(cat, conf, kws, raw, pt, ct, elapsed)
        self._cleanup_preview_thread()

    def _cleanup_preview_thread(self):
        if self._preview_thread:
            self._preview_thread.quit()
            self._preview_thread.wait(2000)
            self._preview_thread.deleteLater()
            self._preview_thread = None

    def _make_drag_enter(self):
        def handler(event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
        return handler

    def _make_drop(self, target: QLineEdit):
        def handler(event):
            for url in event.mimeData().urls():
                p = url.toLocalFile()
                if Path(p).is_dir():
                    target.setText(p)
                    break
        return handler
