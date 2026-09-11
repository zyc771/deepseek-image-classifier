"""配置页 — API密钥、文件夹、分类、提示词、频率、单张测试"""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QPlainTextEdit, QSlider, QSpinBox,
    QFileDialog, QMessageBox, QScrollArea, QFormLayout,
    QCheckBox, QDoubleSpinBox, QSplitter,
)
from PySide6.QtCore import Qt, QSettings, QThread
from app.providers import get_provider
from app.secure import decrypt_secret, encrypt_secret

SERVICE = "deepseek"
LEGACY_ORG = "KimiClassifier"

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
    "你是一个图片内容分类助手，所有图片本质上是搞笑/幽默内容。\n"
    "请根据图片的【题材】从以下分类中选择最匹配的一个。\n\n"
    "分类标准：\n{category_definitions}\n\n"
    "判定优先级：题材 > 画风 > 文字信息。\n"
    "- 二次元/漫画/表情包画风的图片，只要题材指向明确（时政讽刺、军事装备、历史事件），按题材归类；"
    "「动漫」仅指没有现实题材指向的纯二次元作品。\n"
    "- 「日常」不是兜底类：只有真实生活场景（美食、宠物、自拍、居家、聊天记录）才归日常。\n"
    "  自然风景/山川/地图 → 地理；书本/教室/笔记/学生 → 学习；电脑/手机/数码/芯片 → 科技。\n"
    "- 历史人物（含近现代）、老照片、文物、年代场景 → 历史；当代时政活动（会议、外交、国旗、政府）→ 政治；"
    "武器装备、军装、阅兵 → 军事。\n"
    "- 游戏界面/电竞/手柄/网游画面 → 游戏。\n\n"
    "置信度分档（严格遵守）：\n"
    "- 0.90-1.00 主体与题材一眼可辨，无歧义\n"
    "- 0.70-0.89 较有把握，存在少量其他可能\n"
    "- 0.60-0.69 倾向性判断\n"
    "- 0.00-0.59 画面信息不足、多类都可能、主体不明 —— 请诚实给低分\n\n"
    "输出格式（仅输出 JSON，不要任何解释或代码块标记）：\n"
    '{"category": "分类名", "confidence": 0到1之间的数字, "keywords": ["关键词1", "关键词2", "关键词3"]}'
)
DEFAULT_RPM = 30
DEFAULT_CONCURRENCY = 3


def _parse_categories(text: str) -> list[str]:
    """解析分号分隔的分类列表"""
    text = text.replace("；", ";")
    return [c.strip() for c in text.split(";") if c.strip()]


def _as_bool(value) -> bool:
    """健壮地解析 QSettings 布尔值（注册表中可能是 'true'/'false' 字符串或布尔）"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


class ConfigTab(QWidget):
    def __init__(self, parent=None, settings: QSettings = None, legacy_settings: QSettings = None):
        super().__init__(parent)
        self._settings = settings or QSettings("DeepSeekImageClassifier", "Config")
        # legacy_settings 用于一次性迁移旧工具配置；None = 真实旧工具命名空间（注册表）
        self._legacy_settings = legacy_settings
        self._kw_inputs: dict[str, QLineEdit] = {}

        left_container = QWidget()
        self._left_layout = QVBoxLayout(left_container)
        self._left_layout.setSpacing(10)
        right_container = QWidget()
        self._right_layout = QVBoxLayout(right_container)
        self._right_layout.setSpacing(10)

        # ── API密钥 ──
        group_key = QGroupBox("API 密钥 (DeepSeek)")
        gk = QHBoxLayout(group_key)
        self._key_input = QLineEdit()
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("输入 DeepSeek API Key (sk-...开头)")
        gk.addWidget(self._key_input)
        self._key_toggle = QPushButton("显示")
        self._key_toggle.setCheckable(True)
        self._key_toggle.toggled.connect(self._toggle_key_visibility)
        gk.addWidget(self._key_toggle)
        self._left_layout.addWidget(group_key)

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
        self._left_layout.addWidget(group_dir)

        # ── 分类列表 ──
        group_cat = QGroupBox("分类列表（用分号隔开，中英文均可）")
        gc = QVBoxLayout(group_cat)
        self._cat_input = QLineEdit()
        self._cat_input.setPlaceholderText("科技;日常;学习;体育;...")
        self._cat_input.textChanged.connect(self._on_categories_changed)
        gc.addWidget(self._cat_input)
        self._left_layout.addWidget(group_cat)

        # ── 各分类关键词 ──
        group_kw = QGroupBox("各分类关键词（分号隔开）")
        self._kw_layout = QFormLayout()
        self._kw_container = QWidget()
        self._kw_container.setLayout(self._kw_layout)
        gkw = QVBoxLayout(group_kw)
        gkw.addWidget(self._kw_container)
        self._right_layout.addWidget(group_kw)

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
        self._right_layout.addWidget(group_prompt)

        # ── 图像处理与筛选 ──
        group_img = QGroupBox("图像处理与筛选")
        gi = QVBoxLayout(group_img)
        self._original_check = QCheckBox(
            "原图模式（跳过缩图/压缩，原图直传：更慢更贵，仅在需要极致细节时开启）"
        )
        gi.addWidget(self._original_check)
        row6 = QHBoxLayout()
        row6.addWidget(QLabel("低置信度阈值:"))
        self._low_conf_spin = QDoubleSpinBox()
        self._low_conf_spin.setRange(0.5, 0.9)
        self._low_conf_spin.setSingleStep(0.05)
        self._low_conf_spin.setDecimals(2)
        self._low_conf_spin.setValue(0.6)
        row6.addWidget(self._low_conf_spin)
        self._low_conf_hint = QLabel("低于该值归入『待确认』文件夹")
        row6.addWidget(self._low_conf_hint)
        row6.addStretch()
        gi.addLayout(row6)
        self._left_layout.addWidget(group_img)

        # ── 请求频率与并发 ──
        group_model = QGroupBox("请求频率与并发")
        gm = QVBoxLayout(group_model)
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("请求频率:"))
        self._rpm_slider = QSlider(Qt.Orientation.Horizontal)
        self._rpm_slider.setRange(1, 600)
        self._rpm_slider.setValue(DEFAULT_RPM)
        self._rpm_slider.valueChanged.connect(lambda v: self._rpm_spin.setValue(v))
        row4.addWidget(self._rpm_slider)
        self._rpm_spin = QSpinBox()
        self._rpm_spin.setRange(1, 600)
        self._rpm_spin.setValue(DEFAULT_RPM)
        self._rpm_spin.valueChanged.connect(lambda v: self._rpm_slider.setValue(v))
        self._rpm_label = QLabel("30 张/分钟")
        self._rpm_spin.valueChanged.connect(lambda v: self._rpm_label.setText(f"{v} 张/分钟"))
        row4.addWidget(self._rpm_spin)
        row4.addWidget(self._rpm_label)
        gm.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("并发数:"))
        self._conc_spin = QSpinBox()
        self._conc_spin.setRange(1, 8)
        self._conc_spin.setValue(DEFAULT_CONCURRENCY)
        row5.addWidget(self._conc_spin)
        row5.addWidget(QLabel("（同时发送的请求数，越高越快；429 频繁时请调低）"))
        row5.addStretch()
        gm.addLayout(row5)
        self._left_layout.addWidget(group_model)

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
        self._right_layout.addWidget(group_preview)

        self._left_layout.addStretch()
        self._right_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_container)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(left_scroll)
        self._splitter.addWidget(right_scroll)
        self._splitter.setSizes([430, 590])

        outer = QVBoxLayout(self)
        outer.addWidget(self._splitter)

        self._preview_thread = None

        self._load_settings()  # 此初始化顺序: 先建控件再读配置
        self._on_categories_changed(self._cat_input.text())

    # ── 公开接口 ──
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

    def set_global_prompt(self, text: str):
        """由评估页回写提示词"""
        self._prompt_input.setPlainText(text)

    def get_rpm(self) -> int:
        return self._rpm_spin.value()

    def get_concurrency(self) -> int:
        return self._conc_spin.value()

    def get_use_original(self) -> bool:
        return self._original_check.isChecked()

    def get_low_conf(self) -> float:
        return self._low_conf_spin.value()

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
        s.setValue("keys", encrypt_secret(self.get_api_key()))

        s.setValue("source_dir", self.get_source_dir())
        s.setValue("output_dir", self.get_output_dir())
        s.setValue("categories_raw", self._cat_input.text())
        s.setValue("global_prompt", self.get_global_prompt())
        s.setValue("rpm", self.get_rpm())
        s.setValue("concurrency", self.get_concurrency())
        s.setValue("use_original", self.get_use_original())
        s.setValue("low_conf_threshold", self.get_low_conf())
        # 保存每类关键词
        s.remove("kw")
        s.beginGroup("kw")
        for cat, input_widget in self._kw_inputs.items():
            s.setValue(cat, input_widget.text())
        s.endGroup()
        s.sync()

    def _load_settings(self):
        s = self._settings

        # ── Key：读取 + 兼容迁移 ──
        plain = decrypt_secret(s.value("keys", "")) if s.value("keys", "") else ""
        if not plain:
            # 兼容1: 旧格式 api_key_b64 → keys
            legacy_b64 = s.value("api_key_b64", "")
            if legacy_b64:
                try:
                    import base64 as _b64
                    plain = _b64.b64decode(legacy_b64).decode()
                except Exception:
                    plain = ""
                if plain:
                    s.setValue("keys", encrypt_secret(plain))
                s.remove("api_key_b64")
        if not plain:
            # 兼容2: 一次性迁移旧工具（KimiClassifier/Config）的 DeepSeek Key
            old = self._legacy_settings if self._legacy_settings is not None else QSettings(LEGACY_ORG, "Config")
            old.beginGroup("keys")
            old_blob = old.value(SERVICE, "")
            old.endGroup()
            if old_blob:
                plain = decrypt_secret(old_blob)
                if plain:
                    s.setValue("keys", encrypt_secret(plain))

        self._key_input.setText(plain or "")

        # ── 其余配置 ──
        self._src_input.setText(s.value("source_dir", ""))
        self._out_input.setText(s.value("output_dir", ""))
        self._cat_input.setText(s.value("categories_raw", DEFAULT_CATEGORIES))
        self._prompt_input.setPlainText(s.value("global_prompt", DEFAULT_GLOBAL_PROMPT))
        rpm = int(s.value("rpm", DEFAULT_RPM))
        self._rpm_slider.setValue(rpm)
        self._rpm_spin.setValue(rpm)
        self._conc_spin.setValue(int(s.value("concurrency", DEFAULT_CONCURRENCY)))
        self._original_check.setChecked(_as_bool(s.value("use_original", False)))
        self._low_conf_spin.setValue(float(s.value("low_conf_threshold", 0.6)))
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
            service=SERVICE,
            api_key=self.get_api_key(),
            model=get_provider(SERVICE)["default_model"],
            source_dir=self.get_source_dir() or str(Path(path).parent),
            output_dir=self.get_output_dir() or str(Path(path).parent),
            categories=self.get_categories(),
            global_prompt=self.get_global_prompt(),
            category_keywords=self.get_category_keywords(),
            rpm=self.get_rpm(),
            use_original=self.get_use_original(),
            low_conf_threshold=self.get_low_conf(),
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
