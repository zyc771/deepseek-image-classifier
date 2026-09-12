"""主窗口 — 组装配置页、运行页与评估页"""
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QStatusBar, QLabel, QPushButton

from app.config_tab import ConfigTab
from app.run_tab import RunTab
from app.eval_tab import EvalTab
from app.providers import get_provider
from app.theme import apply_theme, load_theme, save_theme


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings = None):
        super().__init__()
        self.setWindowTitle("DeepSeek图片分类工具")
        self.resize(1040, 720)
        self._settings = settings or QSettings("DeepSeekImageClassifier", "Config")
        self._theme = load_theme(self._settings)

        # 状态栏（含主题切换按钮）
        self._status = QStatusBar()
        self._status_label = QLabel("就绪")
        self._status.addWidget(self._status_label)
        self._theme_btn = QPushButton()
        self._theme_btn.setFixedWidth(120)
        self._theme_btn.setToolTip("切换浅色 / 夜间主题（会记住选择）")
        self._theme_btn.clicked.connect(self._toggle_theme)
        self._status.addPermanentWidget(self._theme_btn)
        self.setStatusBar(self._status)
        self._refresh_theme_btn()

        # Tab
        tabs = QTabWidget()
        self._config_tab = ConfigTab(settings=self._settings)
        self._run_tab = RunTab()
        self._eval_tab = EvalTab()
        self._eval_tab.save_to_config = self._config_tab.set_global_prompt
        self._eval_tab.get_current_threshold = self._config_tab.get_low_conf
        tabs.addTab(self._config_tab, "配置")
        tabs.addTab(self._run_tab, "运行")
        tabs.addTab(self._eval_tab, "评估")
        tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(tabs)
        self._tabs = tabs

    # ── 主题 ──
    def _refresh_theme_btn(self):
        if self._theme == "dark":
            self._theme_btn.setText("☀ 日间模式")
        else:
            self._theme_btn.setText("🌙 夜间模式")

    def _toggle_theme(self):
        self._theme = "dark" if self._theme == "light" else "light"
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self._theme)
        save_theme(self._theme, self._settings)
        self._refresh_theme_btn()

    def apply_current_theme(self):
        """启动时调用：应用已保存的主题"""
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self._theme)
        self._refresh_theme_btn()

    def _on_tab_changed(self, index: int):
        if index == 1:
            # 切换到运行页时，从配置页同步配置
            self._config_tab.save_settings()
            cfg = (
                self._config_tab.get_api_key(),
                self._config_tab.get_source_dir(),
                self._config_tab.get_output_dir(),
                self._config_tab.get_categories(),
                self._config_tab.get_global_prompt(),
                self._config_tab.get_category_keywords(),
                self._config_tab.get_rpm(),
                self._config_tab.get_use_original(),
                self._config_tab.get_low_conf(),
                self._config_tab.get_concurrency(),
                self._config_tab.get_fast_mode(),
            )
            self._run_tab.set_config(*cfg)
            self._status_label.setText(
                f"就绪 | 模型: {get_provider('deepseek')['default_model']} | 分类: {len(cfg[3])}类"
            )
        elif index == 2:
            # 切换到评估页时同步配置
            self._config_tab.save_settings()
            self._eval_tab.set_config(
                self._config_tab.get_api_key(),
                get_provider("deepseek")["default_model"],
                self._config_tab.get_categories(),
                self._config_tab.get_global_prompt(),
                self._config_tab.get_rpm(),
                self._config_tab.get_use_original(),
                self._config_tab.get_category_keywords(),
                self._config_tab.get_concurrency(),
                category_mapping=self._config_tab.get_category_mapping(),
            )

    def closeEvent(self, event):
        self._config_tab.save_settings()
        super().closeEvent(event)
