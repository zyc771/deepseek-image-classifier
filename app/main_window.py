"""主窗口 — 组装配置页和运行页"""
from PySide6.QtWidgets import QMainWindow, QTabWidget, QStatusBar, QLabel
from app.config_tab import ConfigTab
from app.run_tab import RunTab
from app.providers import get_provider


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSeek图片分类工具")
        self.resize(900, 700)

        # 状态栏
        self._status = QStatusBar()
        self._status_label = QLabel("就绪")
        self._status.addWidget(self._status_label)
        self.setStatusBar(self._status)

        # Tab
        tabs = QTabWidget()
        self._config_tab = ConfigTab()
        self._run_tab = RunTab()
        tabs.addTab(self._config_tab, "配置")
        tabs.addTab(self._run_tab, "运行")
        tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(tabs)
        self._tabs = tabs

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
            )
            self._run_tab.set_config(*cfg)
            self._status_label.setText(
                f"就绪 | 模型: {get_provider('deepseek')['default_model']} | 分类: {len(cfg[3])}类"
            )

    def closeEvent(self, event):
        self._config_tab.save_settings()
        super().closeEvent(event)
