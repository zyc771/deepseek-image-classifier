"""双主题（浅色 / 夜间）QSS 主题系统"""
from PySide6.QtCore import QSettings

THEME_SETTINGS_KEY = "theme"
DEFAULT_THEME = "light"

# ── 配色（与 docs/design/ui-preview.html 保持一致）──
LIGHT = {
    "bg": "#eef1f6", "panel": "#ffffff", "panel2": "#f4f6fa", "panel3": "#e9edf4",
    "border": "#dde3ec", "border_strong": "#c9d2e0",
    "text": "#1e2430", "text_dim": "#69748a",
    "primary": "#2f7cf6", "primary_hover": "#4a8ef8", "primary_soft": "#e8f0fe",
    "primary_text": "#ffffff",
    "danger": "#e05252", "warn": "#d98a1f", "ok": "#22a06b",
    "table_head": "#f2f5fa", "row_alt": "#fafbfd", "log_bg": "#f7f9fc",
    "scrollbar": "#c9d2e0", "scrollbar_hover": "#aab6c8",
}

DARK = {
    "bg": "#14171f", "panel": "#1c2029", "panel2": "#242934", "panel3": "#2b313d",
    "border": "#313846", "border_strong": "#3d4557",
    "text": "#e7eaf2", "text_dim": "#9ba4b8",
    "primary": "#4b8ef7", "primary_hover": "#5f9cf8", "primary_soft": "#1e2b45",
    "primary_text": "#ffffff",
    "danger": "#ef5f5f", "warn": "#e0a24a", "ok": "#35c08a",
    "table_head": "#242934", "row_alt": "#1f242e", "log_bg": "#14171f",
    "scrollbar": "#3d4557", "scrollbar_hover": "#4d566b",
}

PALETTES = {"light": LIGHT, "dark": DARK}

_QSS = """
/* ── 基础 ── */
QWidget { background-color: %(bg)s; color: %(text)s;
          font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif; font-size: 13px; }
QMainWindow, QDialog { background-color: %(bg)s; }
QLabel { background: transparent; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }

/* ── 分组卡片 ── */
QGroupBox {
    background-color: %(panel)s; border: 1px solid %(border)s; border-radius: 8px;
    margin-top: 15px; padding: 14px 12px 10px 12px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: %(primary)s; font-weight: 600; background-color: %(panel)s;
}

/* ── 输入类 ── */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: %(panel)s; color: %(text)s;
    border: 1px solid %(border_strong)s; border-radius: 6px; padding: 5px 8px;
    selection-background-color: %(primary)s; selection-color: %(primary_text)s;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid %(primary)s; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: %(panel)s; color: %(text)s;
    border: 1px solid %(border_strong)s; selection-background-color: %(primary_soft)s;
    selection-color: %(text)s;
}

/* ── 按钮 ── */
QPushButton {
    background-color: %(panel)s; color: %(text)s;
    border: 1px solid %(border_strong)s; border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover { border-color: %(primary)s; color: %(primary)s; background-color: %(primary_soft)s; }
QPushButton:pressed { background-color: %(primary_soft)s; border-color: %(primary)s; }
QPushButton:disabled { color: %(text_dim)s; background-color: %(panel2)s; border-color: %(border)s; }
QPushButton#primary {
    background-color: %(primary)s; border-color: %(primary)s;
    color: %(primary_text)s; font-weight: 600;
}
QPushButton#primary:hover { background-color: %(primary_hover)s; border-color: %(primary_hover)s; color: %(primary_text)s; }
QPushButton#primary:disabled { background-color: %(panel3)s; border-color: %(border)s; color: %(text_dim)s; }

/* ── 进度条 ── */
QProgressBar {
    background-color: %(panel3)s; border: none; border-radius: 5px;
    height: 10px; text-align: center; color: %(text_dim)s;
}
QProgressBar::chunk { background-color: %(primary)s; border-radius: 5px; }

/* ── Tab ── */
QTabWidget::pane { border: 1px solid %(border)s; background-color: %(panel)s; top: -1px; }
QTabBar::tab {
    background: %(panel2)s; color: %(text_dim)s; padding: 8px 22px;
    border: 1px solid %(border)s; border-bottom: none;
    border-top-left-radius: 7px; border-top-right-radius: 7px; margin-right: 2px;
}
QTabBar::tab:selected { background: %(panel)s; color: %(text)s; font-weight: 600; }
QTabBar::tab:hover { color: %(primary)s; }

/* ── 表格 / 列表 ── */
QTableWidget, QTableView {
    background-color: %(panel)s; alternate-background-color: %(row_alt)s;
    gridline-color: %(border)s; border: 1px solid %(border)s; border-radius: 6px;
    selection-background-color: %(primary_soft)s; selection-color: %(text)s;
}
QHeaderView::section {
    background-color: %(table_head)s; color: %(text_dim)s; padding: 6px 8px;
    border: none; border-bottom: 1px solid %(border)s; border-right: 1px solid %(border)s;
    font-weight: 600;
}
QTableCornerButton::section { background-color: %(table_head)s; border: none; }
QListWidget {
    background-color: %(panel)s; border: 1px solid %(border)s; border-radius: 6px;
    selection-background-color: %(primary_soft)s; selection-color: %(text)s;
}
QListWidget::item { border-radius: 6px; padding: 3px; }
QListWidget::item:hover { background-color: %(panel2)s; }

/* ── 复选框 / 滑块 ── */
QCheckBox { color: %(text)s; spacing: 7px; background: transparent; }
QCheckBox::indicator {
    width: 15px; height: 15px; border: 1px solid %(border_strong)s;
    border-radius: 4px; background-color: %(panel)s;
}
QCheckBox::indicator:hover { border-color: %(primary)s; }
QCheckBox::indicator:checked { background-color: %(primary)s; border-color: %(primary)s; }
QSlider::groove:horizontal { height: 4px; background: %(panel3)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(primary)s; border-radius: 2px; }
QSlider::handle:horizontal {
    background: %(primary)s; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
}

/* ── 分隔条（左右分栏） ── */
QSplitter::handle { background-color: %(border)s; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical { height: 2px; }
QSplitter::handle:hover { background-color: %(primary)s; }

/* ── 滚动条 ── */
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: %(scrollbar)s; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: %(scrollbar_hover)s; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: %(scrollbar)s; border-radius: 5px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: %(scrollbar_hover)s; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ── 状态栏 ── */
QStatusBar { background-color: %(panel2)s; color: %(text_dim)s; border-top: 1px solid %(border)s; }
QStatusBar QLabel { color: %(text_dim)s; }

QToolTip {
    background-color: %(panel)s; color: %(text)s;
    border: 1px solid %(border_strong)s; padding: 4px 6px;
}
"""


def normalize_theme(name: str | None) -> str:
    """把任意输入规范为主题名（未知值回退默认）"""
    return name if name in PALETTES else DEFAULT_THEME


def build_qss(theme: str) -> str:
    """生成指定主题的完整样式表"""
    palette = PALETTES[normalize_theme(theme)]
    return _QSS % palette


def load_theme(settings: QSettings | None = None) -> str:
    """从配置读取主题（默认浅色）"""
    s = settings or QSettings("DeepSeekImageClassifier", "Config")
    return normalize_theme(s.value(THEME_SETTINGS_KEY, DEFAULT_THEME))


def save_theme(theme: str, settings: QSettings | None = None) -> None:
    s = settings or QSettings("DeepSeekImageClassifier", "Config")
    s.setValue(THEME_SETTINGS_KEY, normalize_theme(theme))
    s.sync()


def apply_theme(app, theme: str) -> str:
    """应用主题到 QApplication，返回规范化后的主题名"""
    name = normalize_theme(theme)
    app.setStyleSheet(build_qss(name))
    return name
