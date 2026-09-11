"""主题系统测试"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from app import theme


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def iso_settings(tmp_path):
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    return tmp_path


class TestBuildQss:
    def test_light_and_dark_differ(self):
        light = theme.build_qss("light")
        dark = theme.build_qss("dark")
        assert light != dark
        assert theme.LIGHT["bg"] in light
        assert theme.DARK["bg"] in dark

    def test_no_unresolved_placeholders(self):
        for name in ("light", "dark"):
            qss = theme.build_qss(name)
            assert "%(" not in qss, f"{name} 主题存在未替换的占位符"

    def test_contains_key_selectors(self):
        qss = theme.build_qss("light")
        for selector in ("QGroupBox", "QPushButton#primary", "QProgressBar::chunk",
                         "QHeaderView::section", "QSplitter::handle", "QScrollBar::handle:vertical",
                         "QTabBar::tab:selected", "QCheckBox::indicator:checked"):
            assert selector in qss, f"缺少选择器 {selector}"

    def test_unknown_theme_falls_back_to_light(self):
        assert theme.build_qss("nonsense") == theme.build_qss("light")


class TestNormalize:
    def test_valid_names(self):
        assert theme.normalize_theme("light") == "light"
        assert theme.normalize_theme("dark") == "dark"

    def test_invalid_and_none(self):
        assert theme.normalize_theme("") == "light"
        assert theme.normalize_theme(None) == "light"
        assert theme.normalize_theme("DARK") == "light"  # 大小写不匹配即回退


class TestPersistence:
    def _settings(self):
        return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                         "DeepSeekImageClassifier", "Config")

    def test_default_theme_is_light(self):
        assert theme.load_theme(self._settings()) == "light"

    def test_save_and_load_roundtrip(self):
        theme.save_theme("dark", self._settings())
        assert theme.load_theme(self._settings()) == "dark"

    def test_apply_theme_sets_stylesheet(self, qapp):
        name = theme.apply_theme(qapp, "dark")
        assert name == "dark"
        assert qapp.styleSheet() == theme.build_qss("dark")

        name = theme.apply_theme(qapp, "light")
        assert name == "light"
        assert qapp.styleSheet() == theme.build_qss("light")
        qapp.setStyleSheet("")  # 清理，避免影响其它测试
