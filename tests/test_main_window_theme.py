"""主窗口主题切换测试（offscreen，QSettings 隔离）"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import build_qss


@pytest.fixture(autouse=True)
def iso_settings(tmp_path):
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    return tmp_path


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _settings():
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                     "DeepSeekImageClassifier", "Config")


def test_theme_button_exists_with_default_light(qapp, iso_settings):
    w = MainWindow(settings=_settings())
    assert w._theme == "light"
    assert "夜间" in w._theme_btn.text()
    w.close()


def test_toggle_switches_theme_and_persists(qapp, iso_settings):
    w = MainWindow(settings=_settings())
    w._theme_btn.click()
    assert w._theme == "dark"
    assert "日间" in w._theme_btn.text()
    assert qapp.styleSheet() == build_qss("dark")
    assert _settings().value("theme") == "dark"

    # 重新打开应保持夜间
    w2 = MainWindow(settings=_settings())
    assert w2._theme == "dark"
    w.close()
    w2.close()
    qapp.setStyleSheet("")


def test_apply_current_theme_sets_stylesheet(qapp, iso_settings):
    s = _settings()
    s.setValue("theme", "dark")
    s.sync()
    w = MainWindow(settings=s)
    w.apply_current_theme()
    assert qapp.styleSheet() == build_qss("dark")
    w.close()
    qapp.setStyleSheet("")


def test_tabs_use_splitter_layout(qapp, iso_settings):
    """回归：三个页面应为左右分栏（QSplitter）"""
    from PySide6.QtWidgets import QSplitter
    w = MainWindow(settings=_settings())
    config_tab = w._tabs.widget(0)
    eval_tab = w._tabs.widget(2)
    assert config_tab.findChild(QSplitter) is not None
    assert eval_tab.findChild(QSplitter) is not None
    w.close()
