"""布局回归测试 — 防止容器未挂载导致整页空白"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QSplitter

from app.main_window import MainWindow


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


def test_every_tab_has_content_and_splitter(qapp, iso_settings):
    """每个页面的根布局必须有子项，且包含左右分栏"""
    w = MainWindow(settings=_settings())
    assert w._tabs.count() == 3
    for i in range(w._tabs.count()):
        page = w._tabs.widget(i)
        name = w._tabs.tabText(i)
        layout = page.layout()
        assert layout is not None, f"{name} 页无根布局"
        assert layout.count() > 0, f"{name} 页根布局为空"
        assert page.findChild(QSplitter) is not None, f"{name} 页缺少左右分栏"
    w.close()


def test_run_tab_widgets_attached_to_page(qapp, iso_settings):
    """运行页的关键控件必须挂在页面控件树下（回归：曾因容器未挂载导致空白）"""
    w = MainWindow(settings=_settings())
    run_page = w._tabs.widget(1)
    run_tab = w._run_tab
    widgets = {
        "进度条": run_tab._progress_bar,
        "开始按钮": run_tab._start_btn,
        "日志区": run_tab._log,
        "统计表": run_tab._stats_table,
        "速度标签": run_tab._speed_label,
        "Token标签": run_tab._token_label,
        "打开输出按钮": run_tab._open_btn,
    }
    for label, widget in widgets.items():
        assert run_page.isAncestorOf(widget), f"运行页 {label} 未挂载到页面"
    w.close()


def test_config_and_eval_widgets_attached(qapp, iso_settings):
    w = MainWindow(settings=_settings())
    config_page = w._tabs.widget(0)
    eval_page = w._tabs.widget(2)
    assert config_page.isAncestorOf(w._config_tab._key_input)
    assert config_page.isAncestorOf(w._config_tab._prompt_input)
    assert eval_page.isAncestorOf(w._eval_tab._dataset_input)
    assert eval_page.isAncestorOf(w._eval_tab._error_list)
    w.close()
