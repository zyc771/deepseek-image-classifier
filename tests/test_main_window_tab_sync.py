"""主窗口 ↔ 页面配置同步契约测试

锁定切页时的参数传递：历史上 run_tab 的解包顺序错过位，这里补上端到端覆盖。
用 INI 文件承载 QSettings，避免污染真实注册表。
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _settings(tmp_path):
    return QSettings(str(tmp_path / "cfg.ini"), QSettings.Format.IniFormat)


def _window(qapp, tmp_path):
    w = MainWindow(settings=_settings(tmp_path))
    w._config_tab.set_global_prompt("提示词{category_definitions}")
    return w


class TestTabSync:
    def test_switch_to_run_tab_passes_all_fields(self, qapp, tmp_path):
        w = _window(qapp, tmp_path)
        w._config_tab._cat_input.setText("科技;日常")
        w._config_tab._fast_check.setChecked(True)
        w._tabs.setCurrentIndex(1)
        cfg = w._run_tab._config
        assert len(cfg) == 11                 # 元组长度锁定，防止两侧解包错位
        (api_key, src, out, cats, prompt, kws, rpm,
         use_original, low_conf, concurrency, fast_mode) = cfg
        assert cats == ["科技", "日常"]
        assert prompt == "提示词{category_definitions}"
        # 已知分类会自动带上默认关键词（否则提示词里的分类定义为空）
        assert kws == {"科技": "电脑;手机;芯片;AI;数码产品",
                       "日常": "美食;宠物;聊天记录;自拍;家庭"}
        assert (src, out) == ("", "")
        assert rpm == w._config_tab.get_rpm()
        assert use_original is False
        assert low_conf == 0.6
        assert concurrency == 3
        assert fast_mode is True          # 快速模式开关一路传到运行页

    def test_switch_to_eval_tab_syncs_prompt_and_categories(self, qapp, tmp_path):
        w = _window(qapp, tmp_path)
        w._config_tab._cat_input.setText("科技;日常")
        w._config_tab._kw_row_cache = None
        w._tabs.setCurrentIndex(2)
        assert w._eval_tab._categories == ["科技", "日常"]
        assert "提示词" in w._eval_tab._prompt_edit.toPlainText()

    def test_threshold_hint_reads_config_tab(self, qapp, tmp_path):
        """评估页的建议阈值提示要能读到配置页当前阈值"""
        w = _window(qapp, tmp_path)
        assert w._eval_tab._low_conf_value() == w._config_tab.get_low_conf()

    def test_switching_tabs_repeatedly_is_stable(self, qapp, tmp_path):
        w = _window(qapp, tmp_path)
        for idx in (1, 2, 1, 2, 0, 1):
            w._tabs.setCurrentIndex(idx)
        assert w._tabs.currentIndex() == 1
