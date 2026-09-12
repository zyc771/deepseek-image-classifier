"""ConfigTab 冒烟测试（offscreen，QSettings 隔离到 tmp ini）

覆盖：单服务商 UI（无下拉/无模型框）、Key 加密持久化、
api_key_b64 兼容迁移、旧工具（KimiClassifier/Config）一次性迁移。
"""
import os
import base64

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from app.config_tab import ConfigTab
from app.secure import encrypt_secret


@pytest.fixture(autouse=True)
def iso_settings(tmp_path):
    """将 QSettings 隔离到临时 ini，避免污染注册表（新命名空间与旧命名空间同隔离）"""
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    return tmp_path


def _new_settings():
    return QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, "DeepSeekImageClassifier", "Config"
    )


def _old_settings():
    return QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, "KimiClassifier", "Config"
    )


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_single_provider_ui(qapp, iso_settings):
    tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    # 无服务商下拉、无模型输入框
    assert not hasattr(tab, "_service_combo")
    assert not hasattr(tab, "_model_input")
    # 隔离环境无任何旧配置，默认无 key
    assert tab.get_api_key() == ""
    # 默认：原图模式关、阈值 0.6
    assert tab.get_use_original() is False
    assert tab.get_low_conf() == 0.6


def test_processing_settings_persistence(qapp, iso_settings):
    tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    tab._original_check.setChecked(True)
    tab._low_conf_spin.setValue(0.75)
    tab.save_settings()
    tab2 = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    assert tab2.get_use_original() is True
    assert tab2.get_low_conf() == 0.75


def test_key_persistence_dpapi(qapp, iso_settings):
    tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    tab._key_input.setText("sk-deepseek-123")
    tab.save_settings()

    s = _new_settings()
    blob = s.value("keys", "")
    assert blob.startswith("dpapi:")
    assert "sk-deepseek-123" not in blob

    tab2 = ConfigTab(settings=_new_settings())
    assert tab2.get_api_key() == "sk-deepseek-123"


def test_api_key_b64_compat_migration(qapp, iso_settings):
    s = _new_settings()
    s.setValue("api_key_b64", base64.b64encode(b"sk-old-b64").decode())
    s.sync()
    tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    assert tab.get_api_key() == "sk-old-b64"
    assert not _new_settings().value("api_key_b64")


def test_legacy_tool_one_time_migration(qapp, iso_settings):
    # 旧工具配置：keys 组下 deepseek 子键 = DPAPI 加密的 DeepSeek Key（与真实 v2 格式一致）
    old = _old_settings()
    old.beginGroup("keys")
    old.setValue("deepseek", encrypt_secret("sk-from-old-tool"))
    old.endGroup()
    old.sync()

    tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    assert tab.get_api_key() == "sk-from-old-tool"
    # 写入新命名空间后，再次启动不会再依赖旧配置（迁移已完成）
    assert _new_settings().value("keys", "").startswith("dpapi:")
    tab2 = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    assert tab2.get_api_key() == "sk-from-old-tool"


def test_use_original_string_false_parsed_correctly(qapp, iso_settings):
    """回归：注册表中存字符串 'false' 时必须解析为 False（bool('false') 是 True 的坑）"""
    s = _new_settings()
    s.setValue("use_original", "false")
    s.sync()
    tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    assert tab.get_use_original() is False


def test_use_original_string_true_parsed_correctly(qapp, iso_settings):
    s = _new_settings()
    s.setValue("use_original", "true")
    s.sync()
    tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    assert tab.get_use_original() is True


def test_speed_settings_ranges_and_persistence(qapp, iso_settings):
    tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    # RPM 上限放开到 600，并发默认 3
    assert tab._rpm_spin.maximum() == 600
    assert tab._rpm_slider.maximum() == 600
    assert tab.get_concurrency() == 3

    tab._rpm_spin.setValue(300)
    tab._conc_spin.setValue(5)
    tab.save_settings()

    tab2 = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
    assert tab2.get_rpm() == 300
    assert tab2.get_concurrency() == 5


class TestEightCategoryDefaults:
    """P2：默认类别体系切到 8 类（删动漫、史政军合并为史政）"""

    def test_default_categories(self, qapp, iso_settings):
        tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
        assert tab.get_categories() == ["科技", "日常", "学习", "体育",
                                        "地理", "游戏", "经济", "史政"]

    def test_removed_categories_absent(self, qapp, iso_settings):
        cats = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings()).get_categories()
        for gone in ("动漫", "历史", "政治", "军事"):
            assert gone not in cats

    def test_merged_keywords_present(self, qapp, iso_settings):
        kws = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings()).get_category_keywords()
        assert "史政" in kws
        assert "老照片" in kws["史政"] and "阅兵" in kws["史政"]

    def test_prompt_has_no_anime_rule(self, qapp, iso_settings):
        prompt = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings()).get_global_prompt()
        assert "动漫" not in prompt
        assert "史政" in prompt

    def test_prompt_keeps_output_format_contract(self, qapp, iso_settings):
        """回归：提示词必须保留 JSON 输出契约与置信度分档"""
        prompt = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings()).get_global_prompt()
        assert '"category"' in prompt and "confidence" in prompt
        assert "{category_definitions}" in prompt


class TestCategoryAlias:
    def test_default_alias_available(self, qapp, iso_settings):
        tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
        assert "史政" in tab.get_category_alias()

    def test_mapping_parsed(self, qapp, iso_settings):
        tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
        m = tab.get_category_mapping()
        assert m["历史"] == "史政" and m["政治"] == "史政" and m["军事"] == "史政"
        assert m["动漫"] is None and m["节假日"] is None and m["黄"] is None

    def test_alias_persists(self, qapp, iso_settings):
        tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
        tab._alias_input.setPlainText("世界 = 地理, 历史")
        tab.save_settings()
        tab2 = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
        assert tab2.get_category_mapping() == {"地理": "世界", "历史": "世界"}

    def test_empty_alias_means_no_merging(self, qapp, iso_settings):
        tab = ConfigTab(settings=_new_settings(), legacy_settings=_old_settings())
        tab._alias_input.setPlainText("")
        assert tab.get_category_mapping() == {}
