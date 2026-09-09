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
