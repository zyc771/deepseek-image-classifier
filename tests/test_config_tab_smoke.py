"""ConfigTab 服务商联动/存储/迁移冒烟测试（offscreen，QSettings 隔离到 tmp ini）"""
import os
import base64

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from app.config_tab import ConfigTab
from app.providers import get_provider


@pytest.fixture(autouse=True)
def iso_settings(tmp_path):
    """将 QSettings 隔离到临时 ini，避免污染注册表"""
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    return tmp_path


def _make_settings():
    """显式 IniFormat 构造（PySide6 的 (org, app) 重载不遵循 setDefaultFormat）"""
    return QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, "KimiClassifier", "Config"
    )


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_default_service_and_model(qapp, iso_settings):
    tab = ConfigTab(settings=_make_settings())
    assert tab.get_service() == "deepseek"
    assert tab._service_combo.count() == 2
    assert tab._model_input.text() == get_provider("deepseek")["default_model"]


def test_switch_service_and_persistence(qapp, iso_settings):
    tab = ConfigTab(settings=_make_settings())
    tab._key_input.setText("sk-deep-1")
    tab._model_input.setText("deep-model-1")
    tab.save_settings()

    # 切到 kimi
    tab._service_combo.setCurrentIndex(tab._service_combo.findData("kimi"))
    assert tab.get_service() == "kimi"
    assert tab._model_input.text() == get_provider("kimi")["default_model"]
    assert tab._key_input.text() == ""
    tab._key_input.setText("sk-kimi-1")
    tab.save_settings()

    # ini 中为 DPAPI 加密 blob，非明文
    s = _make_settings()
    s.beginGroup("keys")
    blob = s.value("deepseek", "")
    s.endGroup()
    assert blob.startswith("dpapi:")
    assert "sk-deep-1" not in blob

    # 重载：记忆 kimi 当前值，切回 deepseek 恢复
    tab2 = ConfigTab(settings=_make_settings())
    assert tab2.get_service() == "kimi"
    assert tab2.get_api_key() == "sk-kimi-1"
    tab2._service_combo.setCurrentIndex(tab2._service_combo.findData("deepseek"))
    assert tab2.get_api_key() == "sk-deep-1"
    assert tab2.get_model() == "deep-model-1"


def test_legacy_api_key_migration(qapp, iso_settings):
    s = _make_settings()
    s.setValue("api_key_b64", base64.b64encode(b"sk-legacy-key").decode())
    s.sync()
    tab = ConfigTab(settings=_make_settings())
    assert tab._saved_keys.get("kimi") == "sk-legacy-key"
    s2 = _make_settings()
    assert not s2.value("api_key_b64")
