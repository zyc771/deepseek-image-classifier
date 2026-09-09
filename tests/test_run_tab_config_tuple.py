"""RunTab.set_config 参数顺序锁定 — 防止 main_window 与 run_tab 解包错位"""
from PySide6.QtWidgets import QApplication
from app.run_tab import RunTab


def test_set_config_unpack_order():
    app = QApplication.instance() or QApplication([])
    tab = RunTab()
    tab.set_config(
        "deepseek", "key", "model-x", "/src", "/out",
        ["科技"], "prompt {categories}", {"科技": "芯片"}, 42,
    )
    service, api_key, model, src, out, cats, prompt, kws, rpm = tab._config
    assert service == "deepseek"
    assert api_key == "key"
    assert model == "model-x"
    assert src == "/src"
    assert out == "/out"
    assert cats == ["科技"]
    assert prompt == "prompt {categories}"
    assert kws == {"科技": "芯片"}
    assert rpm == 42
