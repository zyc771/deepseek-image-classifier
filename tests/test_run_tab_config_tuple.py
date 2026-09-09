"""RunTab.set_config 参数顺序锁定 — 防止 main_window 与 run_tab 解包错位"""
from PySide6.QtWidgets import QApplication
from app.run_tab import RunTab


def test_set_config_unpack_order():
    app = QApplication.instance() or QApplication([])
    tab = RunTab()
    tab.set_config(
        "key", "/src", "/out",
        ["科技"], "prompt {categories}", {"科技": "芯片"}, 42,
        use_original=True, low_conf=0.7,
    )
    (api_key, src, out, cats, prompt, kws, rpm,
     use_original, low_conf) = tab._config
    assert api_key == "key"
    assert src == "/src"
    assert out == "/out"
    assert cats == ["科技"]
    assert prompt == "prompt {categories}"
    assert kws == {"科技": "芯片"}
    assert rpm == 42
    assert use_original is True
    assert low_conf == 0.7


def test_set_config_defaults():
    app = QApplication.instance() or QApplication([])
    tab = RunTab()
    tab.set_config("key", "/src", "/out", ["科技"], "p", {}, 30)
    _, _, _, _, _, _, _, use_original, low_conf = tab._config
    assert use_original is False
    assert low_conf == 0.6
