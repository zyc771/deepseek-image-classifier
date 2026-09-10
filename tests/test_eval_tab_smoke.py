"""评估页冒烟测试（offscreen）"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication
from app.eval_tab import EvalTab
from app.eval_store import EvalStore


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_defaults(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    assert tab._per_cat_spin.value() == 15
    assert tab._fixed_check.isChecked() is True
    assert tab._full_check.isChecked() is False
    assert tab._start_btn.isEnabled() is True
    assert tab._cancel_btn.isEnabled() is False


def test_set_config_fills_prompt(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    tab.set_config("key", "model-x", ["科技", "日常"], "提示词{category_definitions}", 30, False)
    assert "提示词" in tab._prompt_edit.toPlainText()
    assert tab._categories == ["科技", "日常"]
    assert tab._model == "model-x"


def test_history_table_loads_runs(qapp, tmp_path):
    store = EvalStore(base_dir=tmp_path / "eval")
    store.save_run({"id": "20260909-120000", "timestamp": "2026-09-09 12:00:00",
                    "prompt_hash": "abcd1234", "accuracy": 55.0, "total": 20,
                    "avg_confidence": 0.8, "confusion": {}, "errors": []})
    tab = EvalTab(store=store)
    assert tab._history_table.rowCount() == 1
    assert tab._history_table.item(0, 3).text().startswith("55.0")
    assert tab._history_table.item(0, 1).text() == "abcd1234"


def test_render_record_populates_views(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    record = {
        "id": "20260909-130000", "timestamp": "2026-09-09 13:00:00",
        "accuracy": 50.0, "correct": 1, "total": 2, "avg_confidence": 0.7,
        "elapsed_seconds": 10.0, "total_tokens": 500, "prompt_hash": "deadbeef",
        "confusion": {"科技": {"科技": 1, "日常": 1}},
        "errors": [{"path": str(tmp_path / "missing.jpg"), "gt": "科技",
                    "pred": "日常", "conf": 0.6}],
    }
    tab._on_finished(record)
    assert "50.0%" in tab._metrics_label.text()
    assert tab._confusion_table.rowCount() == 1
    assert tab._error_list.count() == 1
    assert tab._history_table.rowCount() == 1
    assert tab._start_btn.isEnabled() is True


def test_cancel_state_after_empty_record(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    tab._on_finished({})
    assert "未完成" in tab._metrics_label.text()
    assert tab._history_table.rowCount() == 0
