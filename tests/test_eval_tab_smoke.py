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
    assert tab._category_keywords == {}


def test_eval_rpm_is_independent_and_configurable(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    assert tab._eval_rpm_spin.value() == 120
    assert tab._eval_rpm_spin.maximum() == 600


def test_set_config_stores_keywords(qapp, tmp_path):
    """回归：评估页必须接收分类关键词，否则分类定义为空"""
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    tab.set_config("key", "m", ["科技"], "定义：\n{category_definitions}", 30, False,
                   {"科技": "芯片;CPU"})
    assert tab._category_keywords == {"科技": "芯片;CPU"}


def test_history_table_loads_runs(qapp, tmp_path):
    store = EvalStore(base_dir=tmp_path / "eval")
    store.save_run({"id": "20260909-120000", "timestamp": "2026-09-09 12:00:00",
                    "prompt_hash": "abcd1234", "accuracy": 55.0, "total": 20,
                    "avg_confidence": 0.8, "confusion": {}, "errors": []})
    tab = EvalTab(store=store)
    assert tab._history_table.rowCount() == 1
    assert tab._history_table.item(0, 5).text().startswith("55.0")
    assert tab._history_table.item(0, 3).text() == "abcd1234"


def _record(rid="20260909-130000", variant="现状", rnd=1, samples=None):
    return {
        "id": rid, "timestamp": "2026-09-09 13:00:00", "variant": variant, "round": rnd,
        "accuracy": 50.0, "correct": 1, "total": 2, "avg_confidence": 0.7,
        "elapsed_seconds": 10.0, "total_tokens": 500, "prompt_hash": "deadbeef",
        "confusion": {"科技": {"科技": 1, "日常": 1}},
        "samples": samples if samples is not None else [
            {"path": "/d/科技/a.jpg", "gt": "科技", "pred": "科技", "conf": 0.9, "ok": True},
            {"path": "/d/科技/b.jpg", "gt": "科技", "pred": "日常", "conf": 0.4, "ok": False},
        ],
        "errors": [{"path": "/d/科技/b.jpg", "gt": "科技", "pred": "日常", "conf": 0.4}],
    }


def test_render_batch_populates_views(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    tab._on_finished([_record()])
    assert "50.0%" in tab._metrics_label.text()
    assert tab._confusion_table.rowCount() == 1
    assert tab._error_list.count() == 1
    assert tab._history_table.rowCount() == 1
    assert tab._start_btn.isEnabled() is True
    assert tab._threshold_table.rowCount() > 0


def test_multi_round_batch_renders_stability(qapp, tmp_path):
    """多轮批次要在界面上给出均值/极差与一致率，而不是只显示最后一轮"""
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    r1 = _record("b-r1", rnd=1)
    r2 = _record("b-r2", rnd=2)
    r2["samples"][1] = {"path": "/d/科技/b.jpg", "gt": "科技", "pred": "科技",
                        "conf": 0.8, "ok": True}
    r2["accuracy"] = 100.0
    tab._on_finished([r1, r2])
    assert "2 轮均值" in tab._metrics_label.text()
    assert "极差" in tab._metrics_label.text()
    assert "跨轮一致" in tab._summary_label.text()
    assert tab._history_table.rowCount() == 2


def test_variant_combo_switches_view(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    a = _record("b-A-r1", variant="A")
    b = _record("b-B-r1", variant="B")
    b["accuracy"] = 100.0
    tab._on_finished([a, b])
    assert tab._variant_combo.count() == 2
    assert tab._variant_combo.currentText() == "A"
    tab._variant_combo.setCurrentIndex(1)
    assert "100.0%" in tab._metrics_label.text()


def test_two_variants_show_paired_comparison(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    a = _record("b-A-r1", variant="A")
    b = _record("b-B-r1", variant="B")
    b["samples"] = [
        {"path": "/d/科技/a.jpg", "gt": "科技", "pred": "科技", "conf": 0.9, "ok": True},
        {"path": "/d/科技/b.jpg", "gt": "科技", "pred": "科技", "conf": 0.9, "ok": True},
    ]
    b["accuracy"] = 100.0
    tab._on_finished([a, b])
    assert "配对比较" in tab._summary_label.text()


def test_plan_label_counts_requests(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    tab._rounds_spin.setValue(3)
    tab._update_plan_label()
    assert "1 个变体 × 3 轮" in tab._plan_label.text()
    tab._ab_check.setChecked(True)
    assert "2 个变体 × 3 轮" in tab._plan_label.text()
    tab._fast_check.setChecked(True)
    assert "4 个变体 × 3 轮" in tab._plan_label.text()


def test_build_variants_respects_toggles(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    tab.set_config("k", "m", ["科技"], "P1", 30, False, {"科技": "芯片"})
    assert [v.name for v in tab.build_variants()] == ["现状"]

    tab._ab_check.setChecked(True)
    tab._prompt_edit_b.setPlainText("P2")
    names = [v.name for v in tab.build_variants()]
    assert names == ["A", "B"]
    assert "P2" in tab.build_variants()[1].prompt_text

    tab._fast_check.setChecked(True)
    variants = tab.build_variants()
    assert [v.name for v in variants] == ["A", "B", "A+快速", "B+快速"]
    assert variants[2].extra_body == {"thinking": {"type": "disabled"}}
    assert variants[3].extra_body == {"thinking": {"type": "disabled"}}
    assert variants[0].prompt_text == variants[2].prompt_text


def test_cancel_state_after_empty_record(qapp, tmp_path):
    tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
    tab._on_finished([])
    assert "未完成" in tab._metrics_label.text()
    assert tab._history_table.rowCount() == 0


class TestCategoryMappingWiring:
    """P1：评估页必须拿到并透传类别归并映射"""

    def test_set_config_stores_mapping(self, qapp, tmp_path):
        tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
        tab.set_config("k", "m", ["科技", "史政"], "p", 30, False,
                       {"史政": "..."}, 3, category_mapping={"历史": "史政", "动漫": None})
        assert tab._category_mapping == {"历史": "史政", "动漫": None}

    def test_mapping_defaults_to_empty(self, qapp, tmp_path):
        tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
        assert tab._category_mapping == {}

    def test_plan_label_shows_merging(self, qapp, tmp_path):
        tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
        tab.set_config("k", "m", ["史政"], "p", 30, False, None, 3,
                       category_mapping={"历史": "史政", "动漫": None})
        text = tab._plan_label.text()
        assert "类别归并" in text
        assert "历史→史政" in text and "动漫→排除" in text

    def test_plan_label_hides_merging_when_empty(self, qapp, tmp_path):
        tab = EvalTab(store=EvalStore(base_dir=tmp_path / "eval"))
        tab.set_config("k", "m", ["科技"], "p", 30, False)
        assert "类别归并" not in tab._plan_label.text()
