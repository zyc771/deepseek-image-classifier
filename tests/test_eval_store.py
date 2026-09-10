"""EvalStore 持久化测试"""
from app.eval_store import EvalStore


def _store(tmp_path):
    return EvalStore(base_dir=tmp_path / "eval")


class TestEvalSet:
    def test_save_and_load_roundtrip(self, tmp_path):
        s = _store(tmp_path)
        s.save_eval_set("datasetA", ["a/1.jpg", "a/2.jpg"])
        assert s.load_eval_set("datasetA") == ["a/1.jpg", "a/2.jpg"]

    def test_load_missing_returns_none(self, tmp_path):
        assert _store(tmp_path).load_eval_set("nope") is None

    def test_unsafe_name_sanitized(self, tmp_path):
        s = _store(tmp_path)
        s.save_eval_set("my set/../x", ["p.jpg"])
        assert s.load_eval_set("my set/../x") == ["p.jpg"]


class TestRuns:
    def _record(self, run_id="20260909-120000", acc=50.0):
        return {
            "id": run_id, "timestamp": "2026-09-09 12:00:00",
            "accuracy": acc, "total": 10, "correct": 5,
            "confusion": {"科技": {"科技": 5, "日常": 5}},
            "errors": [{"path": "x.jpg", "gt": "科技", "pred": "日常", "conf": 0.5}],
        }

    def test_save_and_load_run(self, tmp_path):
        s = _store(tmp_path)
        s.save_run(self._record())
        runs = s.load_runs()
        assert len(runs) == 1
        assert runs[0]["accuracy"] == 50.0
        assert s.load_run("20260909-120000")["total"] == 10

    def test_runs_sorted_desc(self, tmp_path):
        s = _store(tmp_path)
        s.save_run(self._record("20260909-110000", 40.0))
        s.save_run(self._record("20260909-130000", 60.0))
        runs = s.load_runs()
        assert [r["accuracy"] for r in runs] == [60.0, 40.0]

    def test_corrupted_run_skipped(self, tmp_path):
        s = _store(tmp_path)
        s.save_run(self._record())
        bad = s.data_dir() / "run_broken.json"
        bad.write_text("{ not json", encoding="utf-8")
        assert len(s.load_runs()) == 1

    def test_load_missing_run_returns_none(self, tmp_path):
        assert _store(tmp_path).load_run("nope") is None
