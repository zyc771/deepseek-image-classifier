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

    def test_overwrite_backs_up_previous(self, tmp_path):
        """重建固定评估集不得无声丢弃旧清单（跨版本可比性依赖它）"""
        s = _store(tmp_path)
        s.save_eval_set("ds", ["old1.jpg", "old2.jpg"])
        s.save_eval_set("ds", ["new.jpg"])
        assert s.load_eval_set("ds") == ["new.jpg"]
        bak = s.data_dir() / "eval_set_ds.bak.json"
        assert bak.exists()
        assert "old1.jpg" in bak.read_text(encoding="utf-8")

    def test_first_save_makes_no_backup(self, tmp_path):
        s = _store(tmp_path)
        s.save_eval_set("fresh", ["a.jpg"])
        assert not (s.data_dir() / "eval_set_fresh.bak.json").exists()


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


class TestBatches:
    def _rec(self, rid, batch_id=None, variant="A", rnd=1, seq=0, acc=50.0):
        r = {
            "id": rid, "timestamp": "2026-09-09 12:00:00", "accuracy": acc,
            "total": 10, "correct": 5, "confusion": {}, "errors": [],
            "variant": variant, "round": rnd, "batch_seq": seq,
        }
        if batch_id:
            r["batch_id"] = batch_id
        return r

    def test_groups_by_batch_id(self, tmp_path):
        s = _store(tmp_path)
        s.save_run(self._rec("b1-A-r1", "b1", "A", 1, 0))
        s.save_run(self._rec("b1-B-r1", "b1", "B", 1, 1))
        s.save_run(self._rec("b1-A-r2", "b1", "A", 2, 2))
        groups = s.load_batches()
        assert list(groups) == ["b1"]
        assert [r["id"] for r in groups["b1"]] == ["b1-A-r1", "b1-B-r1", "b1-A-r2"]

    def test_legacy_runs_grouped_individually(self, tmp_path):
        """旧的单轮记录没有 batch_id，应各自独立成组，不互相干扰"""
        s = _store(tmp_path)
        s.save_run(self._rec("old1"))
        s.save_run(self._rec("old2"))
        groups = s.load_batches()
        assert set(groups) == {"old1", "old2"}

    def test_newest_batch_first(self, tmp_path):
        s = _store(tmp_path)
        s.save_run(self._rec("b20260101-000000-A-r1", "b20260101-000000"))
        s.save_run(self._rec("b20260202-000000-A-r1", "b20260202-000000"))
        assert list(s.load_batches())[0] == "b20260202-000000"

    def test_save_batch_writes_all(self, tmp_path):
        s = _store(tmp_path)
        paths = s.save_batch([self._rec("x1", "bx"), self._rec("x2", "bx")])
        assert len(paths) == 2 and len(s.load_runs()) == 2
