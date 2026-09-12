"""多轮 / 多变体编排测试（monkeypatch 掉网络）"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import app.eval_batch as eb
import app.evaluator as ev


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for cat, n in (("科技", 3), ("日常", 2)):
        d = root / cat
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{cat}{i}.jpg").write_bytes(b"x")
    return root


def _samples(tmp_path):
    root = _dataset(tmp_path)
    return [(cat, f) for cat in ("科技", "日常")
            for f in sorted((root / cat).glob("*.jpg"))]


def _cfg(tmp_path, **over):
    base = dict(service="deepseek", api_key="k", model="m", prompt_text="p",
                categories=["科技", "日常"], dataset_root=str(tmp_path),
                rpm=600, concurrency=2)
    base.update(over)
    return ev.RoundConfig(**base)


class TestBuildSchedule:
    def test_interleaves_within_round(self):
        """必须是 A B A B 而不是 A A B B —— 交替才能抵消时间漂移"""
        assert eb.build_schedule(2, 3) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    def test_single_variant_is_sequential_rounds(self):
        assert eb.build_schedule(1, 3) == [(0, 0), (1, 0), (2, 0)]

    def test_degenerate_inputs(self):
        assert eb.build_schedule(0, 3) == []
        assert eb.build_schedule(2, 0) == []


class TestMakeVariants:
    def test_placeholder_substitution(self):
        variants = eb.make_variants(
            {"简洁版": "只输出分类\n{categories}", "详细版": "规则\n{category_definitions}"},
            ["科技", "日常"], {"科技": "芯片;CPU"},
        )
        assert [v.name for v in variants] == ["简洁版", "详细版"]
        assert "科技、日常" in variants[0].prompt_text
        assert "- 科技: 芯片;CPU" in variants[1].prompt_text

    def test_extra_body_attached_by_name(self):
        variants = eb.make_variants(
            {"普通": "p", "快速": "p"}, ["科技"], None,
            {"快速": {"thinking": {"type": "disabled"}}},
        )
        assert variants[0].extra_body is None
        assert variants[1].extra_body == {"thinking": {"type": "disabled"}}

    def test_prompt_hash_stable(self):
        a = eb.make_variants({"x": "p"}, ["科技"])[0]
        b = eb.make_variants({"y": "p"}, ["科技"])[0]
        assert a.prompt_hash() == b.prompt_hash()


class TestRunBatch:
    def _fake(self, monkeypatch):
        seen = []

        def fake(service, api_key, model, path, prompt, use_original=False, extra_body=None):
            seen.append({"prompt": prompt, "extra_body": extra_body})
            return '{"category": "科技", "confidence": 0.9}', 10, 5

        monkeypatch.setattr(ev, "classify_image", fake)
        return seen

    def test_records_count_and_tags(self, tmp_path, monkeypatch):
        self._fake(monkeypatch)
        variants = [eb.Variant("A", "pa"), eb.Variant("B", "pb")]
        recs = eb.run_batch(_samples(tmp_path), variants, 3, _cfg(tmp_path))
        assert len(recs) == 6                                  # 2 变体 × 3 轮
        assert [r["variant"] for r in recs] == ["A", "B"] * 3   # 交替
        assert [r["round"] for r in recs] == [1, 1, 2, 2, 3, 3]
        assert len({r["id"] for r in recs}) == 6                # id 唯一
        assert all(r["total"] == 5 for r in recs)
        assert all(r["rounds_total"] == 3 for r in recs)

    def test_each_variant_uses_its_own_prompt(self, tmp_path, monkeypatch):
        seen = self._fake(monkeypatch)
        variants = [eb.Variant("A", "pa"), eb.Variant("B", "pb")]
        eb.run_batch(_samples(tmp_path), variants, 1, _cfg(tmp_path))
        assert {s["prompt"] for s in seen} == {"pa", "pb"}

    def test_extra_body_reaches_api_per_variant(self, tmp_path, monkeypatch):
        seen = self._fake(monkeypatch)
        variants = [eb.Variant("普通", "p"),
                    eb.Variant("快速", "p", {"thinking": {"type": "disabled"}})]
        eb.run_batch(_samples(tmp_path), variants, 1, _cfg(tmp_path))
        fast = [s for s in seen if s["extra_body"]]
        slow = [s for s in seen if not s["extra_body"]]
        assert len(fast) == 5 and len(slow) == 5

    def test_records_carry_prompt_hash_matching_variant(self, tmp_path, monkeypatch):
        self._fake(monkeypatch)
        variants = [eb.Variant("A", "pa"), eb.Variant("B", "pb")]
        recs = eb.run_batch(_samples(tmp_path), variants, 1, _cfg(tmp_path))
        assert recs[0]["prompt_hash"] == variants[0].prompt_hash()
        assert recs[1]["prompt_hash"] == variants[1].prompt_hash()

    def test_samples_saved_for_stats(self, tmp_path, monkeypatch):
        self._fake(monkeypatch)
        recs = eb.run_batch(_samples(tmp_path), [eb.Variant("A", "p")], 1, _cfg(tmp_path))
        assert len(recs[0]["samples"]) == 5

    def test_cancel_returns_empty(self, tmp_path, monkeypatch):
        self._fake(monkeypatch)
        recs = eb.run_batch(_samples(tmp_path), [eb.Variant("A", "p")], 2,
                            _cfg(tmp_path), should_cancel=lambda: True)
        assert recs == []

    def test_empty_inputs(self, tmp_path, monkeypatch):
        self._fake(monkeypatch)
        assert eb.run_batch([], [eb.Variant("A", "p")], 1, _cfg(tmp_path)) == []
        assert eb.run_batch(_samples(tmp_path), [], 1, _cfg(tmp_path)) == []
        assert eb.run_batch(_samples(tmp_path), [eb.Variant("A", "p")], 0, _cfg(tmp_path)) == []

    def test_progress_reports_across_whole_batch(self, tmp_path, monkeypatch):
        self._fake(monkeypatch)
        seen = []
        eb.run_batch(_samples(tmp_path), [eb.Variant("A", "p"), eb.Variant("B", "p")], 2,
                     _cfg(tmp_path), on_progress=lambda *a: seen.append(a))
        assert len(seen) == 20                 # 5 张 × 4 轮
        assert seen[-1][0] == 20 and seen[-1][1] == 20   # 全局进度封顶

    def test_on_round_callback_fires_per_round(self, tmp_path, monkeypatch):
        self._fake(monkeypatch)
        rounds = []
        eb.run_batch(_samples(tmp_path), [eb.Variant("A", "p")], 3, _cfg(tmp_path),
                     on_round=rounds.append)
        assert len(rounds) == 3


class TestBatchEvaluator:
    def test_emits_all_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ev, "classify_image",
            lambda *a, **kw: ('{"category": "科技", "confidence": 0.9}', 1, 1),
        )
        root = _dataset(tmp_path)
        evl = eb.BatchEvaluator("deepseek", "k", "m", str(root), ["科技", "日常"],
                                [eb.Variant("A", "p")], rounds=3, full=True, rpm=600,
                                concurrency=2)
        got = []
        evl.finished_batch.connect(got.append)
        evl.run()
        assert len(got) == 1 and len(got[0]) == 3
        assert got[0][0]["dataset_root"] == str(root)

    def test_total_runs_helper(self, tmp_path):
        evl = eb.BatchEvaluator("s", "k", "m", str(tmp_path), [], 
                                [eb.Variant("A", "p"), eb.Variant("B", "p")], rounds=2)
        assert evl.total_runs() == 4

    def test_empty_dataset_emits_empty(self, tmp_path, monkeypatch):
        evl = eb.BatchEvaluator("deepseek", "k", "m", str(tmp_path / "nope"), ["科技"],
                                [eb.Variant("A", "p")], rounds=1)
        got = []
        evl.finished_batch.connect(got.append)
        evl.run()
        assert got == [[]]
