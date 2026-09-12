"""eval_stats 纯函数测试 — 多轮聚合 / 跨轮一致性 / 阈值曲线 / 配对比较"""
import pytest

from app import eval_stats as st


def _rec(acc, samples, rid="20260101-000000", prompt_hash="aaaa"):
    """构造一个最小评估记录"""
    total = len(samples)
    correct = sum(1 for s in samples if s["ok"])
    conf = {}
    for s in samples:
        conf.setdefault(s["gt"], {})[s["pred"]] = conf.setdefault(s["gt"], {}).get(s["pred"], 0) + 1
    return {
        "id": rid, "prompt_hash": prompt_hash, "total": total, "correct": correct,
        "accuracy": correct / total * 100 if total else 0.0,
        "confusion": conf, "samples": samples, "errors": [],
        "elapsed_seconds": 100.0, "total_tokens": 1000,
    }


def _s(path, gt, pred, conf):
    return {"path": path, "gt": gt, "pred": pred, "conf": conf, "ok": gt == pred}


class TestAggregate:
    def test_mean_and_range(self):
        r1 = _rec(0, [_s("/a", "科技", "科技", .9), _s("/b", "日常", "科技", .8)])
        r2 = _rec(0, [_s("/a", "科技", "科技", .9), _s("/b", "日常", "日常", .7)])
        agg = st.aggregate([r1, r2])
        assert agg["rounds"] == 2
        assert agg["accuracy_mean"] == pytest.approx(75.0)
        assert agg["accuracy_range"] == pytest.approx(50.0)

    def test_per_class_recall(self):
        r1 = _rec(0, [_s("/a", "科技", "科技", .9), _s("/b", "科技", "日常", .8),
                      _s("/c", "日常", "日常", .9)])
        agg = st.aggregate([r1])
        assert agg["per_class"]["科技"]["mean"] == pytest.approx(50.0)
        assert agg["per_class"]["日常"]["mean"] == pytest.approx(100.0)
        assert agg["per_class"]["科技"]["n"] == 2

    def test_empty_records(self):
        agg = st.aggregate([])
        assert agg["rounds"] == 0 and agg["accuracy_mean"] == 0.0


class TestStability:
    def _three_rounds(self):
        # /a 三轮全对（一致）；/b 两轮错一轮对（分歧）；/c 三轮全错（一致但错）
        return [
            _rec(0, [_s("/a", "科技", "科技", .9), _s("/b", "科技", "日常", .8),
                     _s("/c", "日常", "科技", .8)], rid="r1"),
            _rec(0, [_s("/a", "科技", "科技", .9), _s("/b", "科技", "日常", .8),
                     _s("/c", "日常", "科技", .8)], rid="r2"),
            _rec(0, [_s("/a", "科技", "科技", .9), _s("/b", "科技", "科技", .9),
                     _s("/c", "日常", "科技", .8)], rid="r3"),
        ]

    def test_unanimous_rate_and_agreement(self):
        stab = st.stability(self._three_rounds())
        assert stab["rounds"] == 3 and stab["images"] == 3
        assert stab["unanimous_rate"] == pytest.approx(2 / 3 * 100, abs=0.1)
        assert stab["agreement_mean"] == pytest.approx((3 + 2 + 3) / 3 / 3 * 100, abs=0.1)

    def test_majority_vote_accuracy(self):
        stab = st.stability(self._three_rounds())
        # /a 对；/b 多数票=日常(错)；/c 多数票=科技(错) → 1/3
        assert stab["majority_accuracy"] == pytest.approx(1 / 3 * 100, abs=0.1)

    def test_agreement_split_predicts_error(self):
        """一致组准确率应明显高于分歧组（这是「一致率当错误信号」的核心假设）"""
        stab = st.stability(self._three_rounds())
        unan, split = stab["agreement_split"]["unanimous"], stab["agreement_split"]["split"]
        assert unan["n"] == 2 and split["n"] == 1
        assert unan["accuracy"] == pytest.approx(50.0)   # /a 对，/c 错
        assert split["accuracy"] == pytest.approx(0.0)   # /b 多数票错

    def test_single_round_requires_samples(self):
        legacy = [{"id": "x", "accuracy": 50.0, "total": 2, "confusion": {}}]
        assert st.stability(legacy)["available"] is False


class TestThresholdCurve:
    def _records(self):
        return [_rec(0, [
            _s("/e1", "科技", "日常", 0.95),   # 错但很自信
            _s("/e2", "科技", "日常", 0.55),   # 错且不自信
            _s("/c1", "科技", "科技", 0.90),   # 对
            _s("/c2", "日常", "日常", 0.58),   # 对但不自信
        ])]

    def test_curve_counts(self):
        curve = st.threshold_curve(self._records(), thresholds=[0.6])
        row = curve[0]
        assert row["threshold"] == 0.6
        assert row["blocked_errors"] == 1 and row["blocked_correct"] == 1
        assert row["error_block_rate"] == pytest.approx(50.0)
        assert row["correct_block_rate"] == pytest.approx(50.0)
        assert row["bucket_precision"] == pytest.approx(50.0)

    def test_low_threshold_blocks_nothing(self):
        curve = st.threshold_curve(self._records(), thresholds=[0.1])
        assert curve[0]["blocked_errors"] == 0 and curve[0]["blocked_correct"] == 0

    def test_kept_accuracy_rises_with_threshold(self):
        recs = [_rec(0, [
            _s("/e1", "科技", "日常", 0.95),   # 错，自信
            _s("/e2", "科技", "日常", 0.30),   # 错，不自信 ← 抬高阈值能挡掉它
            _s("/c1", "科技", "科技", 0.90),   # 对
            _s("/c2", "日常", "日常", 0.85),   # 对
        ])]
        curve = st.threshold_curve(recs, thresholds=[0.1, 0.5])
        assert curve[1]["kept_accuracy"] > curve[0]["kept_accuracy"]
        assert curve[1]["kept"] == 3

    def test_default_thresholds_are_sorted_and_bounded(self):
        curve = st.threshold_curve(self._records())
        ts = [r["threshold"] for r in curve]
        assert ts == sorted(ts) and ts[0] > 0 and ts[-1] <= 1.0

    def test_no_samples_returns_empty(self):
        assert st.threshold_curve([{"id": "x", "total": 1, "confusion": {}}]) == []


class TestBestThreshold:
    def test_picks_threshold_within_budget(self):
        recs = [_rec(0, [
            _s("/e1", "科技", "日常", 0.95),   # 自信地错 —— 挡不住
            _s("/e2", "科技", "日常", 0.35),   # 不自信地错 —— 可挡
            _s("/c1", "科技", "科技", 0.90),
            _s("/c2", "日常", "日常", 0.60),
        ])]
        pick = st.best_threshold(recs, max_correct_block_rate=10.0)
        assert pick["threshold"] is not None
        assert pick["blocked_errors"] == 1 and pick["blocked_correct"] == 0
        assert pick["correct_block_rate"] == pytest.approx(0.0)
        assert 0.35 < pick["threshold"] <= 0.60

    def test_returns_none_when_budget_impossible(self):
        """错误样本全都比正确样本自信时，任何阈值都无法只挡错不挡对"""
        recs = [_rec(0, [
            _s("/e1", "科技", "日常", 0.95), _s("/c1", "科技", "科技", 0.20),
        ])]
        pick = st.best_threshold(recs, max_correct_block_rate=1.0)
        assert pick["threshold"] is None
        assert "无法" in pick["reason"] or "没有" in pick["reason"]


class TestPairedCompare:
    def _pair(self):
        a = [_rec(0, [_s("/1", "科技", "科技", .9), _s("/2", "科技", "日常", .9),
                      _s("/3", "日常", "日常", .9), _s("/4", "日常", "科技", .9)], rid="a1")]
        b = [_rec(0, [_s("/1", "科技", "科技", .9), _s("/2", "科技", "科技", .9),
                      _s("/3", "日常", "日常", .9), _s("/4", "日常", "科技", .9)], rid="b1")]
        return a, b

    def test_delta_and_discordant(self):
        a, b = self._pair()
        cmp = st.paired_compare(a, b)
        assert cmp["a"]["accuracy_mean"] == pytest.approx(50.0)
        assert cmp["b"]["accuracy_mean"] == pytest.approx(75.0)
        assert cmp["delta"] == pytest.approx(25.0)
        assert cmp["discordant"]["b_only"] == 1 and cmp["discordant"]["a_only"] == 0

    def test_single_flip_not_significant(self):
        a, b = self._pair()
        assert st.paired_compare(a, b)["significant"] is False

    def test_many_flips_significant(self):
        n = 12
        sa = [_s(f"/{i}", "科技", "科技" if i < 6 else "日常", .9) for i in range(n)]
        sb = [_s(f"/{i}", "科技", "科技", .9) for i in range(n)]
        cmp = st.paired_compare([_rec(0, sa)], [_rec(0, sb)])
        assert cmp["discordant"]["b_only"] == 6
        assert cmp["significant"] is True

    def test_flip_list_reports_direction(self):
        a, b = self._pair()
        flips = st.paired_compare(a, b)["flips"]
        assert flips and flips[0]["path"] == "/2"
        assert flips[0]["a"] == "日常" and flips[0]["b"] == "科技"


class TestGroupByVariant:
    def test_groups_preserving_order(self):
        recs = [_rec(0, [], rid="a"), _rec(0, [], rid="b")]
        recs[0]["variant"], recs[1]["variant"] = "A", "B"
        groups = st.group_by_variant(recs)
        assert list(groups) == ["A", "B"]

    def test_missing_variant_falls_back(self):
        r = _rec(0, [])
        assert list(st.group_by_variant([r])) == ["（未标记）"]

    def test_empty(self):
        assert st.group_by_variant([]) == {}


class TestReportBatch:
    def _recs(self):
        a1 = _rec(0, [_s("/1", "科技", "科技", .9), _s("/2", "科技", "日常", .9)], rid="a1")
        a2 = _rec(0, [_s("/1", "科技", "科技", .9), _s("/2", "科技", "日常", .9)], rid="a2")
        b1 = _rec(0, [_s("/1", "科技", "科技", .9), _s("/2", "科技", "科技", .9)], rid="b1")
        b2 = _rec(0, [_s("/1", "科技", "科技", .9), _s("/2", "科技", "科技", .9)], rid="b2")
        for r in (a1, a2):
            r["variant"] = "A"
        for r in (b1, b2):
            r["variant"] = "B"
        return [a1, b1, a2, b2]

    def test_mentions_every_variant(self):
        text = st.report_batch(self._recs())
        assert "A" in text and "B" in text

    def test_includes_paired_section_for_two_variants(self):
        text = st.report_batch(self._recs())
        assert "配对" in text

    def test_single_variant_has_no_paired_section(self):
        recs = [r for r in self._recs() if r["variant"] == "A"]
        assert "配对" not in st.report_batch(recs)

    def test_four_variants_compare_all_pairs(self):
        """多变体时必须两两都比较 —— 只比第一对会漏掉真正的结论"""
        recs = self._recs()
        for r in recs:
            if r["variant"] == "B":
                r["variant"] = "C"          # 造出 A / C 两个变体再加两个
        extra = []
        for name in ("D", "E"):
            for r in recs[:2]:
                clone = dict(r, variant=name, id=f"{name}{r['id']}")
                extra.append(clone)
        text = st.report_batch(recs + extra)
        for pair in ("A", "C", "D", "E"):
            assert pair in text
        assert text.count("vs") >= 6        # 4 个变体 → 6 对比较

    def test_empty_records(self):
        assert "暂无" in st.report_batch([])


class TestBinomialP:
    def test_balanced_is_one(self):
        assert st.binom_two_sided_p(5, 10) == pytest.approx(1.0)

    def test_zero_of_ten_is_small(self):
        assert st.binom_two_sided_p(0, 10) < 0.01

    def test_empty_returns_one(self):
        assert st.binom_two_sided_p(0, 0) == 1.0

    def test_symmetric(self):
        assert st.binom_two_sided_p(2, 10) == pytest.approx(st.binom_two_sided_p(8, 10))
