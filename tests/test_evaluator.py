"""评估线程与抽样逻辑测试（monkeypatch 掉网络调用）"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import app.evaluator as ev


def _make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for cat, n in (("科技", 3), ("日常", 2)):
        d = root / cat
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{cat}{i}.jpg").write_bytes(b"x")
    return root


class TestScan:
    def test_scan_groups_by_category(self, tmp_path):
        root = _make_dataset(tmp_path)
        by_cat = ev.scan_dataset(root, ["科技", "日常", "历史"])
        assert len(by_cat["科技"]) == 3
        assert len(by_cat["日常"]) == 2
        assert by_cat["历史"] == []


class TestRecursiveScan:
    def test_subdirectory_images_included(self, tmp_path):
        root = tmp_path / "ds"
        (root / "历史" / "子A").mkdir(parents=True)
        (root / "历史" / "a.jpg").write_bytes(b"1")
        (root / "历史" / "子A" / "b.jpg").write_bytes(b"2")
        by_cat = ev.scan_dataset(root, ["历史"])
        assert len(by_cat["历史"]) == 2

    def test_non_image_files_ignored(self, tmp_path):
        root = tmp_path / "ds"
        (root / "历史").mkdir(parents=True)
        (root / "历史" / "a.jpg").write_bytes(b"1")
        (root / "历史" / "note.vesf").write_bytes(b"x")
        (root / "历史" / "readme.txt").write_bytes(b"y")
        assert len(ev.scan_dataset(root, ["历史"])["历史"]) == 1

    def test_deep_nesting(self, tmp_path):
        root = tmp_path / "ds"
        deep = root / "日常" / "2024" / "01" / "week1"
        deep.mkdir(parents=True)
        (deep / "x.png").write_bytes(b"1")
        assert len(ev.scan_dataset(root, ["日常"])["日常"]) == 1


class TestFixedSetWithSubdir:
    def test_gt_derived_from_top_level_dir(self, tmp_path):
        root = tmp_path / "ds"
        (root / "历史" / "子A").mkdir(parents=True)
        img = root / "历史" / "子A" / "b.jpg"
        img.write_bytes(b"2")
        picked = ev.pick_samples({}, 5, False, [str(img)], root)
        assert picked[0][0] == "历史"
        assert picked[0][1] == img

    def test_flat_file_gt_from_parent(self, tmp_path):
        root = tmp_path / "ds"
        (root / "科技").mkdir(parents=True)
        img = root / "科技" / "a.jpg"
        img.write_bytes(b"1")
        picked = ev.pick_samples({}, 5, False, [str(img)], root)
        assert picked[0][0] == "科技"


class TestSampling:
    def _by_cat(self, tmp_path):
        root = _make_dataset(tmp_path)
        return root, ev.scan_dataset(root, ["科技", "日常"])

    def test_per_category_sampling(self, tmp_path):
        root, by_cat = self._by_cat(tmp_path)
        picked = ev.pick_samples(by_cat, 2, False, None, root)
        assert len(picked) == 4  # 2 类 × 2 张

    def test_full_sampling_ignores_limit(self, tmp_path):
        root, by_cat = self._by_cat(tmp_path)
        picked = ev.pick_samples(by_cat, 1, True, None, root)
        assert len(picked) == 5

    def test_fixed_set_used_verbatim(self, tmp_path):
        root, _ = self._by_cat(tmp_path)
        fixed = [str(root / "科技" / "科技0.jpg")]
        picked = ev.pick_samples({}, 2, False, fixed, root)
        assert [str(p) for _, p in picked] == fixed
        assert picked[0][0] == "科技"  # 类别取自父目录名


ALIAS = "史政 = 历史, 政治, 军事\n排除 = 动漫, 节假日, 黄"


def _merge_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for cat, n in (("历史", 2), ("政治", 3), ("军事", 1), ("动漫", 2), ("科技", 2)):
        d = root / cat
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{cat}{i}.jpg").write_bytes(b"x")
    return root


class TestCategoryMapping:
    """评估期类别归并：不改动数据集目录，只映射标准答案"""

    def _mapping(self):
        from app.category_map import parse_alias
        return parse_alias(ALIAS)

    def test_scan_collects_merged_sources(self, tmp_path):
        root = _merge_dataset(tmp_path)
        by_cat = ev.scan_dataset(root, ["史政", "科技"], self._mapping())
        assert len(by_cat["史政"]) == 6          # 历史2 + 政治3 + 军事1
        assert len(by_cat["科技"]) == 2

    def test_scan_excludes_dirs(self, tmp_path):
        root = _merge_dataset(tmp_path)
        by_cat = ev.scan_dataset(root, ["史政", "科技"], self._mapping())
        assert all("动漫" not in str(p) for p in by_cat["科技"])
        assert "动漫" not in by_cat

    def test_scan_without_mapping_unchanged(self, tmp_path):
        """回归：不配映射时行为与以前完全一致"""
        root = _merge_dataset(tmp_path)
        by_cat = ev.scan_dataset(root, ["历史", "政治", "史政"])
        assert len(by_cat["历史"]) == 2 and by_cat["史政"] == []

    def test_gt_from_path_maps_to_target(self, tmp_path):
        root = _merge_dataset(tmp_path)
        assert ev.gt_from_path(root / "政治" / "政治0.jpg", root, self._mapping()) == "史政"

    def test_gt_from_path_excluded_returns_none(self, tmp_path):
        root = _merge_dataset(tmp_path)
        assert ev.gt_from_path(root / "动漫" / "动漫0.jpg", root, self._mapping()) is None

    def test_gt_from_path_uses_top_level_dir(self, tmp_path):
        root = _merge_dataset(tmp_path)
        img = root / "历史" / "子目录" / "x.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"x")
        assert ev.gt_from_path(img, root, self._mapping()) == "史政"

    def test_pick_samples_drops_excluded(self, tmp_path):
        root = _merge_dataset(tmp_path)
        fixed = [str(root / "历史" / "历史0.jpg"),
                 str(root / "动漫" / "动漫0.jpg"),
                 str(root / "政治" / "政治0.jpg")]
        picked = ev.pick_samples({}, 5, False, fixed, root, self._mapping())
        assert [gt for gt, _ in picked] == ["史政", "史政"]

    def test_pick_samples_sampling_uses_mapping(self, tmp_path):
        root = _merge_dataset(tmp_path)
        by_cat = ev.scan_dataset(root, ["史政"], self._mapping())
        picked = ev.pick_samples(by_cat, 5, False, None, root, self._mapping())
        assert len(picked) == 5                   # min(5, 6)
        assert {gt for gt, _ in picked} == {"史政"}


class TestBuildPrompt:
    def test_prompt_includes_category_keywords(self, tmp_path):
        """回归：分类定义必须包含每类关键词"""
        evaluator = ev.Evaluator(
            "deepseek", "k", "m", str(tmp_path), ["科技", "日常"],
            "分类标准：\n{category_definitions}", category_keywords={"科技": "芯片;CPU"},
        )
        prompt = evaluator.build_prompt()
        assert "- 科技: 芯片;CPU" in prompt
        assert "- 日常" in prompt


class TestConcurrentEvaluation:
    def test_full_evaluation_with_mock_api(self, tmp_path, monkeypatch):
        import time as _time
        root = _make_dataset(tmp_path)  # 科技3 + 日常2
        calls = []

        def fake_classify(service, api_key, model, path, prompt_text,
                          use_original=False, extra_body=None):
            calls.append(Path(path).name)
            _time.sleep(0.05)
            return '{"category": "科技", "confidence": 0.9, "keywords": []}', 10, 5

        monkeypatch.setattr(ev, "classify_image", fake_classify)
        evaluator = ev.Evaluator(
            "deepseek", "k", "m", str(root), ["科技", "日常"], "p {category_definitions}",
            full=True, rpm=600, concurrency=3,
        )
        records = []
        evaluator.finished_record.connect(lambda r: records.append(r))
        evaluator.run()

        assert len(calls) == 5
        assert len(records) == 1
        rec = records[0]
        assert rec["total"] == 5
        assert rec["correct"] == 3          # 科技 3 张正确；日常 2 张被误判为科技
        assert rec["confusion"]["日常"]["科技"] == 2
        assert rec["total_tokens"] == 75    # 5 × (10 + 5)

    def test_cancel_before_run_emits_empty(self, tmp_path, monkeypatch):
        root = _make_dataset(tmp_path)
        monkeypatch.setattr(
            ev, "classify_image",
            lambda *a, **kw: ('{"category": "科技", "confidence": 0.9}', 1, 1),
        )
        evaluator = ev.Evaluator(
            "deepseek", "k", "m", str(root), ["科技", "日常"], "p",
            full=True, rpm=600, concurrency=2,
        )
        records = []
        evaluator.finished_record.connect(lambda r: records.append(r))
        evaluator.cancel()
        evaluator.run()
        assert records == [{}]


class TestClassifySamples:
    """并发核心：被 Evaluator（单轮）与 BatchEvaluator（多轮/多变体）共用"""

    def _samples(self, tmp_path):
        root = _make_dataset(tmp_path)
        return [(cat, f) for cat in ("科技", "日常")
                for f in sorted((root / cat).glob("*.jpg"))]

    def _cfg(self, **over):
        base = dict(service="deepseek", api_key="k", model="m", prompt_text="p",
                    categories=["科技", "日常"], rpm=600, concurrency=3)
        base.update(over)
        return ev.RoundConfig(**base)

    def test_returns_results_and_token_sum(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ev, "classify_image",
            lambda *a, **kw: ('{"category": "科技", "confidence": 0.9}', 10, 5),
        )
        cfg = self._cfg()
        results, tokens = ev.classify_samples(self._samples(tmp_path), cfg)
        assert len(results) == 5
        assert tokens == 75                       # 5 × (10 + 5)
        assert results[0][0] in ("科技", "日常")   # (gt, pred, conf, path)

    def test_progress_callback_fires_per_sample(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ev, "classify_image",
            lambda *a, **kw: ('{"category": "科技", "confidence": 0.9}', 1, 1),
        )
        seen = []
        cfg = self._cfg(concurrency=2)
        ev.classify_samples(self._samples(tmp_path), cfg,
                            on_progress=lambda *a: seen.append(a))
        assert len(seen) == 5

    def test_cancel_stops_submitting(self, tmp_path, monkeypatch):
        calls = []

        def fake(*a, **kw):
            calls.append(1)
            return '{"category": "科技", "confidence": 0.9}', 1, 1

        monkeypatch.setattr(ev, "classify_image", fake)
        cfg = self._cfg(concurrency=1)
        results, _ = ev.classify_samples(self._samples(tmp_path), cfg,
                                        should_cancel=lambda: True)
        assert calls == []
        assert results == []

    def test_api_failure_recorded_as_error(self, tmp_path, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("网络炸了")

        monkeypatch.setattr(ev, "classify_image", boom)
        logs = []
        cfg = self._cfg(concurrency=2)
        results, tokens = ev.classify_samples(self._samples(tmp_path), cfg,
                                             on_log=logs.append)
        assert all(r[1] == "ERROR" for r in results)
        assert tokens == 0
        assert any("网络炸了" in m for m in logs)

    def test_extra_body_forwarded_to_api(self, tmp_path, monkeypatch):
        """快速模式参数必须一路传到 classify_image"""
        captured = {}

        def fake(service, api_key, model, path, prompt, use_original=False, extra_body=None):
            captured["extra_body"] = extra_body
            return '{"category": "科技", "confidence": 0.9}', 1, 1

        monkeypatch.setattr(ev, "classify_image", fake)
        cfg = self._cfg(concurrency=1,
                        extra_body={"thinking": {"type": "disabled"}})
        ev.classify_samples(self._samples(tmp_path), cfg)
        assert captured["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_order_matches_sample_order(self, tmp_path, monkeypatch):
        """结果顺序应与输入样本一致（便于跨轮配对比较）"""
        monkeypatch.setattr(
            ev, "classify_image",
            lambda *a, **kw: ('{"category": "科技", "confidence": 0.9}', 1, 1),
        )
        samples = self._samples(tmp_path)
        cfg = self._cfg(concurrency=3)
        results, _ = ev.classify_samples(samples, cfg)
        assert [r[3] for r in results] == [str(p) for _, p in samples]


class TestBuildRecord:
    def test_statistics(self):
        rec = ev.build_record(
            results=[("科技", "科技", 0.9, "/d/科技/a.jpg"),
                     ("科技", "日常", 0.5, "/d/科技/b.jpg"),
                     ("日常", "日常", 0.8, "/d/日常/c.jpg")],
            prompt_text="p",
            dataset_root="d",
            sample_size=3,
            elapsed=12.5,
            total_tokens=1000,
            use_original=False,
        )
        assert rec["total"] == 3 and rec["correct"] == 2
        assert abs(rec["accuracy"] - 66.666) < 0.01
        assert rec["confusion"]["科技"]["日常"] == 1
        assert len(rec["errors"]) == 1
        assert rec["errors"][0]["gt"] == "科技" and rec["errors"][0]["pred"] == "日常"
        assert rec["errors"][0]["path"] == "/d/科技/b.jpg"
        assert rec["prompt_hash"] and rec["id"]

    def test_error_pred_counted(self):
        rec = ev.build_record(
            results=[("科技", "ERROR", 0.0, "/d/科技/x.jpg")], prompt_text="p",
            dataset_root="d", sample_size=1, elapsed=1.0, total_tokens=0,
            use_original=False,
        )
        assert rec["accuracy"] == 0.0
        assert rec["confusion"]["科技"]["ERROR"] == 1

    def test_records_all_samples_with_confidence(self):
        """阈值曲线需要全部样本的置信度，而不只是错误的那些"""
        rec = ev.build_record(
            results=[("科技", "科技", 0.93, "/d/科技/a.jpg"),
                     ("科技", "日常", 0.88, "/d/科技/b.jpg"),
                     ("日常", "日常", 0.41, "/d/日常/c.jpg")],
            prompt_text="p", dataset_root="d", sample_size=3, elapsed=1.0,
            total_tokens=0, use_original=False,
        )
        assert len(rec["samples"]) == 3
        assert rec["samples"][0] == {"path": "/d/科技/a.jpg", "gt": "科技",
                                     "pred": "科技", "conf": 0.93, "ok": True}
        assert rec["samples"][1]["ok"] is False
        assert rec["samples"][2]["conf"] == 0.41
        assert [s["path"] for s in rec["samples"]] == [r[3] for r in
                                                      [("科技", "科技", 0.93, "/d/科技/a.jpg"),
                                                       ("科技", "日常", 0.88, "/d/科技/b.jpg"),
                                                       ("日常", "日常", 0.41, "/d/日常/c.jpg")]]

    def test_samples_feed_threshold_curve(self):
        """端到端：build_record 的产物必须能直接喂给 eval_stats"""
        from app import eval_stats as st
        rec = ev.build_record(
            results=[("科技", "科技", 0.93, "/a.jpg"), ("科技", "日常", 0.31, "/b.jpg")],
            prompt_text="p", dataset_root="d", sample_size=2, elapsed=1.0,
            total_tokens=0, use_original=False,
        )
        pick = st.best_threshold([rec], max_correct_block_rate=10.0)
        assert pick["threshold"] is not None
        assert pick["blocked_errors"] == 1 and pick["blocked_correct"] == 0

    def test_empty_results(self):
        rec = ev.build_record([], "p", "d", 0, 0.0, 0, False)
        assert rec["total"] == 0 and rec["accuracy"] == 0.0
