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

        def fake_classify(service, api_key, model, path, prompt_text, use_original=False):
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

    def test_empty_results(self):
        rec = ev.build_record([], "p", "d", 0, 0.0, 0, False)
        assert rec["total"] == 0 and rec["accuracy"] == 0.0
