"""并发分类测试（monkeypatch 掉真实网络调用）"""
import time

import pytest

from app.classifier import Classifier


def _mk_images(tmp_path, n=6):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        (src / f"img{i}.jpg").write_bytes(b"data" * 10 + bytes([i]))
    return src


def _make_clf(tmp_path, src, out, **kw):
    return Classifier(
        "deepseek", "k", "m", str(src), str(out), ["科技"], "prompt {categories}",
        **kw,
    )


class TestConcurrency:
    def test_all_images_processed_and_copied(self, tmp_path, monkeypatch):
        src = _mk_images(tmp_path, 6)
        out = tmp_path / "out"
        clf = _make_clf(tmp_path, src, out, rpm=600, concurrency=3)

        calls = []

        def fake_classify(self, filepath, prompt_text):
            calls.append(filepath.name)
            return "科技", 0.9, [], "raw", 10, 5

        monkeypatch.setattr(Classifier, "_classify_one", fake_classify)
        clf.run()

        assert len(calls) == 6
        assert sorted(calls) == sorted(f"img{i}.jpg" for i in range(6))
        assert len(list((out / "科技").iterdir())) == 6

    def test_parallel_is_faster_than_serial(self, tmp_path, monkeypatch):
        src = _mk_images(tmp_path, 6)
        out = tmp_path / "out"

        def slow_classify(self, filepath, prompt_text):
            time.sleep(0.2)
            return "科技", 0.9, [], "raw", 1, 1

        monkeypatch.setattr(Classifier, "_classify_one", slow_classify)

        clf_serial = _make_clf(tmp_path, src, tmp_path / "out1", rpm=600, concurrency=1)
        t0 = time.monotonic()
        clf_serial.run()
        serial = time.monotonic() - t0

        clf_par = _make_clf(tmp_path, src, tmp_path / "out2", rpm=600, concurrency=3)
        t0 = time.monotonic()
        clf_par.run()
        parallel = time.monotonic() - t0

        assert parallel < serial * 0.75, f"serial={serial:.2f}s parallel={parallel:.2f}s"

    def test_default_concurrency_is_three(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        assert clf._concurrency == 3

    def test_failure_counted_and_run_continues(self, tmp_path, monkeypatch):
        src = _mk_images(tmp_path, 3)
        out = tmp_path / "out"
        clf = _make_clf(tmp_path, src, out, rpm=600, concurrency=2)

        def flaky_classify(self, filepath, prompt_text):
            if filepath.name == "img1.jpg":
                raise Exception("boom")
            return "科技", 0.8, [], "raw", 1, 1

        monkeypatch.setattr(Classifier, "_classify_one", flaky_classify)
        clf.run()
        assert len(list((out / "科技").iterdir())) == 2  # 失败那张不落盘

    def test_cancel_before_run_processes_nothing(self, tmp_path, monkeypatch):
        src = _mk_images(tmp_path, 3)
        out = tmp_path / "out"
        clf = _make_clf(tmp_path, src, out, rpm=600, concurrency=2)
        monkeypatch.setattr(
            Classifier, "_classify_one",
            lambda self, f, p: ("科技", 0.9, [], "raw", 1, 1),
        )
        clf.cancel()
        clf.run()
        assert not (out / "科技").exists() or list((out / "科技").iterdir()) == []
