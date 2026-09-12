"""生产分类器的快速模式透传测试"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app.classifier as cl
import app.vlm as vlm


def _clf(tmp_path, fast_mode):
    return cl.Classifier(
        service="deepseek", api_key="k", model="m",
        source_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
        categories=["科技"], global_prompt="p",
        fast_mode=fast_mode,
    )


class TestFastMode:
    def test_disabled_by_default(self, tmp_path, monkeypatch):
        captured = {}

        def fake(service, api_key, model, path, prompt, use_original=False, extra_body=None):
            captured["extra_body"] = extra_body
            return '{"category": "科技", "confidence": 0.9}', 1, 1

        monkeypatch.setattr(vlm, "classify_image", fake)
        clf = cl.Classifier(service="s", api_key="k", model="m", source_dir=str(tmp_path),
                            output_dir=str(tmp_path), categories=["科技"], global_prompt="p")
        clf._classify_one(tmp_path / "a.jpg", "p")
        assert captured["extra_body"] is None

    def test_enabled_sends_thinking_disabled(self, tmp_path, monkeypatch):
        captured = {}

        def fake(service, api_key, model, path, prompt, use_original=False, extra_body=None):
            captured["extra_body"] = extra_body
            return '{"category": "科技", "confidence": 0.9}', 1, 1

        monkeypatch.setattr(vlm, "classify_image", fake)
        _clf(tmp_path, True)._classify_one(tmp_path / "a.jpg", "p")
        assert captured["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_fallback_notice_surfaces_in_log(self, tmp_path, monkeypatch):
        """参数被服务端拒绝时必须让用户看见，不能静默降级"""
        clf = _clf(tmp_path, True)
        vlm.reset_extra_body_state()
        vlm._mark_unsupported("m", {"thinking": {}}, 400)
        logs = []
        clf.signals.log.connect(logs.append)
        clf._drain_fallback_notices()
        assert any("不支持参数" in m for m in logs)
        assert vlm.pop_fallback_notices() == []      # 只提示一次
