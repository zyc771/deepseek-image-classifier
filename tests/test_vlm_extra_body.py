"""classify_image 的 extra_body（快速模式）与 400 自动降级测试"""
import pytest

import app.vlm as vlm


class FakeResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _ok_payload(text='{"category": "科技", "confidence": 0.9}'):
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class FakeSession:
    """按脚本返回响应，并记录每次请求体"""

    def __init__(self, script):
        self.script = list(script)
        self.bodies = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.bodies.append(json)
        return self.script.pop(0) if self.script else FakeResp(200, _ok_payload())


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    vlm.reset_extra_body_state()
    monkeypatch.setattr(vlm, "encode_image", lambda p, o: ("jpeg", "QUJD"))
    yield
    vlm.reset_extra_body_state()


def _patch_session(monkeypatch, session):
    monkeypatch.setattr(vlm, "_session", lambda: session)


class TestExtraBody:
    def test_param_merged_into_request(self, monkeypatch):
        s = FakeSession([FakeResp(200, _ok_payload())])
        _patch_session(monkeypatch, s)
        vlm.classify_image("deepseek", "k", "deepseek-flash", "x.jpg", "p",
                           extra_body={"thinking": {"type": "disabled"}})
        assert s.bodies[0]["thinking"] == {"type": "disabled"}
        assert s.bodies[0]["model"] == "deepseek-flash"

    def test_no_extra_body_keeps_payload_clean(self, monkeypatch):
        s = FakeSession([FakeResp(200, _ok_payload())])
        _patch_session(monkeypatch, s)
        vlm.classify_image("deepseek", "k", "m", "x.jpg", "p")
        assert "thinking" not in s.bodies[0]

    def test_returns_raw_and_token_counts(self, monkeypatch):
        s = FakeSession([FakeResp(200, _ok_payload())])
        _patch_session(monkeypatch, s)
        raw, pt, ct = vlm.classify_image("deepseek", "k", "m", "x.jpg", "p")
        assert raw.startswith("{") and (pt, ct) == (10, 5)


class TestFallback:
    def test_400_falls_back_immediately_without_param(self, monkeypatch):
        """参数不被支持 → 立刻去掉参数重试，并拿到结果"""
        s = FakeSession([FakeResp(400, {"error": "unknown field thinking"}),
                         FakeResp(200, _ok_payload())])
        _patch_session(monkeypatch, s)
        raw, _, _ = vlm.classify_image("deepseek", "k", "m", "x.jpg", "p",
                                       extra_body={"thinking": {"type": "disabled"}})
        assert raw.startswith("{")
        assert len(s.bodies) == 2
        assert "thinking" in s.bodies[0]
        assert "thinking" not in s.bodies[1]      # 降级后的请求不含该参数

    def test_fallback_recorded_as_notice(self, monkeypatch):
        s = FakeSession([FakeResp(400, {}), FakeResp(200, _ok_payload())])
        _patch_session(monkeypatch, s)
        vlm.classify_image("deepseek", "k", "m", "x.jpg", "p",
                           extra_body={"thinking": {"type": "disabled"}})
        notices = vlm.pop_fallback_notices()
        assert len(notices) == 1
        assert "thinking" in notices[0]
        assert vlm.pop_fallback_notices() == []    # 取走后清空

    def test_unsupported_remembered_for_next_image(self, monkeypatch):
        """同一模型只探测一次：后续图片直接不带该参数发请求"""
        s = FakeSession([FakeResp(400, {}), FakeResp(200, _ok_payload()),
                         FakeResp(200, _ok_payload())])
        _patch_session(monkeypatch, s)
        vlm.classify_image("deepseek", "k", "m", "a.jpg", "p",
                           extra_body={"thinking": {"type": "disabled"}})
        vlm.classify_image("deepseek", "k", "m", "b.jpg", "p",
                           extra_body={"thinking": {"type": "disabled"}})
        assert len(s.bodies) == 3
        assert "thinking" not in s.bodies[2]       # 第二次直接不发该参数
        assert len(vlm.pop_fallback_notices()) == 1  # 只提示一次

    def test_unrelated_error_does_not_disable_param(self, monkeypatch):
        """500 之类的错误不应误判为参数不支持"""
        s = FakeSession([FakeResp(500, {}), FakeResp(200, _ok_payload()),
                         FakeResp(200, _ok_payload())])
        _patch_session(monkeypatch, s)
        vlm.classify_image("deepseek", "k", "m", "a.jpg", "p",
                           extra_body={"thinking": {"type": "disabled"}})
        vlm.classify_image("deepseek", "k", "m", "b.jpg", "p",
                           extra_body={"thinking": {"type": "disabled"}})
        assert "thinking" in s.bodies[2]           # 参数仍然带着
        assert vlm.pop_fallback_notices() == []

    def test_all_retries_fail_raises(self, monkeypatch):
        s = FakeSession([FakeResp(500, {})] * 3)
        _patch_session(monkeypatch, s)
        monkeypatch.setattr(vlm.time, "sleep", lambda *_: None)
        with pytest.raises(Exception):
            vlm.classify_image("deepseek", "k", "m", "x.jpg", "p")
