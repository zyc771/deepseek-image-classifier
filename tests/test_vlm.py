"""vlm 共享层测试（不发起网络请求）"""
from app.vlm import build_prompt_text, parse_response

CATS = ["科技", "日常", "动漫"]


class TestBuildPromptText:
    def test_replaces_placeholders(self):
        out = build_prompt_text(
            "分类：{categories}\n定义：\n{category_definitions}",
            CATS, {"科技": "芯片;CPU"},
        )
        assert "科技、日常、动漫" in out
        assert "- 科技: 芯片;CPU" in out
        assert "- 日常" in out

    def test_missing_keyword_renders_bare(self):
        out = build_prompt_text("{category_definitions}", CATS, {})
        assert "- 动漫" in out


class TestParseResponse:
    def test_json_standard(self):
        raw = '{"category": "动漫", "confidence": 0.91, "keywords": ["火影"]}'
        assert parse_response(raw, CATS) == ("动漫", 0.91, ["火影"])

    def test_json_with_reason_field_tolerated(self):
        raw = '{"category": "科技", "confidence": 0.8, "reason": "有芯片", "keywords": ["芯片"]}'
        assert parse_response(raw, CATS) == ("科技", 0.8, ["芯片"])

    def test_json_wrapped_in_text(self):
        raw = '结果如下：\n```json\n{"category": "日常", "confidence": 0.7}\n```'
        cat, conf, kws = parse_response(raw, CATS)
        assert (cat, conf) == ("日常", 0.7)

    def test_pipe_format(self):
        assert parse_response("动漫||0.85||火影,鸣人", CATS) == ("动漫", 0.85, ["火影", "鸣人"])

    def test_fuzzy_fallback(self):
        cat, conf, _ = parse_response("这张图是动漫风格", CATS)
        assert cat == "动漫" and conf > 0

    def test_unknown_becomes_unclassified(self):
        assert parse_response("未知||0.5||a", CATS)[0] == "未整理"

    def test_confidence_clamped(self):
        assert parse_response("科技||1.7||a", CATS)[1] == 1.0

    def test_empty_returns_unclassified(self):
        assert parse_response("", CATS)[0] == "未整理"
