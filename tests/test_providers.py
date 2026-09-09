"""providers 注册表测试"""
import pytest
from app.providers import PROVIDERS, default_service, get_provider, provider_choices


class TestProviders:
    def test_default_service_is_deepseek(self):
        assert default_service() == "deepseek"

    def test_providers_contain_deepseek_and_kimi(self):
        assert set(PROVIDERS.keys()) == {"deepseek", "kimi"}

    def test_deepseek_endpoint_and_model(self):
        p = get_provider("deepseek")
        assert p["endpoint"] == "https://api.deepseek.com/chat/completions"
        assert p["default_model"] == "deepseek-v4-flash-vision-exp"

    def test_kimi_endpoint_and_model(self):
        p = get_provider("kimi")
        assert p["endpoint"] == "https://api.moonshot.cn/v1/chat/completions"
        assert p["default_model"] == "kimi-k2.6"

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError):
            get_provider("unknown")

    def test_choices_preserve_order(self):
        assert provider_choices() == [
            ("deepseek", "DeepSeek"),
            ("kimi", "Kimi (Moonshot)"),
        ]
