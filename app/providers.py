"""API 服务商注册表 — 查询端点与默认模型"""
PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "label": "DeepSeek",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "default_model": "deepseek-flash",
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
        "default_model": "kimi-k2.6",
    },
}

DEFAULT_SERVICE = "deepseek"


def default_service() -> str:
    return DEFAULT_SERVICE


def get_provider(service: str) -> dict:
    if service not in PROVIDERS:
        raise ValueError(f"未知服务商: {service}")
    return PROVIDERS[service]


def provider_choices() -> list[tuple[str, str]]:
    return [(sid, p["label"]) for sid, p in PROVIDERS.items()]
