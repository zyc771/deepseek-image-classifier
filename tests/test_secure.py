"""secure 加解密测试（Windows 主路径 + 回退 + 兼容）"""
import base64
from app import secure


class TestSecure:
    def test_roundtrip_dpapi(self):
        blob = secure.encrypt_secret("sk-test123")
        assert blob.startswith("dpapi:")
        assert "sk-test123" not in blob
        assert secure.decrypt_secret(blob) == "sk-test123"

    def test_b64_prefix_roundtrip(self):
        blob = "b64:" + base64.b64encode("sk-plain".encode()).decode()
        assert secure.decrypt_secret(blob) == "sk-plain"

    def test_legacy_bare_base64(self):
        blob = base64.b64encode("sk-legacy".encode()).decode()
        assert secure.decrypt_secret(blob) == "sk-legacy"

    def test_garbage_returns_empty(self):
        assert secure.decrypt_secret("not-a-valid-blob!!!") == ""
