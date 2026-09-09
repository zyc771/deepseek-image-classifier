"""API Key 安全存储 — Windows DPAPI 加密，非 Windows 回退 base64"""
import base64
import ctypes
import sys
from ctypes import wintypes


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _crypt_protect(data: bytes) -> bytes:
    """DPAPI 加密，仅当前 Windows 用户可解密"""
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = _DATA_BLOB(
        len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
    )
    blob_out = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _crypt_unprotect(data: bytes) -> bytes:
    """DPAPI 解密"""
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = _DATA_BLOB(
        len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
    )
    blob_out = _DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def encrypt_secret(plain: str) -> str:
    if not plain:
        return ""
    if sys.platform == "win32":
        try:
            blob = _crypt_protect(plain.encode("utf-8"))
            return "dpapi:" + base64.b64encode(blob).decode()
        except Exception:
            pass
    # 回退：base64 明文（与旧行为一致）
    return "b64:" + base64.b64encode(plain.encode("utf-8")).decode()


def decrypt_secret(blob: str) -> str:
    if not blob:
        return ""
    if blob.startswith("dpapi:"):
        try:
            return _crypt_unprotect(base64.b64decode(blob[6:])).decode("utf-8")
        except Exception:
            return ""
    if blob.startswith("b64:"):
        try:
            return base64.b64decode(blob[4:]).decode("utf-8")
        except Exception:
            return ""
    # 兼容旧格式：无前缀 base64
    try:
        return base64.b64decode(blob).decode("utf-8")
    except Exception:
        return ""
