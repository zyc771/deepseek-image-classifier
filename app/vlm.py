"""视觉模型共享调用层 — 提示词构建 / 响应解析 / 单图分类

被 Classifier（批量分类）与 Evaluator（批量评估）共用，避免逻辑重复。
含并发支持：时隙限速器 + 线程本地 HTTP 连接复用。
"""
import base64
import threading
import time
from pathlib import Path

import requests

from app.image_prep import prepare_image
from app.providers import get_provider

RETRY_TIMES = 3

# 快速模式：关闭模型思考。实测输出 token 600→35、单张延迟 3.99s→0.90s（约 4.4 倍），
# 在 60 张难图上未检出准确率差异（配对 14:17，p=0.72）——但样本量不足以证明等价，
# 因此默认关闭，由用户自行权衡后开启。服务端若不支持该参数会自动降级。
FAST_MODE_BODY = {"thinking": {"type": "disabled"}}

# 服务端明确拒绝（参数不存在/不支持）的状态码
_PARAM_REJECT_CODES = {400, 404, 422}

_thread_local = threading.local()

# 已判定「该模型不支持该参数」的集合，形如 {(模型名, 参数名)}。
# 探测一次即可，后续图片不再重复发送注定失败的参数。
_unsupported_params: set[tuple[str, str]] = set()
_fallback_notices: list[str] = []
_notice_lock = threading.Lock()


def reset_extra_body_state():
    """清空参数兼容性记忆与降级提示（测试与切换模型时使用）"""
    with _notice_lock:
        _unsupported_params.clear()
        _fallback_notices.clear()


def pop_fallback_notices() -> list[str]:
    """取出并清空降级提示（供界面写日志用）"""
    with _notice_lock:
        out = list(_fallback_notices)
        _fallback_notices.clear()
    return out


def supported_extra_body(model: str, extra_body: dict | None) -> dict:
    """过滤掉该模型已被判定不支持的参数"""
    if not extra_body:
        return {}
    return {k: v for k, v in extra_body.items() if (model, k) not in _unsupported_params}


def _mark_unsupported(model: str, body: dict, code: int):
    """记录参数不被支持，并生成一次降级提示"""
    with _notice_lock:
        fresh = [k for k in body if (model, k) not in _unsupported_params]
        if not fresh:
            return
        for k in fresh:
            _unsupported_params.add((model, k))
        _fallback_notices.append(
            f"模型 {model} 不支持参数 {', '.join(fresh)}（HTTP {code}），已自动降级为普通模式"
        )


def _session() -> requests.Session:
    """线程本地 requests.Session（复用 TCP/TLS 连接，省去每张图握手开销）"""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


class RateLimiter:
    """时隙限速器：全局平均速率不超过 rpm，线程安全

    每次 acquire() 预留一个时间槽，多线程并发时按槽位先后放行，
    从而在并发下依然严格守住 RPM。
    """

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(int(rpm or 0), 1)
        if self.interval > 60.0:      # rpm <= 1 时兜底为至少 1 张/分钟
            self.interval = 60.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            if self._next_slot < now:
                self._next_slot = now
            wait = self._next_slot - now
            self._next_slot += self.interval
        if wait > 0:
            time.sleep(wait)


def build_prompt_text(global_prompt: str, categories: list[str],
                      category_keywords: dict[str, str]) -> str:
    """替换 {categories} 与 {category_definitions} 占位符"""
    cat_list = "、".join(categories)
    defs = []
    for cat in categories:
        kws = (category_keywords or {}).get(cat, "")
        defs.append(f"- {cat}: {kws}" if kws else f"- {cat}")
    text = global_prompt.replace("{categories}", cat_list)
    return text.replace("{category_definitions}", "\n".join(defs))


def parse_response(raw: str, categories: list[str]) -> tuple[str, float, list[str]]:
    """四级解析：JSON → || 分隔 → 模糊匹配 → 未整理（向后兼容）"""
    raw_clean = (raw or "").strip()

    parsed = _try_parse_json(raw_clean)
    if parsed is not None:
        cat, conf, kws = parsed
    elif "||" in raw_clean:
        parts = [p.strip() for p in raw_clean.split("||")]
        cat = parts[0] if len(parts) > 0 else "未整理"
        conf = 0.8
        if len(parts) > 1:
            try:
                conf = float(parts[1])
            except ValueError:
                conf = 0.8
        kws = []
        if len(parts) > 2:
            kws = [k.strip() for k in parts[2].replace(",", "，").replace("，", ",").split(",") if k.strip()]
    else:
        cat = "未整理"
        conf = 0.0
        for c in categories:
            if c in raw_clean:
                cat = c
                conf = 0.8 if raw_clean != c else 1.0
                break
        kws = []

    # 校验分类名
    if cat not in categories:
        for c in categories:
            if c in cat:
                cat = c
                break
        else:
            cat = "未整理"

    # 规范化置信度
    conf = max(0.0, min(1.0, conf))
    return cat, conf, kws


def _try_parse_json(raw_clean: str) -> tuple[str, float, list[str]] | None:
    """提取 JSON 对象（容忍包裹文字/代码围栏与额外字段）；失败返回 None"""
    import json as _json

    start = raw_clean.find("{")
    end = raw_clean.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = _json.loads(raw_clean[start:end + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    cat = str(obj.get("category", "")).strip()
    conf = obj.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0

    kws_obj = obj.get("keywords", []) or []
    if isinstance(kws_obj, str):
        kws = [k.strip() for k in kws_obj.replace(",", "，").replace("，", ",").split(",") if k.strip()]
    elif isinstance(kws_obj, list):
        kws = [str(k).strip() for k in kws_obj if str(k).strip()]
    else:
        kws = []

    return cat or "未整理", conf, kws


def encode_image(filepath: Path, use_original: bool) -> tuple[str, str]:
    """返回 (mime_ext, base64)；use_original=True 时原图直传"""
    if use_original:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = filepath.suffix.lower().replace(".", "").replace("jpg", "jpeg")
        return ext, b64
    return prepare_image(filepath)


def classify_image(service: str, api_key: str, model: str, filepath: Path,
                   prompt_text: str, use_original: bool = False,
                   extra_body: dict | None = None) -> tuple[str, int, int]:
    """单图调用。返回 (raw_response, prompt_tokens, completion_tokens)

    含 3 次重试；全部失败抛异常。
    extra_body 为附加请求参数（如快速模式的 thinking 开关）；若服务端拒绝该参数，
    会自动去掉它重试一次并记录降级提示，不影响本次结果。
    """
    ext, img_b64 = encode_image(filepath, use_original)
    data_url = f"data:image/{ext};base64,{img_b64}"

    for attempt in range(RETRY_TIMES):
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt_text},
                ]
            }]
        }
        active = supported_extra_body(model, extra_body)
        payload.update(active)

        try:
            resp = _session().post(
                get_provider(service)["endpoint"],
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                return raw, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            if active and resp.status_code in _PARAM_REJECT_CODES:
                # 参数不被支持：记住并立刻用「去掉参数」的请求重试（不空等）
                _mark_unsupported(model, active, resp.status_code)
                continue
        except Exception:
            pass
        if attempt < RETRY_TIMES - 1:
            time.sleep(2 * (attempt + 1))
    raise Exception("3次重试均失败")
