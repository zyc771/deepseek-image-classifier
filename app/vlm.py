"""视觉模型共享调用层 — 提示词构建 / 响应解析 / 单图分类

被 Classifier（批量分类）与 Evaluator（批量评估）共用，避免逻辑重复。
"""
import base64
import time
from pathlib import Path

import requests

from app.image_prep import prepare_image
from app.providers import get_provider

RETRY_TIMES = 3


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
                   prompt_text: str, use_original: bool = False) -> tuple[str, int, int]:
    """单图调用。返回 (raw_response, prompt_tokens, completion_tokens)

    含 3 次重试；全部失败抛异常。
    """
    ext, img_b64 = encode_image(filepath, use_original)
    data_url = f"data:image/{ext};base64,{img_b64}"

    for attempt in range(RETRY_TIMES):
        try:
            resp = requests.post(
                get_provider(service)["endpoint"],
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt_text},
                        ]
                    }]
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                return raw, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except Exception:
            pass
        if attempt < RETRY_TIMES - 1:
            time.sleep(2 * (attempt + 1))
    raise Exception("3次重试均失败")
