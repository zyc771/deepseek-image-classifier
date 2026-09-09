"""图像预处理 — EXIF 方向校正 + 超限缩图 + JPEG 压缩

理论依据（2026-09-09 分析定稿）：
- EXIF 校正纯增益（修复竖拍图方向）
- 缩图到 1024px 语义信息完整保留（模型内部通常按 1024-2048 处理，
  超限原图本身会被内部降采样）
- q85 感知无损
"""
import base64
import io
from pathlib import Path
from PIL import Image, ImageOps

MAX_SIDE = 1024
JPEG_QUALITY = 85


def prepare_image(filepath: Path) -> tuple[str, str]:
    """处理单张图片，返回 (mime_ext, base64)。

    流程：exif_transpose 校正方向 → 仅当最大边 > MAX_SIDE 时 LANCZOS 缩图
    → RGBA/LA/P 透明合成白底 → JPEG(quality=JPEG_QUALITY) → base64
    """
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)

    w, h = img.size
    if max(w, h) > MAX_SIDE:
        ratio = MAX_SIDE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "jpeg", base64.b64encode(buf.getvalue()).decode()
