"""生成应用图标：把图层底板的白色替换为指定底色，输出 png 与多尺寸 ico

用法：
    python tools/make_icon.py                    # 默认 #EEF1F6（与浅色主题同色）
    python tools/make_icon.py --bg "#E3EAF4"     # 指定底色
    python tools/make_icon.py --src other.png --out resources
"""
import argparse
from collections import deque
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "resources" / "app.png"
DEFAULT_OUT = ROOT / "resources"
DEFAULT_BG = "#EEF1F6"
ICO_SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)]
CARD_SEED = (175, 120)      # 卡片内部的一个点（用于洪水填充识别卡片区域）


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def find_card_pixels(px, w: int, h: int, seed=CARD_SEED) -> set[tuple[int, int]]:
    """洪水填充出白色卡片区域（避免把卡片也染成底色）"""
    card: set[tuple[int, int]] = set()
    queue = deque([seed])
    seen: set[tuple[int, int]] = set()
    while queue:
        x, y = queue.popleft()
        if (x, y) in seen or not (0 <= x < w and 0 <= y < h):
            continue
        seen.add((x, y))
        r, g, b, a = px[x, y]
        if a > 200 and r > 240 and g > 240 and b > 240:
            card.add((x, y))
            queue.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    return card


def make_icon(src: Path, out_dir: Path, bg: tuple[int, int, int]) -> tuple[Path, Path]:
    img = Image.open(src).convert("RGBA")
    w, h = img.size
    px = img.load()
    card = find_card_pixels(px, w, h)

    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a > 200 and r > 244 and g > 244 and b > 244 and (x, y) not in card:
                px[x, y] = (*bg, a)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "app.png"
    ico_path = out_dir / "app.ico"
    img.save(png_path)
    img.save(ico_path, format="ICO", sizes=ICO_SIZES)
    return png_path, ico_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--bg", default=DEFAULT_BG)
    args = ap.parse_args()

    png_path, ico_path = make_icon(Path(args.src), Path(args.out), hex_to_rgb(args.bg))
    ico = Image.open(ico_path)
    print(f"底色 {args.bg} 已应用")
    print(f"PNG: {png_path}")
    print(f"ICO: {ico_path}  尺寸: {sorted(ico.info.get('sizes', []))}")


if __name__ == "__main__":
    main()
