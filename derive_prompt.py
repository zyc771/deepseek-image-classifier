"""从样本图反推自然语言提示词"""
import base64, io, os, random, sys, time
from pathlib import Path
from collections import defaultdict

import requests
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
if not API_KEY:
    sys.exit("请先设置环境变量 MOONSHOT_API_KEY（不要将 key 写入代码提交到仓库）")
API_URL = "https://api.moonshot.cn/v1/chat/completions"
MODEL = "moonshot-v1-8k-vision-preview"

GT_ROOT = Path(r"D:\BaiduNetdiskDownload\素材\kimi分类")
SAMPLE_PER = 4
MAX_SIZE = 384

CATEGORIES = [
    "科技", "日常", "学习", "体育", "军事",
    "动漫", "历史", "地理", "政治", "游戏", "经济",
]

random.seed(42)


def encode_image(filepath: Path) -> tuple[str, str]:
    """缩图 + base64编码, 返回 (mime_ext, b64)"""
    img = Image.open(filepath).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIZE:
        ratio = MAX_SIZE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return "jpeg", base64.b64encode(buf.getvalue()).decode()


def ask_kimi(images_b64: list[tuple[str, str]], cat_name: str) -> str:
    """让 Kimi 看样本图，描述共同特征"""
    content = []
    for ext, b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}})

    text = (
        f"以上{len(images_b64)}张图片都属于「{cat_name}」分类。"
        f"请仔细观察这些图片，总结它们共同的视觉特征。\n"
        f"要求：\n"
        f"1. 用 3-5 句自然语言描述，重点描述画面中能看到什么\n"
        f"2. 如果与其他类别容易混淆，列出区分要点（如'排除手绘画风'）\n"
        f"3. 输出格式：{cat_name}: (你的描述)\n"
        f"4. 不要输出其他内容"
    )
    content.append({"type": "text", "text": text})

    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": content}]},
        timeout=120,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    return f"{cat_name}: (API错误 {resp.status_code})"


def classify_image(filepath: Path, prompt: str) -> tuple[str, float]:
    """返回 (分类, 置信度)"""
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = filepath.suffix.lower().replace(".", "").replace("jpg", "jpeg")
    for _ in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ]}],
                },
                timeout=60,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                parts = raw.split("||")
                cat = parts[0].strip()
                conf = float(parts[1]) if len(parts) > 1 else 0.8
                for c in CATEGORIES:
                    if c in cat:
                        return c, conf
                return "未整理", conf
            time.sleep(2 * (_ + 1))
        except Exception:
            time.sleep(2)
    return "ERROR", 0.0


def main():
    # Step 1: 每类抽样让 Kimi 生成描述
    print("=" * 50)
    print("Step 1: 让 Kimi 看样本图生成各类描述")
    category_descriptions = []

    for cat in CATEGORIES:
        cat_dir = GT_ROOT / cat
        imgs = [f for f in cat_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not imgs:
            continue
        sample = random.sample(imgs, min(SAMPLE_PER, len(imgs)))

        encoded = [encode_image(img) for img in sample]

        print(f"\n  [{cat}] 发送 {len(encoded)} 张样本...")
        desc = ask_kimi(encoded, cat)
        print(f"  -> {desc[:120]}...")
        category_descriptions.append(desc)
        time.sleep(2)

    # Step 2: 拼接完整提示词
    full_prompt = "你是一个图片分类助手。根据图片内容从以下分类中选择最匹配的一个。\n\n分类标准：\n"
    full_prompt += "\n".join(category_descriptions)
    full_prompt += "\n\n回复格式：分类名||置信度(0到1之间的数字)||关键词1,关键词2,关键词3"

    print(f"\n{'=' * 50}")
    print("完整提示词:")
    print(full_prompt)

    # Step 3: 跑全量评估
    print(f"\n{'=' * 50}")
    print("Step 3: 全量评估")

    all_images = []
    for cat in CATEGORIES:
        cat_dir = GT_ROOT / cat
        if cat_dir.exists():
            for f in cat_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    all_images.append((cat, f))

    random.shuffle(all_images)
    print(f"总数: {len(all_images)}")

    results = []
    confusion = defaultdict(lambda: defaultdict(int))
    start = time.time()

    for i, (gt, img) in enumerate(all_images):
        ai, conf = classify_image(img, full_prompt)
        results.append((gt, ai))
        confusion[gt][ai] += 1
        status = "O" if gt == ai else "X"
        cat_tag = f"{gt}->{ai}" if gt != ai else gt
        print(f"  [{i+1:4d}/{len(all_images)}] {status} {cat_tag:<16} {img.name[:25]}")
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            ok = sum(1 for g, a in results if g == a)
            print(f"  ... {i+1}/{len(all_images)}, 准确率暂: {ok}/{i+1}, {(i+1)/elapsed*60:.0f}张/分")

    elapsed = time.time() - start
    correct = sum(1 for g, a in results if g == a)
    acc = correct / len(results) * 100

    print(f"\n准确率: {correct}/{len(results)} = {acc:.1f}%")
    print(f"耗时: {elapsed:.0f}s")

    print(f"\n混淆矩阵 (GT->AI, Top 10):")
    pairs = []
    for g in CATEGORIES:
        for a in CATEGORIES:
            if g != a and confusion[g][a] > 0:
                pairs.append((g, a, confusion[g][a]))
    for g, a, c in sorted(pairs, key=lambda x: -x[2])[:10]:
        print(f"  {g} -> {a}: {c}次")


if __name__ == "__main__":
    main()
