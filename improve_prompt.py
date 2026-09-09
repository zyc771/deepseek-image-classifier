"""Kimi 提示词自改进 — 基准评估 + 错误分析 + 迭代优化"""
import base64, json, os, random, requests, sys, time
from pathlib import Path
from collections import defaultdict, Counter

# Fix encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
if not API_KEY:
    sys.exit("请先设置环境变量 MOONSHOT_API_KEY（不要将 key 写入代码提交到仓库）")
API_URL = "https://api.moonshot.cn/v1/chat/completions"
MODEL = "moonshot-v1-8k-vision-preview"

GT_ROOT = Path(r"D:\BaiduNetdiskDownload\素材\kimi分类")
SAMPLE_PER_CATEGORY = 999  # 全量

CATEGORIES = [
    "科技", "日常", "学习", "体育", "军事", "动漫",
    "历史", "地理", "政治", "游戏", "经济",
]
DEFAULT_KEYWORDS = {
    "科技": "电脑;手机;芯片;AI;数码产品",
    "日常": "美食;宠物;聊天记录;自拍;家庭",
    "学习": "书本;教室;考试;笔记;学生",
    "体育": "运动;球赛;运动员;健身;跑步",
    "军事": "军装;武器;坦克;阅兵;装备",
    "动漫": "二次元;动漫角色;漫画;日系画风",
    "历史": "古代;老照片;文物;历史事件",
    "地理": "地图;地球;山川;自然风光",
    "政治": "领导人;会议;国旗;政府;外交",
    "游戏": "游戏界面;电竞;手柄;网游",
    "经济": "金钱;股票;K线;商业;人民币",
}

CURRENT_PROMPT = """你是一个图片分类助手，所有图片本质上是搞笑/幽默内容。
根据图片内容从以下分类中选择最匹配的一个。

分类标准：
{category_definitions}

回复格式：分类名||置信度(0到1之间的数字)||关键词1,关键词2,关键词3"""


def build_prompt(keywords_map: dict[str, str]) -> str:
    defs = []
    for cat in CATEGORIES:
        kws = keywords_map.get(cat, "")
        defs.append(f"- {cat}: {kws}" if kws else f"- {cat}")
    return CURRENT_PROMPT.replace("{category_definitions}", "\n".join(defs))


def classify_image(filepath: Path, prompt: str) -> tuple[str, float]:
    """返回 (分类, 置信度)"""
    with open(filepath, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    ext = filepath.suffix.lower().replace(".", "").replace("jpg", "jpeg")
    data_url = f"data:image/{ext};base64,{img_b64}"

    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ]
                    }]
                },
                timeout=60,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                # 解析 "分类||置信度||关键词"
                parts = raw.split("||")
                cat = parts[0].strip()
                conf = float(parts[1]) if len(parts) > 1 else 0.8
                # 校验分类名
                if cat not in CATEGORIES:
                    for c in CATEGORIES:
                        if c in cat:
                            cat = c
                            break
                    else:
                        cat = "未整理"
                return cat, conf
            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(2)
        except Exception as e:
            print(f"\n    [DEBUG] Attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(2)
    return "ERROR", 0.0


def evaluate(prompt: str, label: str = "") -> dict:
    """用给定提示词分类标注图，返回评估结果"""
    # 收集样本
    samples = []  # (gt_category, filepath)
    for cat in CATEGORIES:
        cat_dir = GT_ROOT / cat
        if not cat_dir.exists():
            continue
        imgs = [f for f in cat_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        picked = random.sample(imgs, min(SAMPLE_PER_CATEGORY, len(imgs)))
        for img in picked:
            samples.append((cat, img))

    random.shuffle(samples)
    print(f"\n{'='*60}")
    print(f"评估 [{label}]: {len(samples)} 张 (每类最多 {SAMPLE_PER_CATEGORY})")

    # 分类
    results = []  # (gt, ai)
    confusion = defaultdict(Counter)  # {gt: {ai: count}}
    errors = []

    start = time.time()
    for i, (gt, img) in enumerate(samples):
        ai, conf = classify_image(img, prompt)
        results.append((gt, ai))
        confusion[gt][ai] += 1
        if gt != ai:
            errors.append((img.name, gt, ai, conf))
            status = "X"
        else:
            status = "O"
        cat_tag = f"{gt}→{ai}" if gt != ai else f"{gt}"
        print(f"  [{i+1:3d}/{len(samples)}] {status} {cat_tag:<16} {img.name[:30]}")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  ... {i+1}/{len(samples)}, {(i+1)/elapsed*60:.0f}张/分")

    elapsed = time.time() - start

    # 统计
    correct = sum(1 for gt, ai in results if gt == ai)
    accuracy = correct / len(results) * 100

    print(f"\n  准确率: {correct}/{len(results)} = {accuracy:.1f}%")
    print(f"  耗时: {elapsed:.0f}s")

    # 混淆矩阵
    print(f"\n  混淆矩阵 (GT→AI):")
    confused_pairs = []
    for gt_cat in CATEGORIES:
        for ai_cat in CATEGORIES:
            cnt = confusion[gt_cat][ai_cat]
            if gt_cat != ai_cat and cnt > 0:
                confused_pairs.append((gt_cat, ai_cat, cnt))
    confused_pairs.sort(key=lambda x: -x[2])
    for gt, ai, cnt in confused_pairs[:10]:
        print(f"    {gt} → {ai}: {cnt}次")

    return {
        "label": label,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "confusion": {str(k): dict(v) for k, v in confusion.items()},
        "errors": [(n, g, a, c) for n, g, a, c in errors],
        "confused_pairs": confused_pairs,
        "elapsed": elapsed,
        "prompt": prompt,
    }


def ask_kimi_to_improve(current_prompt: str, errors: list, confused_pairs: list, confusion: dict) -> str:
    """让 Kimi 分析错误并生成改进版提示词"""
    error_lines = []
    for name, gt, ai, conf in errors[:30]:
        error_lines.append(f"  {name[:25]} | {gt} | {ai} | 置信度{conf:.2f}")

    confusion_lines = []
    for gt, ai, cnt in confused_pairs[:15]:
        confusion_lines.append(f"  {gt} → {ai}: {cnt}次")

    meta_prompt = f"""你是一个提示词优化专家。当前有一个图片分类任务，
使用以下提示词对搞笑图片进行分类，但准确率不理想。

==== 当前提示词 ====
{current_prompt}
==== 当前提示词结束 ====

==== 分类错误的案例（文件名 | 正确分类 | 错误分类）====
{chr(10).join(error_lines)}
==== 错误案例结束 ====

==== 主要混淆对（A类经常被误判为B类）====
{chr(10).join(confusion_lines)}
==== 混淆对结束 ====

请分析错误原因，然后给出改进后的提示词。

要求：
1. 保留原有的输出格式要求（分类名||置信度(0-1)||关键词）
2. 重点解决上述混淆对——给容易混淆的类别添加区分性视觉特征描述，
   必要时添加负向约束（如"排除手绘画风"、"排除游戏UI界面"等）
3. 每类描述应该具体、可操作，聚焦于图片的视觉特征
4. 只输出改进后的完整提示词（从"你是一个图片分类助手"开始），
   不要任何解释、不要"改进版提示词："这样的标题
"""

    print(f"\n{'='*60}")
    print("正在让 Kimi 分析错误并改进提示词...")
    print(f"  错误案例数: {len(error_lines)}")
    print(f"  混淆对数: {len(confusion_lines)}")

    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": meta_prompt}],
        },
        timeout=120,
    )
    if resp.status_code == 200:
        improved = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"\n==== 改进后的提示词 ====")
        print(improved)
        return improved
    else:
        print(f"  ERROR: {resp.status_code} {resp.text[:100]}")
        return current_prompt


def main():
    random.seed(42)

    # Step 1: 基准 —— 当前提示词
    current_prompt = build_prompt(DEFAULT_KEYWORDS)
    baseline = evaluate(current_prompt, "基准(当前提示词)")

    # Step 2: 让 Kimi 改进
    improved_prompt = ask_kimi_to_improve(
        current_prompt,
        baseline["errors"],
        baseline["confused_pairs"],
        baseline["confusion"],
    )

    # Step 3: 用改进后提示词再跑一轮
    improved_result = evaluate(improved_prompt, "改进后")

    # 对比
    print(f"\n{'='*60}")
    print(f"==== 对比 ====")
    print(f"基准:   {baseline['accuracy']:.1f}% ({baseline['correct']}/{baseline['total']})")
    print(f"改进后: {improved_result['accuracy']:.1f}% ({improved_result['correct']}/{improved_result['total']})")
    delta = improved_result['accuracy'] - baseline['accuracy']
    print(f"变化:   {delta:+.1f}%")

    # 保存改进后的提示词
    out = {
        "baseline_accuracy": baseline["accuracy"],
        "improved_accuracy": improved_result["accuracy"],
        "delta": delta,
        "improved_prompt": improved_prompt,
        "baseline_confused_pairs": [(g, a, c) for g, a, c in baseline["confused_pairs"]],
        "improved_confused_pairs": [(g, a, c) for g, a, c in improved_result["confused_pairs"]],
    }
    out_path = Path(__file__).parent / "improved_prompt_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存到: {out_path}")


if __name__ == "__main__":
    main()
