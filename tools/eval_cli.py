"""命令行批量评估 — 不开界面即可跑多轮 / A/B / 快速模式对比

用法示例（在项目根目录执行）：

    # 3 轮同提示词，测噪声基线 + 一致率 + 阈值曲线
    python tools/eval_cli.py --rounds 3

    # 提示词 A/B（轮内交替，抵消时间漂移）
    python tools/eval_cli.py --rounds 2 --variant 简洁版=simple.txt --variant 详细版=detail.txt

    # 现状 vs 快速模式（thinking 关闭）
    python tools/eval_cli.py --rounds 2 --compare-fast

    # 只跑不存、结果另存 JSON
    python tools/eval_cli.py --rounds 1 --no-save --json result.json

默认从应用自己的配置（QSettings）读取密钥、分类、提示词、关键词；用 --dataset 指定数据集根目录。
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval_batch import Variant, run_batch
from app.eval_stats import report_batch
from app.eval_store import EvalStore
from app.evaluator import RoundConfig, pick_samples, scan_dataset
from app.providers import PROVIDERS, get_provider

FAST_EXTRA_BODY = {"thinking": {"type": "disabled"}}

# 评估用频率：默认高于生产运行频率（配置页那档偏保守，评估没必要那么慢）
EVAL_RPM_DEFAULT = 120

_cancelled = {"flag": False}


def _on_sigint(signum, frame):
    if _cancelled["flag"]:
        print("\n强制退出", flush=True)
        raise SystemExit(130)
    _cancelled["flag"] = True
    print("\n收到中断，正在收尾（再按一次 Ctrl+C 强制退出）...", flush=True)


def load_settings():
    """从应用配置读取运行参数（QSettings；缺省回落到 ConfigTab 的默认值）"""
    from PySide6.QtCore import QSettings

    from app.config_tab import (DEFAULT_CATEGORIES, DEFAULT_KEYWORDS,
                                _as_bool, _parse_categories)
    from app.secure import decrypt_secret

    s = QSettings("DeepSeekImageClassifier", "Config")
    categories = _parse_categories(s.value("categories_raw", DEFAULT_CATEGORIES))
    keywords = {cat: (s.value(cat, "") or DEFAULT_KEYWORDS.get(cat, "")) for cat in categories}
    return {
        "api_key": decrypt_secret(s.value("keys", "") or ""),
        "model": s.value("model", "") or get_provider("deepseek")["default_model"],
        "categories": categories,
        "keywords": {k: v for k, v in keywords.items() if v},
        "prompt": s.value("global_prompt", "") or "",
        "rpm": EVAL_RPM_DEFAULT,          # 评估独立于配置页的运行频率
        "concurrency": int(s.value("concurrency", 3) or 3),
        "use_original": _as_bool(s.value("use_original", False)),
        "last_dataset": s.value("eval_dataset", "") or "",
    }


def build_variants(args, settings) -> list[Variant]:
    """把命令行参数变成待对比的变体列表"""
    from app.vlm import build_prompt_text

    cats, kws = settings["categories"], settings["keywords"]

    def compile_prompt(raw: str) -> str:
        return build_prompt_text(raw, cats, kws)

    if args.variant:
        variants = []
        for spec in args.variant:
            if "=" not in spec:
                raise SystemExit(f"--variant 需要 名称=提示词文件 形式，收到: {spec}")
            name, path = spec.split("=", 1)
            if path == "@config":       # 用配置页当前提示词
                raw = settings["prompt"]
            else:
                p = Path(path)
                if not p.is_file():
                    raise SystemExit(f"提示词文件不存在: {path}")
                raw = p.read_text(encoding="utf-8")
            variants.append(Variant(name.strip(), compile_prompt(raw)))
    else:
        base = Variant("现状", compile_prompt(args.prompt_text or settings["prompt"]))
        variants = [base]

    if args.also_fast or args.compare_fast:
        # 对每个变体再派生一个「关闭思考」的版本，一次跑完交叉对比
        targets = variants if args.also_fast else variants[:1]
        variants = variants + [
            Variant(f"{v.name}+快速", v.prompt_text, FAST_EXTRA_BODY) for v in targets
        ]
    return variants


def resolve_samples(args, settings, store: EvalStore):
    """确定数据集根目录与样本清单

    优先复用固定评估集（跨版本可比）；`--full` 全量评估时**不落盘**，避免把
    随手抽的样本覆盖成新的固定评估集。
    """
    root = Path(args.dataset or settings["last_dataset"] or "")
    if not root.is_dir():
        raise SystemExit(f"数据集根目录无效：{root or '(未指定)'}；请用 --dataset 指定")

    if args.full:
        by_cat = scan_dataset(root, settings["categories"])
        picked = pick_samples(by_cat, args.per_cat, True, None, root)
        if not picked:
            raise SystemExit(f"数据集为空：{root} 下未找到与分类名一致的子目录")
        return root, [str(p) for _, p in picked]

    fixed = store.load_eval_set(root.name)
    if fixed and not args.no_fixed and not args.regenerate:
        alive = [p for p in fixed if Path(p).exists()]
        missing = len(fixed) - len(alive)
        if missing:
            print(f"注意：固定评估集里有 {missing} 个文件已不存在，已跳过", flush=True)
        if alive:
            return root, alive

    # 临时抽样：不落盘，避免把随手抽的样本固化成新的评估集
    if args.no_fixed and not args.regenerate:
        by_cat = scan_dataset(root, settings["categories"])
        picked = pick_samples(by_cat, args.per_cat, False, None, root)
        if not picked:
            raise SystemExit(f"数据集为空：{root} 下未找到与分类名一致的子目录")
        print(f"按每类 {args.per_cat} 张临时抽样 {len(picked)} 张（未写入固定评估集）", flush=True)
        return root, [str(p) for _, p in picked]

    by_cat = scan_dataset(root, settings["categories"])
    picked = pick_samples(by_cat, args.per_cat, False, None, root)
    if not picked:
        raise SystemExit(f"数据集为空：{root} 下未找到与分类名一致的子目录")
    fixed = [str(p) for _, p in picked]
    if store.load_eval_set(root.name):
        print("正在重建固定评估集（旧清单已自动备份为 .bak.json）", flush=True)
    store.save_eval_set(root.name, fixed)
    print(f"已生成固定评估集：{len(fixed)} 张 → {store.data_dir()}", flush=True)
    return root, fixed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek 图片分类工具 · 命令行批量评估")
    ap.add_argument("--rounds", type=int, default=1, help="每个变体跑几轮（默认 1，建议 3）")
    ap.add_argument("--dataset", help="数据集根目录（默认用上次的）")
    ap.add_argument("--per-cat", type=int, default=15, help="每类抽多少张（无固定评估集时生效）")
    ap.add_argument("--full", action="store_true", help="全量评估（所有图片，忽略固定评估集）")
    ap.add_argument("--no-fixed", action="store_true",
                    help="不用固定评估集，按 --per-cat 临时抽样（不落盘）")
    ap.add_argument("--regenerate", action="store_true", help="重建固定评估集（旧清单自动备份）")
    ap.add_argument("--variant", action="append",
                    help="提示词变体，可重复：--variant 名称=提示词文件.txt（文件可写 @config 用配置页的）")
    ap.add_argument("--prompt-text", help="直接指定单变体的提示词（默认用配置页的）")
    ap.add_argument("--compare-fast", action="store_true",
                    help="对比「现状」与「现状+快速模式（关闭思考）」")
    ap.add_argument("--also-fast", action="store_true",
                    help="给每个变体都派生一个关闭思考的版本，一次跑完交叉对比")
    ap.add_argument("--model", help="覆盖模型名")
    ap.add_argument("--rpm", type=int,
                    help=f"覆盖频率（张/分钟，默认 {EVAL_RPM_DEFAULT}，独立于配置页的运行频率）")
    ap.add_argument("--concurrency", type=int, help="覆盖并发数")
    ap.add_argument("--original", action="store_true", help="强制使用原图（不缩图）")
    ap.add_argument("--budget", type=float, default=10.0,
                    help="阈值建议的误挡率预算%%（默认 10）")
    ap.add_argument("--no-save", action="store_true", help="不写入评估历史")
    ap.add_argument("--json", dest="json_out", help="把原始记录另存为 JSON")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _on_sigint)

    settings = load_settings()
    if not settings["api_key"]:
        raise SystemExit("未找到 API 密钥：请先在界面配置页填写并保存")
    if not settings["categories"]:
        raise SystemExit("分类列表为空：请先在界面配置页设置分类")
    if args.model:
        settings["model"] = args.model
    if args.rpm:
        settings["rpm"] = args.rpm
    if args.concurrency:
        settings["concurrency"] = args.concurrency
    if args.original:
        settings["use_original"] = True

    store = EvalStore()
    root, fixed = resolve_samples(args, settings, store)
    samples = [(str(Path(p).relative_to(root).parts[0]) if Path(p).is_relative_to(root)
                else Path(p).parent.name, Path(p)) for p in fixed]
    variants = build_variants(args, settings)

    cfg = RoundConfig(
        service="deepseek", api_key=settings["api_key"], model=settings["model"],
        prompt_text="", categories=settings["categories"], dataset_root=str(root),
        use_original=settings["use_original"], rpm=settings["rpm"],
        concurrency=settings["concurrency"],
    )

    total_requests = len(samples) * len(variants) * args.rounds
    print(f"数据集   {root}")
    print(f"样本     {len(samples)} 张 ｜ 变体 "
          f"{'、'.join(v.name for v in variants)} ｜ 轮次 {args.rounds}")
    print(f"计划请求 {total_requests} 次 ｜ 模型 {settings['model']} ｜ "
          f"并发 {cfg.concurrency} ｜ 频率 {cfg.rpm}/分 ｜ "
          f"{'原图' if cfg.use_original else '缩图'}")
    print("-" * 72, flush=True)

    t0 = time.time()
    last_pct = {"v": -1}

    def on_progress(done, total, name, pred, conf):
        pct = int(done / total * 100) if total else 100
        if pct != last_pct["v"]:
            last_pct["v"] = pct
            print(f"  {done}/{total} ({pct}%)  已用 {time.time() - t0:.0f}s", flush=True)

    def on_round(record):
        print(f"  ▸ 第 {record['round']} 轮 · {record['variant']}: "
              f"{record['accuracy']:.1f}% ({record['correct']}/{record['total']}) "
              f"用时 {record['elapsed_seconds']:.0f}s", flush=True)

    records = run_batch(samples, variants, args.rounds, cfg,
                        should_cancel=lambda: _cancelled["flag"],
                        on_progress=on_progress, on_log=lambda m: print(f"  {m}", flush=True),
                        on_round=on_round)

    if not records:
        print("\n没有产生结果（已取消或数据为空）")
        return 1

    if not args.no_save:
        store.save_batch(records)
        print(f"\n已保存 {len(records)} 条记录 → {store.data_dir()}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"原始记录已导出 → {args.json_out}")

    print("\n" + "=" * 72)
    print(report_batch(records, max_correct_block_rate=args.budget))
    print("=" * 72)
    print(f"总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
