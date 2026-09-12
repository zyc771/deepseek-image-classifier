"""评估统计 — 多轮聚合 / 跨轮一致性 / 多数投票 / 置信度阈值曲线 / 配对比较

全部为纯函数（输入是 run record 列表，无副作用、不触网），便于单测与命令行复用。

为什么要多轮：实测同一提示词同一批图，轮间准确率极差可达 5–7pt，单轮结论不可信。
为什么要一致性：错误样本约 70% 置信度 ≥0.8（"自信地错"），自报置信度几乎不区分对错；
跨轮是否稳定反而是更可靠的信号。
"""
from collections import Counter, defaultdict
from math import comb
from statistics import mean

# 阈值扫描默认档位（0.30 ~ 0.95）
DEFAULT_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]


def has_samples(records: list[dict]) -> bool:
    """记录里是否含逐样本明细（旧的评估记录没有，无法做阈值/一致性分析）"""
    return any(r and r.get("samples") for r in records or [])


def _samples_of(records: list[dict]) -> list[dict]:
    out = []
    for r in records or []:
        out.extend(r.get("samples") or [])
    return out


def _recall(confusion: dict, cat: str) -> tuple[float, int] | None:
    row = (confusion or {}).get(cat) or {}
    n = sum(row.values())
    if n == 0:
        return None
    return row.get(cat, 0) / n * 100, n


def aggregate(records: list[dict]) -> dict:
    """多轮聚合：总准确率的均值/极差 + 逐类召回的均值/极差"""
    records = [r for r in (records or []) if r]
    if not records:
        return {"rounds": 0, "accuracies": [], "accuracy_mean": 0.0,
                "accuracy_min": 0.0, "accuracy_max": 0.0, "accuracy_range": 0.0,
                "per_class": {}, "total_tokens": 0, "elapsed_mean": 0.0}

    accs = [float(r.get("accuracy", 0.0)) for r in records]

    per_class: dict[str, dict] = {}
    cats = {c for r in records for c in (r.get("confusion") or {})}
    for cat in sorted(cats):
        vals, sizes = [], []
        for r in records:
            got = _recall(r.get("confusion") or {}, cat)
            if got is not None:
                vals.append(got[0])
                sizes.append(got[1])
        if vals:
            per_class[cat] = {
                "mean": mean(vals), "min": min(vals), "max": max(vals),
                "range": max(vals) - min(vals), "n": max(sizes),
                "rounds": len(vals),
            }

    return {
        "rounds": len(records),
        "accuracies": accs,
        "accuracy_mean": mean(accs),
        "accuracy_min": min(accs),
        "accuracy_max": max(accs),
        "accuracy_range": max(accs) - min(accs),
        "per_class": per_class,
        "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in records),
        "elapsed_mean": mean([float(r.get("elapsed_seconds", 0.0) or 0.0) for r in records]),
    }


def _majority(entries: list[dict]) -> dict:
    """一张图的多轮结果 → 多数票判定；票数相同时用平均置信度决胜"""
    counts: Counter = Counter()
    conf_sum: defaultdict = defaultdict(float)
    for e in entries:
        counts[e["pred"]] += 1
        conf_sum[e["pred"]] += float(e.get("conf", 0.0) or 0.0)
    best = max(counts, key=lambda p: (counts[p], conf_sum[p] / counts[p], p))
    return {
        "pred": best,
        "votes": counts[best],
        "rounds": len(entries),
        "gt": entries[0]["gt"],
        "ok": best == entries[0]["gt"],
        "conf_mean": conf_sum[best] / counts[best],
    }


def per_image_verdicts(records: list[dict]) -> dict[str, dict]:
    """按图片路径汇总多轮结果 → {path: {pred, votes, rounds, gt, ok, conf_mean, preds}}"""
    by_path: dict[str, list[dict]] = defaultdict(list)
    for r in records or []:
        for s in r.get("samples") or []:
            by_path[s["path"]].append(s)
    out = {}
    for path, entries in by_path.items():
        verdict = _majority(entries)
        verdict["preds"] = [e["pred"] for e in entries]
        verdict["confs"] = [float(e.get("conf", 0.0) or 0.0) for e in entries]
        out[path] = verdict
    return out


def stability(records: list[dict]) -> dict:
    """跨轮稳定性：一致率、多数投票准确率、以及「一致 vs 分歧」的准确率对比

    agreement_split 是「用分歧度当错误信号」的核心证据：
    若一致组准确率显著高于分歧组，就可以把分歧的图丢进「待确认」。
    """
    records = [r for r in (records or []) if r and r.get("samples")]
    if not records:
        return {"available": False, "rounds": 0, "images": 0, "unanimous_rate": 0.0,
                "agreement_mean": 0.0, "majority_accuracy": 0.0,
                "single_round_accuracy": 0.0, "flip_rate": 0.0,
                "agreement_split": {}, "flips": []}

    verdicts = per_image_verdicts(records)
    images = len(verdicts)
    rounds = len(records)

    unanimous, split = [], []
    agreements = []
    for v in verdicts.values():
        rates = v["votes"] / v["rounds"] * 100
        agreements.append(rates)
        (unanimous if v["votes"] == v["rounds"] else split).append(v)

    def _acc(group):
        return {
            "n": len(group),
            "accuracy": (sum(1 for v in group if v["ok"]) / len(group) * 100) if group else 0.0,
            "share": len(group) / images * 100 if images else 0.0,
        }

    single = mean([float(r.get("accuracy", 0.0)) for r in records])
    return {
        "available": True,
        "rounds": rounds,
        "images": images,
        "unanimous_rate": len(unanimous) / images * 100 if images else 0.0,
        "agreement_mean": mean(agreements) if agreements else 0.0,
        "majority_accuracy": sum(1 for v in verdicts.values() if v["ok"]) / images * 100 if images else 0.0,
        "single_round_accuracy": single,
        "majority_gain": (sum(1 for v in verdicts.values() if v["ok"]) / images * 100 - single) if images else 0.0,
        "flip_rate": len(split) / images * 100 if images else 0.0,
        "agreement_split": {"unanimous": _acc(unanimous), "split": _acc(split)},
        "flips": [
            {"path": p, "gt": v["gt"], "preds": v["preds"], "majority": v["pred"], "ok": v["ok"]}
            for p, v in verdicts.items() if v["votes"] != v["rounds"]
        ],
    }


def threshold_curve(records: list[dict], thresholds: list[float] | None = None) -> list[dict]:
    """置信度阈值扫描 → 每个阈值下的「挡错率 / 误挡率 / 分流桶精度」

    blocked_*：置信度低于阈值的样本会被分流进「待确认」。
    error_block_rate 越高越好（拦住错误），correct_block_rate 越低越好（别误伤）。
    bucket_precision：分流的图里真正是错误的比例 —— 决定「待确认」文件夹有多值得看。
    """
    samples = _samples_of(records)
    if not samples:
        return []

    total_errors = sum(1 for s in samples if not s.get("ok"))
    total_correct = len(samples) - total_errors
    rows = []
    for t in (thresholds if thresholds is not None else DEFAULT_THRESHOLDS):
        blocked = [s for s in samples if float(s.get("conf", 0.0) or 0.0) < t]
        be = sum(1 for s in blocked if not s.get("ok"))
        bc = len(blocked) - be
        kept = len(samples) - len(blocked)
        kept_ok = total_correct - bc
        rows.append({
            "threshold": float(t),
            "blocked": len(blocked),
            "blocked_errors": be,
            "blocked_correct": bc,
            "kept": kept,
            "kept_accuracy": (kept_ok / kept * 100) if kept else 0.0,
            "error_block_rate": (be / total_errors * 100) if total_errors else 0.0,
            "correct_block_rate": (bc / total_correct * 100) if total_correct else 0.0,
            "bucket_precision": (be / len(blocked) * 100) if blocked else 0.0,
        })
    return rows


def best_threshold(records: list[dict], max_correct_block_rate: float = 10.0,
                   thresholds: list[float] | None = None) -> dict:
    """在「误挡率 ≤ 预算」的前提下，挑净收益最高的阈值（并列时取最保守的低阈值）"""
    curve = threshold_curve(records, thresholds)
    if not curve:
        return {"threshold": None, "reason": "该记录没有逐样本明细，无法分析阈值"}

    ok = [r for r in curve
          if r["correct_block_rate"] <= max_correct_block_rate and r["blocked_errors"] > 0]
    if not ok:
        return {
            "threshold": None,
            "reason": f"在误挡率 ≤{max_correct_block_rate:.1f}% 的约束下，"
                      f"没有阈值能挡下任何错误样本（错误样本过于自信）",
            "curve": curve,
        }

    best = max(r["error_block_rate"] - r["correct_block_rate"] for r in ok)
    pool = [r for r in ok if abs((r["error_block_rate"] - r["correct_block_rate"]) - best) < 1e-9]
    pick = min(pool, key=lambda r: r["threshold"])
    return {**pick, "reason": "净收益最高且误挡在预算内", "curve": curve}


def binom_two_sided_p(k: int, n: int) -> float:
    """精确二项检验（双侧，p=0.5）—— 用于判断配对差异是否超出随机噪声"""
    if n <= 0:
        return 1.0
    k = min(int(k), n - int(k))
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(a_records: list[dict], b_records: list[dict]) -> dict:
    """两个变体（提示词 A/B、思考开/关、原图/缩图）的配对比较

    同一张图两边都跑过才构成一个配对观测，因此比「只看总准确率」灵敏得多：
    用不一致配对的二项检验判断差异是否显著。
    """
    va = per_image_verdicts(a_records)
    vb = per_image_verdicts(b_records)
    common = sorted(set(va) & set(vb))

    both_ok = a_only = b_only = both_bad = 0
    flips = []
    for path in common:
        ao, bo = va[path]["ok"], vb[path]["ok"]
        if ao and bo:
            both_ok += 1
        elif ao:
            a_only += 1
            flips.append({"path": path, "gt": va[path]["gt"],
                          "a": va[path]["pred"], "b": vb[path]["pred"], "win": "a"})
        elif bo:
            b_only += 1
            flips.append({"path": path, "gt": va[path]["gt"],
                          "a": va[path]["pred"], "b": vb[path]["pred"], "win": "b"})
        else:
            both_bad += 1

    n_disc = a_only + b_only
    p_value = binom_two_sided_p(min(a_only, b_only), n_disc)

    agg_a, agg_b = aggregate(a_records), aggregate(b_records)
    per_class = {}
    for cat in sorted(set(agg_a["per_class"]) | set(agg_b["per_class"])):
        ma = agg_a["per_class"].get(cat, {}).get("mean")
        mb = agg_b["per_class"].get(cat, {}).get("mean")
        if ma is not None and mb is not None:
            per_class[cat] = {"a": ma, "b": mb, "delta": mb - ma,
                              "n": agg_b["per_class"][cat]["n"]}

    return {
        "pairs": len(common),
        "a": {"accuracy_mean": agg_a["accuracy_mean"], "accuracy_range": agg_a["accuracy_range"],
              "rounds": agg_a["rounds"]},
        "b": {"accuracy_mean": agg_b["accuracy_mean"], "accuracy_range": agg_b["accuracy_range"],
              "rounds": agg_b["rounds"]},
        "delta": agg_b["accuracy_mean"] - agg_a["accuracy_mean"],
        "discordant": {"a_only": a_only, "b_only": b_only, "total": n_disc},
        "both_ok": both_ok, "both_bad": both_bad,
        "p_value": p_value,
        "significant": bool(n_disc > 0 and p_value < 0.05),
        "per_class": per_class,
        "flips": flips,
    }


def summarize_text(records: list[dict]) -> str:
    """把多轮结果压成一段可直接贴进界面/日志的结论文字"""
    agg = aggregate(records)
    if agg["rounds"] == 0:
        return "暂无评估记录"
    lines = [
        f"共 {agg['rounds']} 轮：准确率均值 {agg['accuracy_mean']:.1f}%",
        f"各轮 {', '.join(f'{a:.1f}%' for a in agg['accuracies'])}"
        f"（极差 {agg['accuracy_range']:.1f}pt）",
    ]
    stab = stability(records)
    if stab["available"] and stab["rounds"] >= 2:
        lines.append(
            f"跨轮一致 {stab['unanimous_rate']:.1f}% ｜ 多数投票准确率 {stab['majority_accuracy']:.1f}%"
            f"（单轮均值 {stab['single_round_accuracy']:.1f}%，"
            f"净变化 {stab['majority_gain']:+.1f}pt）"
        )
        unan = stab["agreement_split"].get("unanimous", {})
        spl = stab["agreement_split"].get("split", {})
        lines.append(
            f"一致组准确率 {unan.get('accuracy', 0):.1f}%（占 {unan.get('share', 0):.1f}%）"
            f" vs 分歧组 {spl.get('accuracy', 0):.1f}%（占 {spl.get('share', 0):.1f}%）"
        )
    return "\n".join(lines)


def group_by_variant(records: list[dict]) -> dict[str, list[dict]]:
    """按 variant 字段分组，保持首次出现顺序"""
    groups: dict[str, list[dict]] = {}
    for r in records or []:
        if not r:
            continue
        groups.setdefault(r.get("variant") or "（未标记）", []).append(r)
    return groups


def report_batch(records: list[dict], max_correct_block_rate: float = 10.0) -> str:
    """多轮/多变体的完整文字报告（命令行与界面共用）"""
    records = [r for r in (records or []) if r]
    if not records:
        return "暂无评估记录"

    groups = group_by_variant(records)
    lines: list[str] = []
    for name, recs in groups.items():
        agg = aggregate(recs)
        lines.append(f"── 变体「{name}」({agg['rounds']} 轮) ──")
        lines.append("  " + summarize_text(recs).replace("\n", "\n  "))
        lines.append(f"  耗时合计 {sum(float(r.get('elapsed_seconds', 0) or 0) for r in recs):.0f}s "
                     f"｜ token {agg['total_tokens']:,}")

        rows = threshold_curve(recs)
        if rows:
            pick = best_threshold(recs, max_correct_block_rate, rows and None)
            lines.append("  阈值扫描（置信度低于阈值 → 分流「待确认」）:")
            lines.append(f"    {'阈值':>6}{'挡错':>6}{'误挡':>6}{'挡错率':>8}"
                         f"{'误挡率':>8}{'分流精度':>9}{'保留准确率':>11}")
            for row in rows:
                if row["threshold"] not in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
                    continue
                lines.append(
                    f"    {row['threshold']:>6.2f}{row['blocked_errors']:>6}{row['blocked_correct']:>6}"
                    f"{row['error_block_rate']:>7.1f}%{row['correct_block_rate']:>7.1f}%"
                    f"{row['bucket_precision']:>8.1f}%{row['kept_accuracy']:>10.1f}%"
                )
            if pick.get("threshold") is not None:
                lines.append(f"    → 建议阈值 {pick['threshold']:.2f}"
                             f"（挡错 {pick['blocked_errors']}、误挡 {pick['blocked_correct']}）")
            else:
                lines.append(f"    → {pick.get('reason', '无法给出建议阈值')}")

    if len(groups) >= 2:
        names = list(groups)
        lines.append("")
        lines.append("── 配对比较（按同一张图，二项检验）──")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                na, nb = names[i], names[j]
                cmp = paired_compare(groups[na], groups[nb])
                verdict = ("显著" if cmp["significant"]
                           else f"不显著（p={cmp['p_value']:.3f}，差异在噪声内）")
                lines.append(
                    f"  {na} {cmp['a']['accuracy_mean']:.1f}%  vs  {nb} "
                    f"{cmp['b']['accuracy_mean']:.1f}%  →  {cmp['delta']:+.1f}pt"
                )
                lines.append(
                    f"    配对 {cmp['pairs']} 张 ｜ 只{na}对 {cmp['discordant']['a_only']}"
                    f" ｜ 只{nb}对 {cmp['discordant']['b_only']}"
                    f" ｜ 都错 {cmp['both_bad']}  →  {verdict}"
                )
        if len(names) >= 3:
            lines.append("  提示：多个条件指向同一结论时，可把各自的不一致对数相加再看方向"
                         "（各条件相互独立才可合并）")
    return "\n".join(lines)
