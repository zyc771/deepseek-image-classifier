"""批量评估线程 — 用分好类的数据集测试提示词效果（dry-run，不改动数据集）"""
import hashlib
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.vlm import build_prompt_text, classify_image, parse_response

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def scan_dataset(root: Path, categories: list[str]) -> dict[str, list[Path]]:
    """按类别名递归扫描数据集子目录，返回 {类别: [图片路径]}（缺失目录为空列表）

    类别目录下可有多层子文件夹（如 历史/2024/week1/x.jpg），标准答案类别
    始终取一级目录名。
    """
    by_cat: dict[str, list[Path]] = {}
    for cat in categories:
        d = Path(root) / cat
        if d.is_dir():
            by_cat[cat] = sorted(
                f for f in d.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
        else:
            by_cat[cat] = []
    return by_cat


def gt_from_path(path: Path, root: Path) -> str:
    """从图片路径推导标准答案类别：优先取相对数据集根的第一段目录名"""
    try:
        rel = path.relative_to(Path(root))
    except ValueError:
        return path.parent.name
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return path.parent.name


def pick_samples(by_cat: dict[str, list[Path]], per_category: int, full: bool,
                 fixed: list[str] | None, root: Path) -> list[tuple[str, Path]]:
    """返回 [(标准答案类别, 图片路径)]；fixed 非空时按清单执行（支持子目录内文件）"""
    if fixed:
        return [(gt_from_path(Path(item), root), Path(item)) for item in fixed]

    picked: list[tuple[str, Path]] = []
    for cat, files in by_cat.items():
        if not files:
            continue
        chosen = list(files) if full else random.sample(files, min(per_category, len(files)))
        picked.extend((cat, f) for f in chosen)
    random.shuffle(picked)
    return picked


def build_record(results: list[tuple[str, str, float, str]], prompt_text: str,
                 dataset_root: str, sample_size: int, elapsed: float,
                 total_tokens: int, use_original: bool) -> dict:
    """results: [(gt, pred, conf, path)] → 评估记录 dict"""
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors = []
    correct = 0
    conf_sum = 0.0

    for gt, pred, conf, path in results:
        confusion[gt][pred] += 1
        conf_sum += float(conf)
        if gt == pred:
            correct += 1
        else:
            errors.append({
                "path": str(path), "gt": gt, "pred": pred, "conf": round(float(conf), 2),
            })

    total = len(results)
    now = datetime.now()
    return {
        "id": now.strftime("%Y%m%d-%H%M%S"),
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(dataset_root),
        "prompt_snapshot": prompt_text,
        "prompt_hash": hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:8],
        "use_original": bool(use_original),
        "sample_size": sample_size,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total else 0.0,
        "avg_confidence": (conf_sum / total) if total else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "total_tokens": total_tokens,
        "confusion": {gt: dict(preds) for gt, preds in confusion.items()},
        "errors": errors,
    }


class Evaluator(QThread):
    """评估线程：逐张分类（不复制文件），统计后发出记录"""

    progress = Signal(int, int, str, str, float)   # done, total, filename, pred, conf
    finished_record = Signal(dict)
    log = Signal(str)

    def __init__(self, service: str, api_key: str, model: str, dataset_root: str,
                 categories: list[str], prompt_text: str, per_category: int = 15,
                 full: bool = False, use_original: bool = False, rpm: int = 60,
                 fixed: list[str] | None = None, category_keywords: dict[str, str] | None = None):
        super().__init__()
        self._service = service
        self._api_key = api_key
        self._model = model
        self._root = Path(dataset_root)
        self._categories = categories
        self._prompt_text = prompt_text
        self._per_category = per_category
        self._full = full
        self._use_original = use_original
        self._rpm = rpm
        self._fixed = fixed
        self._category_keywords = category_keywords or {}
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.log.emit("正在取消...")

    def run(self):
        try:
            by_cat = scan_dataset(self._root, self._categories)
            samples = pick_samples(by_cat, self._per_category, self._full, self._fixed, self._root)
            total = len(samples)
            if total == 0:
                self.log.emit("数据集为空：请检查根目录下的类别子目录是否与分类名一致")
                self.finished_record.emit({})
                return

            prompt_text = build_prompt_text(
                self._prompt_text, self._categories, self._category_keywords
            )
            results: list[tuple[str, str, float, str]] = []
            total_tokens = 0
            start = time.time()

            for idx, (gt, path) in enumerate(samples):
                if self._cancelled:
                    self.log.emit("已取消（本次结果不保存）")
                    self.finished_record.emit({})
                    return

                if idx > 0:
                    elapsed = time.time() - start
                    expected = idx / (self._rpm / 60.0)
                    if elapsed < expected:
                        time.sleep(expected - elapsed)

                try:
                    raw, pt, ct = classify_image(
                        self._service, self._api_key, self._model, path,
                        prompt_text, self._use_original,
                    )
                    pred, conf, _ = parse_response(raw, self._categories)
                    total_tokens += pt + ct
                except Exception as e:
                    pred, conf = "ERROR", 0.0
                    self.log.emit(f"[{idx+1}/{total}] {path.name} → 失败: {str(e)[:50]}")

                results.append((gt, pred, conf, str(path)))
                self.progress.emit(idx + 1, total, path.name, pred, conf)

            record = build_record(
                results, prompt_text, str(self._root), total,
                time.time() - start, total_tokens, self._use_original,
            )
            self.finished_record.emit(record)
        except Exception as e:
            self.log.emit(f"评估异常: {e}")
            self.finished_record.emit({})
