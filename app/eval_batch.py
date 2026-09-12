"""多轮 / 多变体评估编排 — 待办 1（重复轮次）、3（快速模式）、4（A/B）共用一套机制

设计要点：**轮内交替**（A、B、A、B…）而不是「先跑完 A 再跑完 B」，以抵消 API 侧
随时间的系统漂移；每条记录都带 `variant` / `round` 标记，事后可按变体分组做配对比较。
"""
import hashlib
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.evaluator import (RoundConfig, build_record, classify_samples,
                           pick_samples, scan_dataset)
from app.vlm import build_prompt_text


@dataclass
class Variant:
    """一个对比变体：不同提示词，或同一提示词 + 不同请求参数（如快速模式）"""

    name: str
    prompt_text: str                 # 已替换占位符的完整提示词
    extra_body: dict | None = None

    def prompt_hash(self) -> str:
        return hashlib.md5(self.prompt_text.encode("utf-8")).hexdigest()[:8]


def make_variants(prompts: dict[str, str], categories: list[str],
                  keywords: dict[str, str] | None = None,
                  extra_bodies: dict[str, dict] | None = None) -> list[Variant]:
    """把 {名称: 原始提示词} 编译成 Variant 列表（占位符在此替换）"""
    out = []
    for name, raw in (prompts or {}).items():
        out.append(Variant(
            name=name,
            prompt_text=build_prompt_text(raw, categories, keywords or {}),
            extra_body=(extra_bodies or {}).get(name),
        ))
    return out


def build_schedule(n_variants: int, rounds: int) -> list[tuple[int, int]]:
    """交替排期 [(轮次, 变体下标)]，例如 2 变体 × 3 轮 → A B A B A B

    轮内交替能让两个变体在时间上尽量靠近，抵消网络/负载/缓存漂移。
    """
    if n_variants <= 0 or rounds <= 0:
        return []
    return [(r, v) for r in range(rounds) for v in range(n_variants)]


def run_batch(samples: list[tuple[str, Path]], variants: list[Variant], rounds: int,
              config: RoundConfig, should_cancel=None, on_progress=None,
              on_log=None, on_round=None) -> list[dict]:
    """按交替排期跑完所有轮次，返回带 variant/round 标记的 record 列表

    config 作为模板；每轮只替换 prompt_text 与 extra_body。
    取消时返回空列表（本轮结果不保存，与单轮行为一致）。
    """
    if not samples or not variants or rounds <= 0:
        return []

    schedule = build_schedule(len(variants), rounds)
    total = len(samples) * len(schedule)
    records: list[dict] = []
    done_offset = 0
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    for seq, (round_idx, variant_idx) in enumerate(schedule):
        if _cancelled():
            return []
        variant = variants[variant_idx]
        cfg = replace(config, prompt_text=variant.prompt_text,
                      extra_body=variant.extra_body)

        def _progress(done, _total, name, pred, conf, _off=done_offset):
            if on_progress:
                on_progress(_off + done, total, name, pred, conf)

        start = time.time()
        results, tokens = classify_samples(
            samples, cfg, should_cancel=should_cancel,
            on_progress=_progress, on_log=on_log,
        )
        if _cancelled():
            return []

        record = build_record(
            results, variant.prompt_text, config.dataset_root,
            len(samples), time.time() - start, tokens, config.use_original,
        )
        record.update({
            "id": f"{stamp}-{variant.name[:12]}-r{round_idx + 1}",
            "batch_id": stamp,
            "variant": variant.name,
            "round": round_idx + 1,
            "batch_seq": seq,
            "concurrency": config.concurrency,
            "extra_body": variant.extra_body or {},
            "rounds_total": rounds,
        })
        records.append(record)
        done_offset += len(samples)
        if on_round:
            on_round(record)

    return records


class BatchEvaluator(QThread):
    """多轮/多变体评估线程（界面与命令行共用 run_batch）"""

    progress = Signal(int, int, str, str, float)   # done, total, filename, pred, conf
    round_finished = Signal(dict)
    finished_batch = Signal(list)
    log = Signal(str)

    def __init__(self, service: str, api_key: str, model: str, dataset_root: str,
                 categories: list[str], variants: list[Variant], rounds: int = 1,
                 per_category: int = 15, full: bool = False, use_original: bool = False,
                 rpm: int = 60, fixed: list[str] | None = None, concurrency: int = 3):
        super().__init__()
        self._service = service
        self._api_key = api_key
        self._model = model
        self._root = Path(dataset_root)
        self._categories = list(categories or [])
        self._variants = list(variants or [])
        self._rounds = max(1, int(rounds or 1))
        self._per_category = per_category
        self._full = full
        self._use_original = use_original
        self._rpm = rpm
        self._fixed = fixed
        self._concurrency = max(1, int(concurrency or 1))
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.log.emit("正在取消...")

    def total_runs(self) -> int:
        return len(self._variants) * self._rounds

    def _template(self) -> RoundConfig:
        return RoundConfig(
            service=self._service, api_key=self._api_key, model=self._model,
            prompt_text="", categories=list(self._categories),
            dataset_root=str(self._root), use_original=self._use_original,
            rpm=self._rpm, concurrency=self._concurrency,
        )

    def run(self):
        try:
            by_cat = scan_dataset(self._root, self._categories)
            samples = pick_samples(by_cat, self._per_category, self._full,
                                   self._fixed, self._root)
            if not samples:
                self.log.emit("数据集为空：请检查根目录下的类别子目录是否与分类名一致")
                self.finished_batch.emit([])
                return
            records = run_batch(
                samples, self._variants, self._rounds, self._template(),
                should_cancel=lambda: self._cancelled,
                on_progress=self.progress.emit,
                on_log=self.log.emit,
                on_round=self.round_finished.emit,
            )
            if self._cancelled:
                self.log.emit("已取消（本次结果不保存）")
                self.finished_batch.emit([])
                return
            self.finished_batch.emit(records)
        except Exception as e:
            self.log.emit(f"评估异常: {e}")
            self.finished_batch.emit([])
