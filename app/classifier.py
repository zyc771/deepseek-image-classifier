"""Kimi API 图片分类核心 — 信号驱动的分类器"""
import base64
import shutil
import time
from pathlib import Path
from collections import defaultdict
import requests
from PySide6.QtCore import QThread, Signal, QObject


class ClassifierSignals(QObject):
    """跨线程信号"""
    progress = Signal(int, int, str, str, float, list, int, int)
    # index, total, filename, category, confidence, keywords, prompt_tokens, completion_tokens

    scan_done = Signal(int, int)
    # total_files, pending_files

    finished = Signal(dict)
    # results summary

    log = Signal(str)
    # log message

    preview_done = Signal(str, float, list, str, int, int, float)
    # category, confidence, keywords, raw_response, prompt_tokens, completion_tokens, elapsed_seconds


class Classifier(QThread):
    """分类工作线程"""

    def __init__(self, service: str, api_key: str, model: str, source_dir: str, output_dir: str,
                 categories: list[str], global_prompt: str, category_keywords: dict[str, str] = None,
                 rpm: int = 30, use_original: bool = False, low_conf_threshold: float = 0.6):
        super().__init__()
        self.signals = ClassifierSignals()
        self._service = service
        self._api_key = api_key
        self._model = model
        self._source_dir = Path(source_dir)
        self._output_dir = Path(output_dir)
        self._categories = categories
        self._global_prompt = global_prompt
        self._category_keywords = category_keywords or {}
        self._rpm = rpm
        self._use_original = use_original
        self._low_conf_threshold = low_conf_threshold

        self._paused = False
        self._cancelled = False

        # Per-category result collector
        self._result_keywords: dict[str, list[str]] = defaultdict(list)
        self._category_confidences: dict[str, list[float]] = defaultdict(list)

    def run(self):
        """主入口"""
        # 扫描
        images = [f for f in self._source_dir.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        self.signals.scan_done.emit(len(images), len(images))

        # 过滤已处理
        pending = self._filter_done(images)
        total = len(images)
        pending_count = len(pending)

        self.signals.log.emit(f"扫描完成: {total} 张, 待处理: {pending_count}")

        results = defaultdict(lambda: {"count": 0, "keywords": [], "confidences": []})
        success = 0
        failed = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        start_time = time.time()

        cat_list = "、".join(self._categories)
        defs = []
        for cat in self._categories:
            kws = self._category_keywords.get(cat, "")
            defs.append(f"- {cat}: {kws}" if kws else f"- {cat}")
        cat_defs = "\n".join(defs)
        prompt_text = self._global_prompt.replace("{categories}", cat_list)
        prompt_text = prompt_text.replace("{category_definitions}", cat_defs)

        for idx, img in enumerate(pending):
            if self._cancelled:
                self.signals.log.emit("已取消")
                break

            while self._paused and not self._cancelled:
                time.sleep(0.2)

            if self._cancelled:
                break

            # 限速
            if idx > 0:
                elapsed = time.time() - start_time
                expected = idx / (self._rpm / 60.0)
                if elapsed < expected:
                    time.sleep(expected - elapsed)

            try:
                category, confidence, keywords, raw, pt, ct = self._classify_one(img, prompt_text)
                category = self._resolve_category(category, confidence)
                if category == "待确认":
                    self.signals.log.emit(f"[{idx+1}/{pending_count}] {img.name} → 低置信度({confidence:.2f}) 待确认")
                results[category]["count"] += 1
                results[category]["confidences"].append(confidence)
                if keywords:
                    results[category]["keywords"].extend(keywords)
                    self._result_keywords[category].extend(keywords)
                self._category_confidences[category].append(confidence)
                total_prompt_tokens += pt
                total_completion_tokens += ct
                success += 1

                dest = self._output_dir / category
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dest / img.name)

                self.signals.progress.emit(
                    idx + 1, pending_count, img.name, category,
                    confidence, keywords, pt, ct
                )
            except Exception as e:
                failed += 1
                self.signals.progress.emit(
                    idx + 1, pending_count, img.name, "ERROR", 0, [], 0, 0
                )
                self.signals.log.emit(f"[{idx+1}/{pending_count}] {img.name} → 失败: {str(e)[:60]}")

        elapsed = time.time() - start_time

        # 构建摘要
        summary = {
            "cancelled": self._cancelled,
            "total": total,
            "pending": pending_count,
            "success": success,
            "failed": failed,
            "pending_review": results["待确认"]["count"] if "待确认" in results else 0,
            "elapsed_seconds": elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "categories": {},
        }

        for cat in self._categories + ["待确认", "ERROR"]:
            if cat in results and results[cat]["count"] > 0:
                r = results[cat]
                confs = r["confidences"]
                # 汇总关键词频次
                kw_counts = defaultdict(int)
                for kw in r["keywords"]:
                    kw_counts[kw] += 1
                top_kw = [k for k, _ in sorted(kw_counts.items(), key=lambda x: -x[1])[:10]]

                summary["categories"][cat] = {
                    "count": r["count"],
                    "percentage": r["count"] / max(success, 1) * 100,
                    "avg_confidence": sum(confs) / len(confs) if confs else 0,
                    "top_keywords": top_kw,
                }

        self.signals.finished.emit(summary)

    def _classify_one(self, filepath: Path, prompt_text: str) -> tuple:
        """单张分类，返回 (category, confidence, keywords, raw, prompt_tokens, completion_tokens)"""
        from app.vlm import classify_image, parse_response
        raw, pt, ct = classify_image(
            self._service, self._api_key, self._model, filepath,
            prompt_text, self._use_original,
        )
        category, confidence, keywords = parse_response(raw, self._categories)
        return category, confidence, keywords, raw, pt, ct

    def _resolve_category(self, category: str, confidence: float) -> str:
        """低置信度分流：低于阈值 → 待确认"""
        if confidence < self._low_conf_threshold:
            return "待确认"
        return category

    def _parse_response(self, raw: str) -> tuple[str, float, list[str]]:
        """解析 API 返回（委托共享层，保持既有接口）"""
        from app.vlm import parse_response
        return parse_response(raw, self._categories)

    def preview(self, filepath: str):
        """单张预览测试（非线程）"""
        try:
            cat_list = "、".join(self._categories)
            defs = []
            for cat in self._categories:
                kws = self._category_keywords.get(cat, "")
                defs.append(f"- {cat}: {kws}" if kws else f"- {cat}")
            cat_defs = "\n".join(defs)
            prompt_text = self._global_prompt.replace("{categories}", cat_list)
            prompt_text = prompt_text.replace("{category_definitions}", cat_defs)

            start = time.time()
            category, confidence, keywords, raw, pt, ct = self._classify_one(Path(filepath), prompt_text)
            elapsed = time.time() - start

            self.signals.preview_done.emit(category, confidence, keywords, raw, pt, ct, elapsed)
        except Exception as e:
            self.signals.preview_done.emit("ERROR", 0.0, [], str(e), 0, 0, 0.0)

    def _filter_done(self, images: list[Path]) -> list[Path]:
        """过滤输出目录中已存在的文件: 匹配键 = (文件名, 大小, 修改时间)"""
        existing = set()
        if self._output_dir.exists():
            for d in self._output_dir.iterdir():
                if d.is_dir():
                    for f in d.iterdir():
                        if f.is_file():
                            st = f.stat()
                            existing.add((f.name, st.st_size, int(st.st_mtime)))
        new_images = []
        for img in images:
            st = img.stat()
            key = (img.name, st.st_size, int(st.st_mtime))
            if key not in existing:
                new_images.append(img)
        skipped = len(images) - len(new_images)
        if skipped > 0:
            self.signals.log.emit(f"跳过已处理: {skipped} 张")
        return new_images

    def pause(self):
        self._paused = True
        self.signals.log.emit("已暂停")

    def resume(self):
        self._paused = False
        self.signals.log.emit("已恢复")

    def cancel(self):
        self._cancelled = True
        self._paused = False
        self.signals.log.emit("正在取消...")
