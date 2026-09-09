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
                 rpm: int = 30):
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
            "elapsed_seconds": elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "categories": {},
        }

        for cat in self._categories + ["ERROR"]:
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
        with open(filepath, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = filepath.suffix.lower().replace(".", "").replace("jpg", "jpeg")
        data_url = f"data:image/{ext};base64,{img_b64}"

        for attempt in range(3):
            try:
                from app.providers import get_provider
                resp = requests.post(
                    get_provider(self._service)["endpoint"],
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
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
                    pt = usage.get("prompt_tokens", 0)
                    ct = usage.get("completion_tokens", 0)

                    category, confidence, keywords = self._parse_response(raw)
                    return category, confidence, keywords, raw, pt, ct
                elif resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(2)
        raise Exception(f"3次重试均失败")

    def _parse_response(self, raw: str) -> tuple[str, float, list[str]]:
        """解析 API 返回 -> (分类, 置信度, 关键词列表)"""
        raw_clean = raw.strip()

        # 尝试按 || 切分
        if "||" in raw_clean:
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
            # 回退：模糊匹配
            cat = "未整理"
            conf = 0.0
            for c in self._categories:
                if c in raw_clean:
                    cat = c
                    conf = 0.8 if raw_clean != c else 1.0
                    break
            kws = []

        # 校验分类名
        if cat not in self._categories:
            for c in self._categories:
                if c in cat:
                    cat = c
                    break
            else:
                cat = "未整理"

        # 规范化置信度
        conf = max(0.0, min(1.0, conf))

        return cat, conf, kws

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
