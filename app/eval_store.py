"""评估历史与固定评估集的持久化

数据目录默认 `%APPDATA%/DeepSeekImageClassifier/eval/`，可用 base_dir 覆盖（测试注入）。
"""
import json
import os
from pathlib import Path


def default_data_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "DeepSeekImageClassifier" / "eval"


def _safe_name(name: str) -> str:
    safe = "".join(ch for ch in str(name) if ch.isalnum() or ch in "-_")
    return safe or "default"


class EvalStore:
    def __init__(self, base_dir: Path | None = None):
        self._dir = Path(base_dir) if base_dir else default_data_dir()

    def data_dir(self) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir

    # ── 固定评估集 ──
    def _set_path(self, name: str) -> Path:
        return self.data_dir() / f"eval_set_{_safe_name(name)}.json"

    def save_eval_set(self, name: str, items: list[str]) -> Path:
        p = self._set_path(name)
        if p.exists():                      # 覆盖旧评估集前自动留一份备份
            try:
                p.replace(p.with_suffix(".bak.json"))
            except OSError:
                pass
        p.write_text(json.dumps(list(items), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def load_eval_set(self, name: str) -> list[str] | None:
        p = self._set_path(name)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, list) else None

    # ── 评估历史 ──
    def save_run(self, record: dict) -> Path:
        run_id = _safe_name(record.get("id") or "unknown")
        p = self.data_dir() / f"run_{run_id}.json"
        p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def load_runs(self) -> list[dict]:
        runs = []
        for p in self.data_dir().glob("run_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                runs.append(data)
        runs.sort(key=lambda r: r.get("id", ""), reverse=True)
        return runs

    def load_run(self, run_id: str) -> dict | None:
        p = self.data_dir() / f"run_{_safe_name(run_id)}.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    # ── 批次（多轮 / 多变体一次运行产生的全部记录）──
    def load_batches(self) -> dict[str, list[dict]]:
        """按 batch_id 分组；没有 batch_id 的旧记录以自身 id 单独成组"""
        groups: dict[str, list[dict]] = {}
        for r in self.load_runs():
            key = r.get("batch_id") or r.get("id") or "unknown"
            groups.setdefault(key, []).append(r)
        for key, items in groups.items():
            items.sort(key=lambda x: (x.get("round", 1), x.get("batch_seq", 0)))
        return dict(sorted(groups.items(), reverse=True))

    def save_batch(self, records: list[dict]) -> list[Path]:
        return [self.save_run(r) for r in records or [] if r]
