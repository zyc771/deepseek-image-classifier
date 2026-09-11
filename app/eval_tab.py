"""评估页 — 数据集批量测试、反馈与历史对比"""
import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QPlainTextEdit, QSpinBox, QCheckBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QHeaderView, QSplitter, QAbstractItemView,
)


from app.eval_store import EvalStore
from app.evaluator import Evaluator, pick_samples, scan_dataset
from app.providers import get_provider

THUMB = 96
EVAL_RPM_DEFAULT = 120


class EvalTab(QWidget):
    def __init__(self, parent=None, store: EvalStore | None = None):
        super().__init__(parent)
        self._store = store or EvalStore()
        self._evaluator = None
        self._current_record = None
        self._categories: list[str] = []
        self._api_key = ""
        self._model = get_provider("deepseek")["default_model"]
        self._rpm = 60
        self._use_original = False
        self._category_keywords: dict[str, str] = {}
        self._concurrency = 3
        self._config_prompt = ""
        self._build_ui()
        self._reload_history()

    # ── UI 构建 ──
    def _build_ui(self):
        outer = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        top = QWidget()
        tl = QVBoxLayout(top)

        g_ds = QGroupBox("数据集（根目录下按类别名分子目录）")
        dl = QVBoxLayout(g_ds)
        row = QHBoxLayout()
        self._dataset_input = QLineEdit()
        self._dataset_input.setPlaceholderText("选择已分好类的数据集根目录（如 D:\\...\\kimi分类）")
        row.addWidget(self._dataset_input)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_dataset)
        row.addWidget(btn_browse)
        dl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("每类张数:"))
        self._per_cat_spin = QSpinBox()
        self._per_cat_spin.setRange(5, 100)
        self._per_cat_spin.setValue(15)
        row2.addWidget(self._per_cat_spin)
        self._full_check = QCheckBox("全量评估")
        row2.addWidget(self._full_check)
        self._fixed_check = QCheckBox("固定评估集（跨版本可比）")
        self._fixed_check.setChecked(True)
        row2.addWidget(self._fixed_check)
        row2.addStretch()
        dl.addLayout(row2)
        tl.addWidget(g_ds)

        g_prompt = QGroupBox("提示词（评估用）")
        pl = QVBoxLayout(g_prompt)
        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setMaximumHeight(150)
        pl.addWidget(self._prompt_edit)
        prow = QHBoxLayout()
        self._load_cfg_btn = QPushButton("载入配置页提示词")
        self._load_cfg_btn.clicked.connect(self._load_from_config)
        prow.addWidget(self._load_cfg_btn)
        self._save_cfg_btn = QPushButton("保存到配置页")
        self._save_cfg_btn.clicked.connect(self._save_to_config)
        prow.addWidget(self._save_cfg_btn)
        self._hash_label = QLabel("哈希: -")
        prow.addWidget(self._hash_label)
        prow.addStretch()
        pl.addLayout(prow)

        g_ctrl = QGroupBox("评估控制")
        cl = QVBoxLayout(g_ctrl)
        row_rpm = QHBoxLayout()
        row_rpm.addWidget(QLabel("评估频率:"))
        self._eval_rpm_spin = QSpinBox()
        self._eval_rpm_spin.setRange(1, 600)
        self._eval_rpm_spin.setValue(EVAL_RPM_DEFAULT)
        row_rpm.addWidget(self._eval_rpm_spin)
        row_rpm.addWidget(QLabel("张/分钟（独立于配置页，可用更高频率加速评估）"))
        row_rpm.addStretch()
        cl.addLayout(row_rpm)
        crow = QHBoxLayout()
        self._start_btn = QPushButton("▶ 开始评估")
        self._start_btn.setObjectName("primary")
        self._start_btn.clicked.connect(self._start)
        crow.addWidget(self._start_btn)
        self._cancel_btn = QPushButton("⏹ 取消")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        crow.addWidget(self._cancel_btn)
        crow.addStretch()
        cl.addLayout(crow)
        self._progress = QProgressBar()
        cl.addWidget(self._progress)
        tl.addWidget(g_ctrl)

        splitter.addWidget(top)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        g_res = QGroupBox("结果")
        rl = QVBoxLayout(g_res)
        self._metrics_label = QLabel("准确率: -")
        rl.addWidget(self._metrics_label)
        self._confusion_table = QTableWidget()
        self._confusion_table.setMaximumHeight(200)
        rl.addWidget(self._confusion_table)
        self._error_list = QListWidget()
        self._error_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._error_list.setIconSize(QSize(THUMB, THUMB))
        self._error_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._error_list.setWordWrap(True)
        self._error_list.itemDoubleClicked.connect(self._open_error_image)
        rl.addWidget(self._error_list)
        erow = QHBoxLayout()
        btn_csv = QPushButton("导出错误 CSV")
        btn_csv.clicked.connect(self._export_errors_csv)
        erow.addWidget(btn_csv)
        btn_json = QPushButton("导出完整结果 JSON")
        btn_json.clicked.connect(self._export_record_json)
        erow.addWidget(btn_json)
        erow.addStretch()
        rl.addLayout(erow)
        bl.addWidget(g_prompt)
        bl.addWidget(g_res)

        g_hist = QGroupBox("历史评估（选中两行可对比）")
        hl = QVBoxLayout(g_hist)
        self._history_table = QTableWidget()
        self._history_table.setColumnCount(5)
        self._history_table.setHorizontalHeaderLabels(
            ["时间", "提示词哈希", "样本数", "准确率", "平均置信度"]
        )
        self._history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hl.addWidget(self._history_table)
        hrow = QHBoxLayout()
        self._compare_btn = QPushButton("对比选中")
        self._compare_btn.clicked.connect(self._compare_selected)
        hrow.addWidget(self._compare_btn)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._reload_history)
        hrow.addWidget(btn_refresh)
        hrow.addStretch()
        hl.addLayout(hrow)
        tl.addWidget(g_hist)

        splitter.addWidget(bottom)
        splitter.setSizes([430, 590])
        outer.addWidget(splitter)

    # ── 配置注入 ──
    def set_config(self, api_key, model, categories, global_prompt, rpm, use_original,
                   category_keywords: dict | None = None, concurrency: int = 3):
        self._api_key = api_key
        self._model = model or get_provider("deepseek")["default_model"]
        self._categories = list(categories or [])
        self._rpm = rpm or 60
        self._use_original = bool(use_original)
        self._category_keywords = dict(category_keywords or {})
        self._concurrency = max(1, int(concurrency or 3))

        current = self._prompt_edit.toPlainText()
        # 编辑框为空或仍等于上次载入的配置提示词时才覆盖（避免丢弃临时编辑）
        if not current.strip() or current == self._config_prompt:
            self._prompt_edit.setPlainText(global_prompt or "")
        self._config_prompt = global_prompt or ""
        self._refresh_hash()

    def _refresh_hash(self):
        text = self._prompt_edit.toPlainText()
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        self._hash_label.setText(f"哈希: {h}")

    def _load_from_config(self):
        if self._config_prompt:
            self._prompt_edit.setPlainText(self._config_prompt)
            self._refresh_hash()

    def _save_to_config(self):
        callback = getattr(self, "save_to_config", None)
        if callable(callback):
            callback(self._prompt_edit.toPlainText())
            self._log_msg("提示词已保存到配置页")
        else:
            QMessageBox.information(self, "提示", "当前未连接到配置页")

    # ── 评估控制 ──
    def _browse_dataset(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据集根目录")
        if d:
            self._dataset_input.setText(d)

    def _log_msg(self, msg):
        self._metrics_label.setText(str(msg))

    def _start(self):
        root = Path(self._dataset_input.text().strip())
        if not root.is_dir():
            QMessageBox.warning(self, "错误", "请选择有效的数据集根目录")
            return
        if not self._categories:
            QMessageBox.warning(self, "错误", "分类列表为空，请先在配置页设置分类")
            return
        if not self._api_key:
            QMessageBox.warning(self, "错误", "请先在配置页输入 API 密钥")
            return

        use_fixed = self._fixed_check.isChecked() and not self._full_check.isChecked()
        fixed = None
        if use_fixed:
            fixed = self._store.load_eval_set(root.name)
            if fixed is None:
                by_cat = scan_dataset(root, self._categories)
                picked = pick_samples(by_cat, self._per_cat_spin.value(), False, None, root)
                if not picked:
                    QMessageBox.warning(self, "错误", "数据集为空：根目录下未找到与分类名一致的子目录")
                    return
                fixed = [str(p) for _, p in picked]
                self._store.save_eval_set(root.name, fixed)
                self._log_msg(
                    f"已生成固定评估集：{len(fixed)} 张（保存于 {self._store.data_dir()}）"
                )

        self._evaluator = Evaluator(
            "deepseek", self._api_key, self._model, str(root), self._categories,
            self._prompt_edit.toPlainText(), per_category=self._per_cat_spin.value(),
            full=self._full_check.isChecked(), use_original=self._use_original,
            rpm=self._eval_rpm_spin.value(), fixed=fixed,
            category_keywords=self._category_keywords,
            concurrency=self._concurrency,
        )
        self._evaluator.progress.connect(self._on_progress)
        self._evaluator.finished_record.connect(self._on_finished)
        self._evaluator.log.connect(self._log_msg)
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        self._evaluator.start()

    def _cancel(self):
        if self._evaluator:
            self._evaluator.cancel()
            self._cancel_btn.setEnabled(False)

    def _on_progress(self, done, total, name, pred, conf):
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._progress.setFormat(f"{done}/{total} (%p%)")
        self._log_msg(f"[{done}/{total}] {name} → {pred} ({conf:.2f})")

    def _on_finished(self, record):
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        if not record:
            self._log_msg("未完成（已取消或无数据）")
            return
        self._store.save_run(record)
        self._current_record = record
        self._render_metrics(record)
        self._render_confusion(record)
        self._render_errors(record)
        self._reload_history()

    # ── 结果渲染 ──
    def _render_metrics(self, record):
        self._metrics_label.setText(
            f"准确率: {record.get('accuracy', 0):.1f}%  "
            f"({record.get('correct', 0)}/{record.get('total', 0)})  |  "
            f"平均置信度: {record.get('avg_confidence', 0):.2f}  |  "
            f"耗时: {record.get('elapsed_seconds', 0):.0f}s  |  "
            f"Token: {record.get('total_tokens', 0):,}  |  "
            f"错误: {len(record.get('errors', []))} 张"
        )

    def _render_confusion(self, record):
        confusion = record.get("confusion", {})
        gts = sorted(confusion.keys())
        preds = sorted({p for m in confusion.values() for p in m})
        self._confusion_table.clear()
        self._confusion_table.setRowCount(len(gts))
        self._confusion_table.setColumnCount(len(preds) + 1)
        self._confusion_table.setHorizontalHeaderLabels(["期望\\实际"] + preds)
        self._confusion_table.setVerticalHeaderLabels(gts)
        for i, gt in enumerate(gts):
            row_total = sum(confusion[gt].values())
            for j, pred in enumerate(preds):
                n = confusion[gt].get(pred, 0)
                item = QTableWidgetItem(str(n) if n else "")
                if n and gt == pred:
                    item.setBackground(Qt.GlobalColor.green)
                self._confusion_table.setItem(i, j, item)
            self._confusion_table.setItem(i, len(preds), QTableWidgetItem(str(row_total)))

    def _thumb_icon(self, path: str) -> QIcon:
        try:
            img = Image.open(path)
            img.thumbnail((THUMB, THUMB))
            img = img.convert("RGB")
            qimg = QImage(img.tobytes(), img.width, img.height,
                          img.width * 3, QImage.Format.Format_RGB888)
            return QIcon(QPixmap.fromImage(qimg))
        except Exception:
            return QIcon()

    def _render_errors(self, record):
        self._error_list.clear()
        for err in record.get("errors", []):
            item = QListWidgetItem()
            item.setIcon(self._thumb_icon(err["path"]))
            item.setText(f"{Path(err['path']).name}\n{err['gt']} → {err['pred']} ({err['conf']:.2f})")
            item.setData(Qt.ItemDataRole.UserRole, err["path"])
            item.setToolTip(err["path"])
            self._error_list.addItem(item)

    def _open_error_image(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(path).exists():
            os.startfile(path)

    # ── 历史 ──
    def _reload_history(self):
        runs = self._store.load_runs()
        self._history_table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            cells = [r.get("timestamp", ""), r.get("prompt_hash", ""),
                     str(r.get("total", "")), f"{r.get('accuracy', 0):.1f}%",
                     f"{r.get('avg_confidence', 0):.2f}"]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j == 0:
                    item.setData(Qt.ItemDataRole.UserRole, r.get("id", ""))
                self._history_table.setItem(i, j, item)

    def _compare_selected(self):
        rows = sorted({idx.row() for idx in self._history_table.selectedIndexes()})
        if len(rows) != 2:
            QMessageBox.information(self, "提示", "请选中两行历史记录进行对比")
            return
        runs = self._store.load_runs()
        if max(rows) >= len(runs):
            QMessageBox.information(self, "提示", "历史记录已变化，请刷新后重试")
            return
        a, b = runs[rows[0]], runs[rows[1]]
        older, newer = (a, b) if a.get("id", "") < b.get("id", "") else (b, a)

        lines = [
            f"旧: {older.get('timestamp')}  准确率 {older.get('accuracy', 0):.1f}%  "
            f"(哈希 {older.get('prompt_hash')})",
            f"新: {newer.get('timestamp')}  准确率 {newer.get('accuracy', 0):.1f}%  "
            f"(哈希 {newer.get('prompt_hash')})",
            f"准确率变化: {newer.get('accuracy', 0) - older.get('accuracy', 0):+.1f} 个百分点",
            "",
            "混淆对变化（新 − 旧，按变化量前 10）:",
        ]
        old_c = older.get("confusion", {})
        new_c = newer.get("confusion", {})
        deltas = []
        for gt in set(old_c) | set(new_c):
            for pred in set(old_c.get(gt, {})) | set(new_c.get(gt, {})):
                d = new_c.get(gt, {}).get(pred, 0) - old_c.get(gt, {}).get(pred, 0)
                if gt != pred and d != 0:
                    deltas.append((abs(d), gt, pred, d))
        for _, gt, pred, d in sorted(deltas, reverse=True)[:10]:
            lines.append(f"  {gt} → {pred}: {d:+d}")
        if len(lines) == 5:
            lines.append("  （无变化）")
        QMessageBox.information(self, "历史对比", "\n".join(lines))

    # ── 导出 ──
    def _export_errors_csv(self):
        if not self._current_record or not self._current_record.get("errors"):
            QMessageBox.information(self, "提示", "没有错误样本")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出错误清单", "eval_errors.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["文件路径", "期望类别", "实际类别", "置信度"])
            for e in self._current_record["errors"]:
                w.writerow([e["path"], e["gt"], e["pred"], e["conf"]])
        QMessageBox.information(self, "完成", f"已导出到 {path}")

    def _export_record_json(self):
        if not self._current_record:
            QMessageBox.information(self, "提示", "暂无评估结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出评估结果", "eval_result.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(
            json.dumps(self._current_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        QMessageBox.information(self, "完成", f"已导出到 {path}")
