"""评估页 — 数据集批量测试、反馈与历史对比

支持三种对比方式（可叠加）：
- 重复轮次：同一提示词跑 N 轮，用均值+极差对抗抽签式波动
- A/B 提示词：两个提示词轮内交替执行，事后按同一张图做配对比较
- 快速模式：同一提示词下「思考开」vs「思考关」
"""
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
    QComboBox,
)

from app.eval_batch import BatchEvaluator, Variant
from app.eval_stats import (aggregate, best_threshold, group_by_variant,
                            paired_compare, summarize_text, threshold_curve)
from app.eval_store import EvalStore
from app.evaluator import pick_samples, scan_dataset
from app.providers import get_provider
from app.vlm import FAST_MODE_BODY, build_prompt_text

THUMB = 96
EVAL_RPM_DEFAULT = 120
MAX_ROUNDS = 5


class EvalTab(QWidget):
    def __init__(self, parent=None, store: EvalStore | None = None):
        super().__init__(parent)
        self._store = store or EvalStore()
        self._evaluator = None
        self._current_record = None
        self._batch_records: list[dict] = []
        self._categories: list[str] = []
        self._api_key = ""
        self._model = get_provider("deepseek")["default_model"]
        self._rpm = 60
        self._use_original = False
        self._category_keywords: dict[str, str] = {}
        self._concurrency = 3
        self._category_mapping: dict = {}
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
        pl.addWidget(QLabel("提示词 A（或单变体提示词）："))
        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setMaximumHeight(120)
        pl.addWidget(self._prompt_edit)
        self._prompt_b_label = QLabel("提示词 B（A/B 对比时启用）：")
        pl.addWidget(self._prompt_b_label)
        self._prompt_edit_b = QPlainTextEdit()
        self._prompt_edit_b.setMaximumHeight(120)
        self._prompt_edit_b.setVisible(False)
        self._prompt_b_label.setVisible(False)
        pl.addWidget(self._prompt_edit_b)
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

        row_rounds = QHBoxLayout()
        row_rounds.addWidget(QLabel("重复轮次:"))
        self._rounds_spin = QSpinBox()
        self._rounds_spin.setRange(1, MAX_ROUNDS)
        self._rounds_spin.setValue(3)
        self._rounds_spin.setToolTip(
            "同一提示词重复跑几轮取平均。实测单轮波动可达 5–7pt，"
            "做提示词对比时少于 3 轮几乎无法分辨真实差异。"
        )
        self._rounds_spin.valueChanged.connect(self._update_plan_label)
        row_rounds.addWidget(self._rounds_spin)
        self._ab_check = QCheckBox("A/B 提示词对比")
        self._ab_check.toggled.connect(self._on_ab_toggled)
        row_rounds.addWidget(self._ab_check)
        self._fast_check = QCheckBox("对比快速模式（关闭思考）")
        self._fast_check.toggled.connect(self._update_plan_label)
        row_rounds.addWidget(self._fast_check)
        row_rounds.addStretch()
        cl.addLayout(row_rounds)

        self._plan_label = QLabel("")
        cl.addWidget(self._plan_label)

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
        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("查看变体:"))
        self._variant_combo = QComboBox()
        self._variant_combo.currentIndexChanged.connect(self._on_variant_changed)
        vrow.addWidget(self._variant_combo)
        vrow.addStretch()
        rl.addLayout(vrow)

        self._metrics_label = QLabel("准确率: -")
        self._metrics_label.setWordWrap(True)
        rl.addWidget(self._metrics_label)
        self._summary_label = QLabel("多轮统计: -")
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rl.addWidget(self._summary_label)
        self._confusion_table = QTableWidget()
        self._confusion_table.setMaximumHeight(160)
        rl.addWidget(self._confusion_table)

        rl.addWidget(QLabel("阈值扫描（置信度低于阈值的图会被分流进「待确认」）："))
        self._threshold_table = QTableWidget()
        self._threshold_table.setColumnCount(7)
        self._threshold_table.setHorizontalHeaderLabels(
            ["阈值", "挡错", "误挡", "挡错率", "误挡率", "分流精度", "保留准确率"]
        )
        self._threshold_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._threshold_table.setMaximumHeight(170)
        rl.addWidget(self._threshold_table)
        self._threshold_hint = QLabel("")
        self._threshold_hint.setWordWrap(True)
        rl.addWidget(self._threshold_hint)

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
        btn_batch = QPushButton("导出本批全部记录")
        btn_batch.clicked.connect(self._export_batch_json)
        erow.addWidget(btn_batch)
        erow.addStretch()
        rl.addLayout(erow)
        bl.addWidget(g_prompt)
        bl.addWidget(g_res)

        g_hist = QGroupBox("历史评估（选中两行可对比）")
        hl = QVBoxLayout(g_hist)
        self._history_table = QTableWidget()
        self._history_table.setColumnCount(6)
        self._history_table.setHorizontalHeaderLabels(
            ["时间", "变体", "轮次", "提示词哈希", "样本数", "准确率"]
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
        self._update_plan_label()

    # ── 配置注入 ──
    def set_config(self, api_key, model, categories, global_prompt, rpm, use_original,
                   category_keywords: dict | None = None, concurrency: int = 3,
                   category_mapping: dict | None = None):
        self._api_key = api_key
        self._model = model or get_provider("deepseek")["default_model"]
        self._categories = list(categories or [])
        self._rpm = rpm or 60
        self._use_original = bool(use_original)
        self._category_keywords = dict(category_keywords or {})
        self._concurrency = max(1, int(concurrency or 3))
        self._category_mapping = dict(category_mapping or {})
        self._update_plan_label()

        current = self._prompt_edit.toPlainText()
        # 编辑框为空或仍等于上次载入的配置提示词时才覆盖（避免丢弃临时编辑）
        if not current.strip() or current == self._config_prompt:
            self._prompt_edit.setPlainText(global_prompt or "")
        self._config_prompt = global_prompt or ""
        self._refresh_hash()

    def _on_ab_toggled(self, checked):
        self._prompt_edit_b.setVisible(checked)
        self._prompt_b_label.setVisible(checked)
        self._update_plan_label()

    def _update_plan_label(self):
        n_var = self._variant_count()
        rounds = self._rounds_spin.value()
        parts = [
            f"计划：{n_var} 个变体 × {rounds} 轮 = 每个样本请求 {n_var * rounds} 次"
        ]
        if n_var > 1 and rounds > 1:
            parts.append("轮内交替执行，抵消时间漂移")
        merged = {src: tgt for src, tgt in (self._category_mapping or {}).items()}
        if merged:
            m = "、".join(f"{src}→{tgt or '排除'}" for src, tgt in merged.items())
            parts.append(f"类别归并：{m}")
        parts.append(f"共 {len(self._categories)} 个类别")
        self._plan_label.setText("　｜　".join(parts))

    def _variant_count(self) -> int:
        n = 2 if self._ab_check.isChecked() else 1
        return n * (2 if self._fast_check.isChecked() else 1)

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

    def build_variants(self) -> list[Variant]:
        """按界面开关组装待对比变体（A/B 与快速模式可叠加）"""
        cats, kws = self._categories, self._category_keywords
        ab = self._ab_check.isChecked()
        variants = [Variant(
            "A" if ab else "现状",
            build_prompt_text(self._prompt_edit.toPlainText(), cats, kws),
        )]
        if ab:
            variants.append(Variant(
                "B", build_prompt_text(self._prompt_edit_b.toPlainText(), cats, kws)))
        if self._fast_check.isChecked():
            variants = variants + [
                Variant(f"{v.name}+快速", v.prompt_text, FAST_MODE_BODY) for v in variants
            ]
        return variants

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
        if self._ab_check.isChecked() and not self._prompt_edit_b.toPlainText().strip():
            QMessageBox.warning(self, "错误", "A/B 对比需要填写提示词 B")
            return

        use_fixed = self._fixed_check.isChecked() and not self._full_check.isChecked()
        fixed = None
        if use_fixed:
            fixed = self._store.load_eval_set(root.name)
            if fixed is None:
                by_cat = scan_dataset(root, self._categories, self._category_mapping)
                picked = pick_samples(by_cat, self._per_cat_spin.value(), False, None, root,
                                      self._category_mapping)
                if not picked:
                    QMessageBox.warning(self, "错误", "数据集为空：根目录下未找到与分类名一致的子目录")
                    return
                fixed = [str(p) for _, p in picked]
                self._store.save_eval_set(root.name, fixed)
                self._log_msg(
                    f"已生成固定评估集：{len(fixed)} 张（保存于 {self._store.data_dir()}）"
                )

        self._evaluator = BatchEvaluator(
            "deepseek", self._api_key, self._model, str(root), self._categories,
            self.build_variants(), rounds=self._rounds_spin.value(),
            per_category=self._per_cat_spin.value(),
            full=self._full_check.isChecked(), use_original=self._use_original,
            rpm=self._eval_rpm_spin.value(), fixed=fixed,
            concurrency=self._concurrency,
            category_mapping=self._category_mapping,
        )
        self._evaluator.progress.connect(self._on_progress)
        self._evaluator.round_finished.connect(self._on_round_finished)
        self._evaluator.finished_batch.connect(self._on_finished)
        self._evaluator.log.connect(self._log_msg)
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        self._batch_records = []
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

    def _on_round_finished(self, record):
        self._log_msg(
            f"第 {record.get('round')} 轮 · {record.get('variant')}: "
            f"{record.get('accuracy', 0):.1f}%"
        )

    def _on_finished(self, records):
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        if not records:
            self._log_msg("未完成（已取消或无数据）")
            return
        self._store.save_batch(records)
        self._batch_records = list(records)
        self._current_record = records[-1]
        self._reload_variant_combo()
        self._reload_history()

    # ── 结果渲染 ──
    def _reload_variant_combo(self):
        names = list(group_by_variant(self._batch_records))
        self._variant_combo.blockSignals(True)
        self._variant_combo.clear()
        self._variant_combo.addItems(names)
        self._variant_combo.blockSignals(False)
        if names:
            self._on_variant_changed(0)

    def _records_for_current_variant(self) -> list[dict]:
        name = self._variant_combo.currentText()
        groups = group_by_variant(self._batch_records)
        if name in groups:
            return groups[name]
        return self._batch_records

    def _on_variant_changed(self, _index):
        recs = self._records_for_current_variant()
        if not recs:
            return
        self._current_record = recs[-1]
        self._render_metrics(recs)
        self._render_confusion(self._current_record)
        self._render_errors(self._current_record)
        self._render_threshold(recs)

    def _render_metrics(self, recs):
        agg = aggregate(recs)
        last = recs[-1]
        text = (
            f"准确率: {agg['accuracy_mean']:.1f}%（{agg['rounds']} 轮均值，"
            f"极差 {agg['accuracy_range']:.1f}pt）  |  "
            f"最近一轮: {last.get('correct', 0)}/{last.get('total', 0)}  |  "
            f"耗时均值: {agg['elapsed_mean']:.0f}s  |  "
            f"Token: {agg['total_tokens']:,}"
        )
        self._metrics_label.setText(text)
        self._summary_label.setText("多轮统计:\n  " + summarize_text(recs).replace("\n", "\n  "))

        groups = group_by_variant(self._batch_records)
        if len(groups) == 2:
            (na, ra), (nb, rb) = list(groups.items())
            cmp = paired_compare(ra, rb)
            verdict = "显著" if cmp["significant"] else "不显著"
            self._summary_label.setText(
                self._summary_label.text()
                + f"\n\n配对比较 A={na} → B={nb}：\n"
                  f"  {na} {cmp['a']['accuracy_mean']:.1f}% vs {nb} {cmp['b']['accuracy_mean']:.1f}%"
                  f"（差 {cmp['delta']:+.1f}pt，配对 {cmp['pairs']} 张）\n"
                  f"  只{na}对 {cmp['discordant']['a_only']} ｜ 只{nb}对 "
                  f"{cmp['discordant']['b_only']} ｜ p={cmp['p_value']:.3f} → {verdict}"
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

    def _render_threshold(self, recs):
        rows = threshold_curve(recs)
        self._threshold_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            cells = [f"{row['threshold']:.2f}", str(row["blocked_errors"]),
                     str(row["blocked_correct"]), f"{row['error_block_rate']:.1f}%",
                     f"{row['correct_block_rate']:.1f}%", f"{row['bucket_precision']:.1f}%",
                     f"{row['kept_accuracy']:.1f}%"]
            for j, text in enumerate(cells):
                self._threshold_table.setItem(i, j, QTableWidgetItem(text))
        pick = best_threshold(recs) if rows else {"threshold": None, "reason": ""}
        current = self._low_conf_value()
        if pick.get("threshold") is not None:
            self._threshold_hint.setText(
                f"建议阈值 {pick['threshold']:.2f}（挡错 {pick['blocked_errors']}、"
                f"误挡 {pick['blocked_correct']}）；配置页当前为 {current:.2f}"
            )
        elif rows:
            self._threshold_hint.setText(pick.get("reason", ""))
        else:
            self._threshold_hint.setText("该记录缺少逐样本置信度明细，无法做阈值分析")

    def _low_conf_value(self) -> float:
        getter = getattr(self, "get_current_threshold", None)
        try:
            return float(getter()) if callable(getter) else 0.6
        except Exception:
            return 0.6

    def _open_error_image(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(path).exists():
            os.startfile(path)

    # ── 历史 ──
    def _reload_history(self):
        runs = self._store.load_runs()
        self._history_table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            cells = [r.get("timestamp", ""), r.get("variant", "—"),
                     f"{r.get('round', 1)}/{r.get('rounds_total', 1)}",
                     r.get("prompt_hash", ""), str(r.get("total", "")),
                     f"{r.get('accuracy', 0):.1f}%"]
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

        def _label(r):
            return (f"{r.get('timestamp')} [{r.get('variant', '—')} "
                    f"r{r.get('round', 1)}] {r.get('accuracy', 0):.1f}% "
                    f"(哈希 {r.get('prompt_hash')})")

        lines = [f"旧: {_label(older)}", f"新: {_label(newer)}",
                 f"准确率变化: {newer.get('accuracy', 0) - older.get('accuracy', 0):+.1f} 个百分点",
                 ""]
        if newer.get("samples") and older.get("samples"):
            cmp = paired_compare([older], [newer])
            verdict = "显著" if cmp["significant"] else "不显著（差异在噪声内）"
            lines += [
                f"配对分析（按同一张图）: {cmp['pairs']} 张",
                f"  只旧对 {cmp['discordant']['a_only']} ｜ 只新对 {cmp['discordant']['b_only']}"
                f" ｜ 都对 {cmp['both_ok']} ｜ 都错 {cmp['both_bad']}",
                f"  二项检验 p = {cmp['p_value']:.3f} → {verdict}",
                "",
            ]
        lines.append("混淆对变化（新 − 旧，按变化量前 10）:")
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
        if not deltas:
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

    def _export_batch_json(self):
        if not self._batch_records:
            QMessageBox.information(self, "提示", "暂无本次批次结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出本批记录", "eval_batch.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(
            json.dumps(self._batch_records, ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "完成", f"已导出 {len(self._batch_records)} 条记录到 {path}")
