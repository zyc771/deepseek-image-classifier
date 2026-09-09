"""运行页 — 进度、日志、结果统计"""
import csv
import os
import subprocess
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QPlainTextEdit, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QThread
from app.classifier import Classifier


class RunTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._classifier: Classifier | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── 进度信息 ──
        group_progress = QGroupBox("进度")
        gp = QVBoxLayout(group_progress)
        self._progress_bar = QProgressBar()
        gp.addWidget(self._progress_bar)
        info_row = QHBoxLayout()
        self._speed_label = QLabel("速度: -")
        self._eta_label = QLabel("预计剩余: -")
        self._count_label = QLabel("成功: 0 | 失败: 0")
        info_row.addWidget(self._speed_label)
        info_row.addWidget(self._eta_label)
        info_row.addWidget(self._count_label)
        info_row.addStretch()
        gp.addLayout(info_row)
        layout.addWidget(group_progress)

        # ── 控制按钮 ──
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶ 开始分类")
        self._start_btn.setStyleSheet("QPushButton{background:#4CAF50;color:white;padding:8px 24px;font-size:14px;}")
        self._start_btn.clicked.connect(self._start)
        btn_row.addWidget(self._start_btn)

        self._pause_btn = QPushButton("⏸ 暂停")
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._pause)
        btn_row.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("⏹ 取消")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── 日志 ──
        group_log = QGroupBox("日志")
        gl = QVBoxLayout(group_log)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        gl.addWidget(self._log)
        layout.addWidget(group_log)

        # ── 统计 ──
        group_stats = QGroupBox("统计 (完成后显示)")
        gs = QVBoxLayout(group_stats)
        self._stats_table = QTableWidget()
        self._stats_table.setColumnCount(5)
        self._stats_table.setHorizontalHeaderLabels(["类别", "数量", "占比", "置信度均值", "关键词"])
        self._stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        gs.addWidget(self._stats_table)

        token_row = QHBoxLayout()
        self._token_label = QLabel("Token: - | 耗时: -")
        token_row.addWidget(self._token_label)
        token_row.addStretch()
        gs.addLayout(token_row)
        layout.addWidget(group_stats)

        # ── 操作按钮 ──
        action_row = QHBoxLayout()
        self._open_btn = QPushButton("打开输出文件夹")
        self._open_btn.clicked.connect(self._open_output)
        action_row.addWidget(self._open_btn)
        self._export_btn = QPushButton("导出失败列表 CSV")
        self._export_btn.clicked.connect(self._export_csv)
        action_row.addWidget(self._export_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        self._success = 0
        self._failed = 0
        self._output_dir = ""
        self._failed_items: list[tuple[str, str]] = []  # (filename, error)
        self._start_time = None

    def set_config(self, api_key, src, out, categories, global_prompt, category_keywords, rpm,
                   use_original=False, low_conf=0.6):
        self._config = (api_key, src, out, categories, global_prompt, category_keywords, rpm,
                        use_original, low_conf)
        self._output_dir = out

    # ── 分类控制 ──
    def _start(self):
        if self._classifier and self._classifier.isRunning():
            QMessageBox.warning(self, "提示", "分类正在进行中，请先等待完成或取消")
            return
        (api_key, src, out, categories, global_prompt, category_keywords, rpm,
         use_original, low_conf) = self._config
        if not api_key:
            QMessageBox.warning(self, "错误", "请先在配置页输入 API 密钥")
            return
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "错误", "源文件夹不存在")
            return
        if not categories:
            QMessageBox.warning(self, "错误", "请至少设置一个分类")
            return

        import time as _time
        self._start_time = _time.time()
        self._success = 0
        self._failed = 0
        self._failed_items = []

        from app.providers import get_provider
        self._classifier = Classifier(
            "deepseek", api_key, get_provider("deepseek")["default_model"],
            src, out, categories, global_prompt, category_keywords, rpm,
            use_original=use_original, low_conf_threshold=low_conf,
        )
        self._classifier.signals.scan_done.connect(self._on_scan)
        self._classifier.signals.progress.connect(self._on_progress)
        self._classifier.signals.finished.connect(self._on_finished)
        self._classifier.signals.log.connect(self._add_log)

        self._progress_bar.setValue(0)
        self._stats_table.setRowCount(0)
        self._log.clear()
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)

        self._classifier.start()

    def _pause(self):
        if self._classifier:
            if self._classifier._paused:
                self._classifier.resume()
                self._pause_btn.setText("⏸ 暂停")
            else:
                self._classifier.pause()
                self._pause_btn.setText("▶ 继续")

    def _cancel(self):
        if self._classifier:
            self._classifier.cancel()
            self._cancel_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setText("取消中…")

    def _reset_buttons(self):
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("⏹ 取消")
        self._pause_btn.setText("⏸ 暂停")

    # ── 信号处理 ──
    def _on_scan(self, total, pending):
        self._add_log(f"扫描完成: 共 {total} 张, 待处理 {pending} 张")

    def _on_progress(self, idx, total, filename, category, confidence, keywords, pt, ct):
        import time as _time
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(idx)
        self._progress_bar.setFormat(f"{idx}/{total} (%p%)")

        if category == "ERROR":
            self._failed += 1
            self._failed_items.append((filename, "API调用失败"))
        else:
            self._success += 1

        self._count_label.setText(f"成功: {self._success} | 失败: {self._failed}")

        elapsed = _time.time() - self._start_time
        rate = (self._success + self._failed) / elapsed * 60 if elapsed > 0 else 0
        remaining = total - idx
        eta_min = remaining / rate if rate > 0 else 0
        self._speed_label.setText(f"速度: {rate:.1f}张/分")
        self._eta_label.setText(f"预计剩余: {eta_min:.0f}分")

        conf_str = f"({confidence:.2f})" if category != "ERROR" else ""
        self._add_log(f"[{idx}/{total}] {filename} → {category}{conf_str}")

    def _on_finished(self, summary):
        self._reset_buttons()
        if summary.get("cancelled"):
            self._add_log("已取消")
            self._progress_bar.setValue(self._progress_bar.value())
            return
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._add_log("\n=== 完成 ===")
        if summary.get("pending_review"):
            self._add_log(f"⚠ {summary['pending_review']} 张低置信度图片已放入『待确认』文件夹")

        cats = summary.get("categories", {})
        self._stats_table.setRowCount(len(cats))
        for i, (cat, data) in enumerate(sorted(cats.items(), key=lambda x: -x[1]["count"])):
            self._stats_table.setItem(i, 0, QTableWidgetItem(cat))
            self._stats_table.setItem(i, 1, QTableWidgetItem(str(data["count"])))
            self._stats_table.setItem(i, 2, QTableWidgetItem(f"{data['percentage']:.1f}%"))
            self._stats_table.setItem(i, 3, QTableWidgetItem(f"{data['avg_confidence']:.2f}"))
            self._stats_table.setItem(i, 4, QTableWidgetItem(", ".join(data.get("top_keywords", []))))

        tokens = summary.get("total_tokens", 0)
        elapsed = summary.get("elapsed_seconds", 0)
        self._token_label.setText(
            f"总Token: {tokens:,} | 输入: {summary.get('total_prompt_tokens',0):,}"
            f" | 输出: {summary.get('total_completion_tokens',0):,}"
            f" | 耗时: {elapsed/60:.0f}分钟 ({elapsed:.0f}秒)"
        )

    def _add_log(self, msg):
        self._log.appendPlainText(msg)

    # ── 操作 ──
    def _open_output(self):
        if self._output_dir and Path(self._output_dir).exists():
            os.startfile(self._output_dir)
        elif self._output_dir:
            QMessageBox.warning(self, "错误", "输出目录不存在")

    def _export_csv(self):
        if not self._failed_items:
            QMessageBox.information(self, "提示", "没有失败项")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出失败列表", "failed.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["文件名", "错误信息"])
                w.writerows(self._failed_items)
            QMessageBox.information(self, "完成", f"已导出 {len(self._failed_items)} 条到 {path}")


from pathlib import Path  # noqa: E402
