# 评估实验室（Evaluation Lab）设计文档

- 日期：2026-09-09
- 状态：已确认（关键点经问答定稿）
- 项目：`C:\Users\HP\projects\deepseek-image-classifier`
- 背景：`improve_prompt.py` 基准显示当前分类准确率仅 48.4%，且"加排除词"式改进无效
  （-0.56%）；根因诊断为分类体系维度冲突（画风"动漫"吸走题材类）。需要**内置的
  测试-反馈-修改-再测试闭环**来持续迭代提示词。

## 1. 目标

在主程序中内置「评估」页：用用户分好类的数据集（根目录下按类别名分子目录 = 标准
答案）批量测试当前提示词与分类体系，输出准确率、混淆矩阵、错误样本，并保存历史以
支持版本对比。形成"改提示词 → 跑测试 → 看反馈 → 再改"的闭环。

## 2. 用户已确认的关键决策

| 决策点 | 结论 |
|---|---|
| 抽样规模 | 每类张数可调（默认 15，5~100），另有「全量」勾选 |
| 历史存储 | 软件数据目录（`%APPDATA%/DeepSeekImageClassifier/eval/`） |
| 错误展示 | 缩略图列表 + 双击打开大图 |

## 3. 架构

### 3.1 新增 `app/vlm.py`（共享视觉调用层，消除重复）

现有 `Classifier._classify_one` / `_parse_response` 与 `config_tab` 的提示词构建
存在重复，评估模块需要同一套逻辑，故抽取：

```python
def build_prompt_text(global_prompt: str, categories: list[str],
                      category_keywords: dict[str, str]) -> str
    # 替换 {categories} 与 {category_definitions}

def parse_response(raw: str, categories: list[str]) -> tuple[str, float, list[str]]
    # 四级解析：JSON → || → 模糊 → 未整理（现 Classifier 实现迁移至此）

def classify_image(service: str, api_key: str, model: str, filepath: Path,
                   prompt_text: str, use_original: bool = False) -> tuple
    # 单图调用：返回 (category, confidence, keywords, raw, prompt_tokens, completion_tokens)
    # 含 3 次重试（429 退避）；内部使用 image_prep.prepare_image 或原图直传
```

`Classifier` 改为调用上述函数（行为不变，覆盖既有测试）。

### 3.2 新增 `app/eval_store.py`（历史与评估集持久化）

数据目录：`Path(os.environ["APPDATA"]) / "DeepSeekImageClassifier" / "eval"`

```python
class EvalStore:
    def __init__(self, base_dir: Path | None = None)   # None → 默认数据目录
    def data_dir(self) -> Path
    def save_eval_set(self, name: str, items: list[str]) -> Path   # eval_set_<name>.json
    def load_eval_set(self, name: str) -> list[str] | None
    def save_run(self, record: dict) -> Path                       # run_<ts>.json + 追加索引
    def load_runs(self) -> list[dict]                              # 按时间倒序
    def load_run(self, run_id: str) -> dict | None
```

记录结构：

```json
{
  "id": "20260909-153012",
  "timestamp": "2026-09-09 15:30:12",
  "dataset_root": "D:/.../kimi分类",
  "prompt_snapshot": "……",
  "prompt_hash": "ab12cd34",
  "sample_size": 165, "total": 165, "correct": 96,
  "accuracy": 58.2, "avg_confidence": 0.81,
  "elapsed_seconds": 240.5, "total_tokens": 123456,
  "confusion": {"政治": {"政治": 12, "动漫": 5}, "...": {}},
  "errors": [{"path": "...", "gt": "政治", "pred": "动漫", "conf": 0.83}],
  "categories": ["科技", "日常", "..."]
}
```

### 3.3 新增 `app/evaluator.py`（评估线程）

```python
class Evaluator(QThread):
    # 信号：progress(done, total, filename, gt, pred, conf)
    #       finished(dict record)、log(str)
    def __init__(self, service, api_key, model, dataset_root: str,
                 categories: list[str], prompt_text: str,
                 per_category: int = 15, full: bool = False,
                 use_original: bool = False, rpm: int = 60,
                 fixed_set: list[str] | None = None)
    def run(self)          # 扫描各类目录 → 抽样 → 逐张分类 → 统计 → emit record
    def cancel(self)
```

要点：
- **只分类不落盘**（不复制图片）——评估是 dry-run
- 类别取自数据集根目录下的子目录名（顺序稳定排序），也用于校验预测
- 抽样：`full=True` 用全部；否则每类随机 `per_category` 张（`random.sample`）
- 固定评估集：传入路径清单时按清单执行（保证跨版本可比）
- 混淆矩阵：`confusion[gt][pred] += 1`
- 尊重 RPM 限速与取消（沿用 Classifier 模式）

### 3.4 新增 `app/eval_tab.py`（评估页 UI）

自上而下：
1. **数据集组**：根目录（QLineEdit + 浏览 + 拖拽）、每类张数（QSpinBox 5~100 默认 15）、
   「全量」勾选、「固定评估集」勾选（默认开）
2. **提示词组**：QPlainTextEdit（载入配置页当前提示词）+ 「载入配置」「保存到配置」按钮
   + 提示词哈希/长度显示
3. **控制组**：开始评估 / 取消 / 进度条 / 预估调用数标签 / 「预估消耗」提示
4. **结果组**：
   - 指标标签：准确率、正确/总数、平均置信度、耗时、Token
   - 混淆矩阵 QTableWidget（行=期望、列=实际，对角线绿底；含"合计"行列）
   - 错误样本 QListWidget（IconMode：缩略图 96px + 文件名 + 期望→实际 + 置信度），
     双击用 `os.startfile` 打开原图
   - 导出按钮：错误 CSV、完整结果 JSON
5. **历史组**：QTableWidget（时间 / 提示词哈希 / 样本数 / 准确率 / 平均置信度），
   选中两行 → 「对比选中」→ 弹出对比结果（准确率差、混淆对改善/恶化前 10）

缩略图生成：`PIL.Image.open(...).thumbnail((96,96))` → 转 `QImage`/`QPixmap`；
错误样本通常 < 100 张，直接同步生成可接受。

### 3.5 `main_window.py`

- 新增第三个 Tab：`tabs.addTab(self._eval_tab, "评估")`
- 切到评估页时把当前配置（服务商固定 deepseek、Key、模型、分类、提示词、关键词、
  RPM、原图模式）推给评估页：`self._eval_tab.set_config(...)`

## 4. 数据流

```
数据集根/类别子目录 → 扫描 + 抽样（固定集复用 or 随机）
  → Evaluator 逐张：prepare_image(或原图) → API → parse_response
  → 与目录名(gt)比对 → confusion[gt][pred]++
  → record → EvalStore.save_run → UI（指标/矩阵/错误缩略图/历史）
  → 用户修改提示词 → 再跑 → 历史对比
```

## 5. 错误处理

- 数据集根不存在 / 无类别子目录 → 开始前校验并提示
- 单张 API 失败 → 记为 `pred="ERROR"`（计入混淆矩阵的 ERROR 列，不计正确）并继续
- 取消 → 保留已完成部分？**不保存**（避免污染历史），仅提示"已取消"
- 缩略图生成失败（损坏文件）→ 显示占位文本项
- 历史文件损坏 → 跳过该条并记录警告，不阻塞

## 6. 测试

- `tests/test_vlm.py`：`parse_response` 全路径（JSON/包裹文字/||/模糊/未整理/reason
  字段容忍）+ `build_prompt_text` 占位符替换
- `tests/test_eval_store.py`：评估集保存/读取、run 保存/列表/读取、目录自动创建、
  损坏 JSON 容错
- `tests/test_evaluator.py`：抽样（每类 N、全量、固定集）、混淆统计与准确率、
  API 失败计入 ERROR（monkeypatch `classify_image`）
- `tests/test_eval_tab_smoke.py`：offscreen 构建、默认值（15 张/固定集开/无结果）
- 回归：既有 37 个测试全部保持通过

## 7. 验收

1. 全量 pytest 通过
2. offscreen 冒烟：三 Tab 构建、评估页默认状态
3. 打包 `python build.py`
4. 用户真机验证：
   - 选 数据集根目录（或实际数据集）→ 每类 15 张 → 开始
   - 观察进度、完成后看准确率/混淆矩阵/错误缩略图
   - 改提示词（题材优先版）→ 再跑 → 历史对比看准确率变化
5. 合并 `feature/eval-lab` → main → push

## 8. 范围外（YAGNI）

- 多提示词并跑（一次一版，靠历史对比）
- 评估集管理 UI（新增/删除/重命名）
- LLM 自动改写提示词
- 结果云端同步
