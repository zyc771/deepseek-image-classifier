# 识图性能改进（准确性 + 速度）设计文档

- 日期：2026-09-09
- 状态：已确认（A 参数经理论分析结论定稿）
- 项目：`C:\Users\HP\projects\deepseek-image-classifier`
- 分支策略：当前 main 上新建 `feature/performance`，完成后合并

## 1. 目标

改进工具识图性能：**速度/成本**（图像预处理管线）与**准确性**（输出格式、
提示词、低置信度分流），保持既有功能不变。

## 2. 方案 A — 图像预处理管线（速度/成本 ↑，准确率 ≈ 不变或略升）

**理论分析结论**（2026-09-09）：EXIF 校正是净增益（修复现在竖拍图可能
"横躺"上传的 bug）；缩图到 1024px 语义信息完整保留（模型内部通常按
1024-2048 截断处理，超限原图本身就会被内部降采样，白费 token）；q85 感知
无损。

实现：
- 新增 `app/image_prep.py`：`prepare_image(path) -> tuple[str, str]` 返回
  `(mime_ext, b64)`：
  1. `Image.open` + `ImageOps.exif_transpose`（修正相机方向）
  2. 仅当 `max(w,h) > 1024` 时 `LANCZOS` 缩图至 1024 最大边
  3. RGBA 透明合成白底 → RGB；JPEG `quality=85` 编码 → base64
- `Classifier` 新增 `use_original: bool = False`：为 True 时**跳过整个管线**
  （与原版行为一致，base64 直传原字节）
- 配置页新增勾选框「原图模式」（保存于 QSettings `use_original`）

## 3. 方案 B — 输出格式与提示词优化（准确性 ↑）

1. 模型优先输出 **JSON**：
   `{"category": "体育", "confidence": 0.93, "keywords": ["篮球", "比赛"]}`
2. 解析器 `_parse_response` 升级为四级：**JSON → || 分隔 → 模糊匹配 → 未整理**，
   旧格式与当前提示词生成的输出仍可解析（向后兼容，解析器全链路测试）
3. 新默认提示词 `DEFAULT_GLOBAL_PROMPT`（仅新安装/点"恢复默认"生效；
   用户已有自定义提示词不受影响——提示词保留在配置中，由用户自行决定何时
   使用新模板）包含：
   - 判定优先级：画面主体 > 背景 > 文字
   - 置信度语义：不确定/多义/画面信息不足 → 给 0.6 以下
   - JSON 输出规则 + 示例
4. 解析失败兜底不变（ERROR 记失败）

## 4. 方案 C — 低置信度分流（准确性 ↑，人工可控）

- `Classifier` 新增 `low_conf_threshold: float = 0.6`
- 分类置信度 < 阈值 → 归入输出目录 `待确认/` 文件夹（不进入类别文件夹）
- summary 增加 `"pending_review": N`；统计表增加「待确认」行（数量/占比）
- 配置页新增「低置信度阈值」数值输入（0.5~0.9，步长 0.05，默认 0.6，
  保存于 `low_conf_threshold`）

## 5. 接口变更

- `ConfigTab`：新增 `get_use_original() -> bool`、`get_low_conf() -> float`
- `RunTab.set_config(api_key, src, out, categories, global_prompt,
  category_keywords, rpm, use_original=False, low_conf=0.6)`
- `Classifier(service, api_key, model, source_dir, output_dir, categories,
  global_prompt, category_keywords, rpm, use_original=False,
  low_conf_threshold=0.6)`；`preview()` 同样透传
- `MainWindow` cfg 元组扩为 9 项并同步状态栏

## 6. 测试

- `test_image_prep.py`：EXIF 反转图校正方向、>1024 缩图、≤1024 不缩、
  RGBA 白底合成、原图模式跳过管线、编码有效性与尺寸边界
- `test_classifier.py` 增补：JSON 解析（标准/缺字段/非法 JSON→回退）、
  低置信度分流（< 阈值 → 待确认；≥ 阈值 → 类别）
- `test_config_tab_smoke.py`：新增开关/阈值默认值与持久化
- 既有 23 测试保持通过

## 7. 验收

1. 全量 pytest 通过
2. offscreen 冒烟（默认新参数 + 既有配置兼容）
3. `python build.py` 打包、桌面快捷方式不变
4. 用户真机验证：预览几张（重点：竖拍照片 + 表情包）→ 分类正常；
   低置信度图进入"待确认"；取消勾选原图模式对比
5. 合并 `feature/performance` → main → push
