# 交接摘要（新会话用这个继续）

> 用法：新会话第一句话发：「读 C:\Users\HP\projects\deepseek-image-classifier\docs\HANDOFF.md，然后继续」

## 项目

- 路径：`C:\Users\HP\projects\deepseek-image-classifier`
- GitHub：`zyc771/deepseek-image-classifier`（私有，main 分支）
- 定位：DeepSeek 单模型版图片分类工具（上游双服务商版为 `image-classifier-gui`）
- 状态：116 个测试全绿；exe 已打包于 `dist/`；桌面快捷方式「DeepSeek图片分类工具」指向该 exe

## 已完成

- 分类：递归扫描、EXIF 校正 + 1024 缩图 + q85、并发（默认 3）+ 时隙限速、连接复用、判重（名+大小+mtime）、重名加序号、低置信度分流（`待确认/`）、失败重试、暂停/取消
- 配置：DPAPI 加密存 Key、分类/关键词/提示词/频率/并发可配、单张预览测试
- 评估实验室：数据集（递归、固定评估集）、混淆矩阵、错误缩略图、历史对比、导出
- 界面：浅色/夜间双主题（`app/theme.py`）、三页左右分栏、应用图标（`resources/app.ico`，底色 `#EEF1F6`）
- 工具：`tools/make_icon.py --bg "#色值"` 换图标底色后需 `python build.py`

## 待办（按优先级）

1. **评估重复轮次**：同提示词同数据下总准确率波动约 ±2pt（单类可达 ±10pt），需自动跑 2-3 轮取平均，提示词 A/B 才有意义
2. **置信度阈值优化**：评估应记录**全部样本**置信度并扫描阈值，输出「挡错率 / 误挡率」曲线，为分流出最优阈值（实测错误样本约 70% 置信度 ≥0.8，"自信的错误"靠 0.6 阈值挡不住）
3. **快速模式（可选）**：尝试关闭/降低模型思考（推理 token 占输出大头），需先验证 API 参数，做成失败自动回退的开关
4. 提示词优化：实测"规则越多越容易负迁移"，建议做"简洁版 vs 详细版"的多轮 A/B（配合待办 1）

## 工作约定

- 测试：`$env:TEMP="C:\Users\HP\Desktop\deepseek\.pytest-tmp"; python -m pytest tests/ -q -p no:cacheprovider`（沙箱需 temp 指向工作区）
- 打包：`python build.py`（打包前先结束正在运行的工具进程，否则 PermissionError）
- 推送：GitHub 直连不稳定，用 `git -c http.proxy="http://127.0.0.1:7890" push origin main`（Clash）
- 流程：改动走特性分支 → 合并 main → 推送；TDD（先写失败测试）
- 图标改动后需清图标缓存：停 explorer → 删 `%LOCALAPPDATA%\Microsoft\Windows\Explorer\iconcache_*.db` → 启 explorer

## 用户偏好

- 中文交流，回复简洁（表格优先），先给方案再动手
- 数据驱动：改动后用评估实验室的真实数据验证效果，不要凭感觉下结论
- 参考数据：276 张 11 类固定评估集，准确率 58%~66%；性能基线 45 张/分（3 并发 + 120 RPM）
