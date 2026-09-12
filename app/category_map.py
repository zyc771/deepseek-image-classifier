"""类别归并映射 — 评估时把若干细分类目录合并成一个粗分类（不改动任何文件）

动机：用户可能希望把细分类合并（如 历史/政治/军事 → 史政），或把某些目录
排除在评估之外（如 节假日/黄）。直接搬移原始素材是破坏性的，因此这里提供
一层「目录名 → 目标类别」的映射，只在评估读取标准答案时生效。

配置格式（多行，行内也可用分号分隔；`#` 开头为注释）：

    史政 = 历史, 政治, 军事
    排除 = 动漫, 节假日, 黄

- 等号左边是**目标类别名**（会出现在评估报告与预测解析里）
- 等号右边是该类别的**来源目录名**列表
- 目标写成「排除」或以 `-` 开头，表示这些目录**不参与评估**
- 未在映射中出现的目录，一律按目录名本身当类别（行为与以前完全一致）
"""

# 目标类别的等价写法：都表示「这些来源目录不参与评估」
_EXCLUDE_TARGETS = {"排除", "忽略", "exclude", "-"}

# 分隔符归一：中文标点一律当作英文标点处理
_NORMALIZE = str.maketrans({"，": ",", "；": ";", "＝": "=", "：": ":", "、": ","})


def _clean(token: str) -> str:
    return token.strip().strip("\"'").strip()


def parse_alias(text: str | None) -> dict:
    """解析映射配置 → `{源目录名: 目标类别}`；值为 None 表示排除该目录"""
    if not text:
        return {}
    mapping: dict = {}
    for raw_line in str(text).translate(_NORMALIZE).replace("\r", "\n").split("\n"):
        for chunk in raw_line.split(";"):
            line = chunk.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line and "=" not in line:
                line = line.replace(":", "=", 1)
            if "=" in line:
                target, sources = line.split("=", 1)
            else:
                # 没有等号 → 整行当排除列表（`-动漫, 节假日` 这种简写）
                target, sources = "排除", line.lstrip("-").strip()

            target = _clean(target)
            names = [_clean(s) for s in sources.split(",")]
            names = [n for n in names if n]
            if not names:
                continue

            value = None if (target in _EXCLUDE_TARGETS or target.startswith("-")) else target
            for name in names:
                mapping[name] = value
    return mapping


def alias_of(category: str, mapping: dict | None) -> str | None:
    """目录名 → 目标类别；未配置时原样返回，被排除时返回 None"""
    if not mapping:
        return category
    return mapping.get(category, category)


def sources_for(target: str, mapping: dict | None) -> list[str]:
    """目标类别 → 需要扫描的目录名列表（含自身，且不含被排除的目录）

    用于反向查找：配置里写的是「史政」，但数据集里只有 历史/政治/军事 三个目录。
    """
    dirs = [target]
    for src, tgt in (mapping or {}).items():
        if tgt is not None and tgt == target and src != target:
            dirs.append(src)
    return dirs


def excluded_dirs(mapping: dict | None) -> list[str]:
    """被排除、不参与评估的目录名"""
    return [src for src, tgt in (mapping or {}).items() if tgt is None]


def target_categories(categories: list[str], mapping: dict | None) -> list[str]:
    """把映射里声明、但配置分类列表中没有的目标类别补进去（去重，保持顺序）"""
    out = list(categories or [])
    for tgt in (mapping or {}).values():
        if tgt is not None and tgt not in out:
            out.append(tgt)
    return out
