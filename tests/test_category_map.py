"""类别归并映射测试 — 纯函数，不改动任何文件"""
from app.category_map import (alias_of, excluded_dirs, parse_alias,
                              sources_for, target_categories)

TEXT = """
史政 = 历史, 政治, 军事
排除 = 动漫, 节假日, 黄
"""


class TestParseAlias:
    def test_basic_mapping(self):
        m = parse_alias(TEXT)
        assert m["历史"] == "史政"
        assert m["政治"] == "史政"
        assert m["军事"] == "史政"

    def test_exclusions_map_to_none(self):
        m = parse_alias(TEXT)
        assert m["动漫"] is None
        assert m["节假日"] is None
        assert m["黄"] is None

    def test_unlisted_dir_absent(self):
        assert "科技" not in parse_alias(TEXT)

    def test_empty_text(self):
        assert parse_alias("") == {}
        assert parse_alias(None) == {}
        assert parse_alias("   \n  ") == {}

    def test_semicolon_separated(self):
        m = parse_alias("史政 = 历史, 政治; 排除 = 动漫")
        assert m == {"历史": "史政", "政治": "史政", "动漫": None}

    def test_chinese_punctuation_tolerated(self):
        """用户很可能用中文逗号/分号/等号"""
        m = parse_alias("史政＝历史，政治，军事；排除＝动漫")
        assert m["政治"] == "史政" and m["动漫"] is None

    def test_comments_and_blank_lines_ignored(self):
        m = parse_alias("# 这是注释\n\n史政 = 历史\n\n")
        assert m == {"历史": "史政"}

    def test_space_style_target(self):
        """『排除』与『-』都表示排除"""
        assert parse_alias("-动漫,节假日")["动漫"] is None
        assert parse_alias("排除: 动漫")["动漫"] is None

    def test_source_list_with_spaces(self):
        m = parse_alias("史政 = 历史 ,  政治 ")
        assert set(m) == {"历史", "政治"}

    def test_duplicate_source_last_wins(self):
        m = parse_alias("史政 = 历史\n世界 = 历史")
        assert m["历史"] == "世界"

    def test_empty_source_list_ignored(self):
        assert parse_alias("史政 =") == {}


class TestAliasOf:
    def test_mapped_and_unmapped(self):
        m = parse_alias(TEXT)
        assert alias_of("历史", m) == "史政"
        assert alias_of("科技", m) == "科技"      # 未配置的目录原样保留
        assert alias_of("动漫", m) is None        # 排除

    def test_none_mapping_is_identity(self):
        assert alias_of("历史", None) == "历史"

    def test_empty_mapping_is_identity(self):
        assert alias_of("历史", {}) == "历史"


class TestSourcesFor:
    def test_target_collects_its_sources(self):
        m = parse_alias(TEXT)
        assert sources_for("史政", m) == ["史政", "历史", "政治", "军事"]

    def test_plain_category_returns_itself(self):
        m = parse_alias(TEXT)
        assert sources_for("科技", m) == ["科技"]

    def test_never_includes_excluded_dirs(self):
        m = parse_alias(TEXT)
        for cat in ("史政", "科技", "动漫"):
            assert "动漫" not in sources_for(cat, m) or cat == "动漫"

    def test_target_order_is_stable(self):
        m = parse_alias(TEXT)
        assert sources_for("史政", m) == sources_for("史政", m)


class TestExcludedDirs:
    def test_lists_only_excluded(self):
        m = parse_alias(TEXT)
        assert set(excluded_dirs(m)) == {"动漫", "节假日", "黄"}

    def test_empty(self):
        assert excluded_dirs(None) == []


class TestTargetCategories:
    def test_merges_declared_targets_with_given_categories(self):
        m = parse_alias(TEXT)
        cats = target_categories(["科技", "史政"], m)
        assert cats == ["科技", "史政"]

    def test_adds_undeclared_targets(self):
        """映射里声明了目标类但配置里没写 → 补进去，否则评估会漏掉它"""
        m = parse_alias("史政 = 历史, 政治")
        assert target_categories(["科技"], m) == ["科技", "史政"]

    def test_exclusions_are_not_categories(self):
        m = parse_alias(TEXT)
        cats = target_categories(["科技"], m)
        assert "动漫" not in cats and "排除" not in cats
