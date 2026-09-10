"""classifier 行为测试（不启动线程、不发网络请求）"""
import shutil
from pathlib import Path
from app.classifier import Classifier, base_name, scan_images, unique_target


def _make_clf(tmp_path: Path, source_dir: Path, output_dir: Path, **kw):
    return Classifier(
        service=kw.pop("service", "deepseek"),
        api_key="test-key",
        model="test-model",
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        categories=["科技", "日常", "动漫"],
        global_prompt="prompt {categories}",
        **kw,
    )


class TestParseResponse:
    def test_standard_format(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("动漫||0.85||火影,鸣人")
        assert cat == "动漫"
        assert conf == 0.85
        assert kws == ["火影", "鸣人"]

    def test_fuzzy_fallback(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("这张图是动漫风格的插画")
        assert cat == "动漫"
        assert conf > 0

    def test_unknown_category_becomes_unclassified(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("未知内容||0.5||a,b")
        assert cat == "未整理"

    def test_confidence_clamped(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("科技||1.7||a")
        assert conf == 1.0


class TestJsonParsing:
    def test_json_standard(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        raw = '{"category": "动漫", "confidence": 0.91, "keywords": ["火影", "鸣人"]}'
        cat, conf, kws = clf._parse_response(raw)
        assert cat == "动漫"
        assert conf == 0.91
        assert kws == ["火影", "鸣人"]

    def test_json_wrapped_in_text(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        raw = '这是一张图片。\n```json\n{"category": "科技", "confidence": 0.88, "keywords": ["芯片"]}\n```\n以上。'
        cat, conf, kws = clf._parse_response(raw)
        assert cat == "科技"
        assert conf == 0.88

    def test_json_invalid_falls_back_to_pipe(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response("动漫||0.85||火影,鸣人")
        assert cat == "动漫"
        assert conf == 0.85

    def test_json_missing_fields(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        cat, conf, kws = clf._parse_response('{"confidence": 0.5}')
        assert cat == "未整理"


class TestLowConfidenceSplit:
    def test_below_threshold_routes_to_pending(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out",
                        low_conf_threshold=0.6)
        assert clf._resolve_category("动漫", 0.59) == "待确认"

    def test_at_or_above_threshold_keeps_category(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out",
                        low_conf_threshold=0.6)
        assert clf._resolve_category("动漫", 0.6) == "动漫"
        assert clf._resolve_category("动漫", 0.95) == "动漫"

    def test_default_threshold_is_0_6(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        assert clf._low_conf_threshold == 0.6


class TestFilterDone:
    def test_same_name_size_mtime_skipped(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        img = src / "a.jpg"
        img.write_bytes(b"imgdata")
        (out / "科技").mkdir(parents=True)
        done = out / "科技" / "a.jpg"
        shutil.copy2(img, done)  # copy2 保留 mtime

        clf = _make_clf(tmp_path, src, out)
        pending = clf._filter_done([img])
        assert pending == []

    def test_same_name_different_content_kept(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        img = src / "a.jpg"
        img.write_bytes(b"imgdata1")  # 8 字节
        (out / "科技").mkdir(parents=True)
        old = out / "科技" / "a.jpg"
        old.write_bytes(b"imgdata222")  # 10 字节，与源文件大小不同

        clf = _make_clf(tmp_path, src, out)
        pending = clf._filter_done([img])
        assert pending == [img]

    def test_non_image_ext_ignored_in_scan(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        (src / "note.txt").write_text("x")
        clf = _make_clf(tmp_path, src, out)
        images = [f for f in src.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        assert images == []


class TestSummaryCancelledFlag:
    def test_init_defaults_not_cancelled(self, tmp_path):
        clf = _make_clf(tmp_path, tmp_path / "src", tmp_path / "out")
        assert clf._cancelled is False


class TestRecursiveScan:
    def test_scan_images_recursive(self, tmp_path):
        src = tmp_path / "src"
        (src / "sub" / "deep").mkdir(parents=True)
        (src / "a.jpg").write_bytes(b"1")
        (src / "sub" / "b.png").write_bytes(b"2")
        (src / "sub" / "deep" / "c.jpeg").write_bytes(b"3")
        (src / "sub" / "note.txt").write_bytes(b"x")
        (src / "video.vesf").write_bytes(b"y")

        files = scan_images(src)
        assert [f.name for f in files] == ["a.jpg", "b.png", "c.jpeg"]

    def test_scan_empty_dir(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()
        assert scan_images(src) == []


class TestUniqueTarget:
    def test_no_collision_returns_same_path(self, tmp_path):
        p = tmp_path / "a.jpg"
        assert unique_target(p) == p

    def test_collision_appends_index(self, tmp_path):
        p = tmp_path / "a.jpg"
        p.write_bytes(b"x")
        assert unique_target(p).name == "a_1.jpg"
        (tmp_path / "a_1.jpg").write_bytes(b"y")
        assert unique_target(p).name == "a_2.jpg"


class TestBaseName:
    def test_strips_serial_suffix(self):
        assert base_name("a_1.jpg") == "a.jpg"
        assert base_name("a_12.png") == "a.png"

    def test_plain_name_unchanged(self):
        assert base_name("a.jpg") == "a.jpg"

    def test_non_serial_underscore_kept(self):
        assert base_name("photo_2024.jpg") == "photo_2024.jpg"
        assert base_name("my_photo.jpg") == "my_photo.jpg"


class TestFilterDoneWithSerial:
    def test_serial_suffixed_file_counts_as_done(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        img = src / "a.jpg"
        img.write_bytes(b"imgdata")
        sub = out / "科技"
        sub.mkdir(parents=True)
        shutil.copy2(img, sub / "a_1.jpg")  # 序号版本，内容/时间一致

        clf = _make_clf(tmp_path, src, out)
        assert clf._filter_done([img]) == []

    def test_output_subdirectory_files_collected(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        img = src / "a.jpg"
        img.write_bytes(b"imgdata")
        nested = out / "科技" / "旧批次"
        nested.mkdir(parents=True)
        shutil.copy2(img, nested / "a.jpg")

        clf = _make_clf(tmp_path, src, out)
        assert clf._filter_done([img]) == []
