"""image_prep 图像预处理测试"""
import base64
import io
import random
from PIL import Image

from app.image_prep import MAX_SIDE, JPEG_QUALITY, prepare_image


def _save_jpg(img: Image.Image, path, exif=None) -> None:
    if exif is None:
        img.save(path, "JPEG", quality=95)
    else:
        img.save(path, "JPEG", quality=95, exif=exif)


class TestPrepareImage:
    def test_exif_orientation_corrected(self, tmp_path):
        # Orientation=6（旋转90°）：100x200 → 校正后 200x100
        img = Image.new("RGB", (100, 200), "red")
        exif = Image.Exif()
        exif[274] = 6
        p = tmp_path / "rot.jpg"
        _save_jpg(img, p, exif=exif)

        ext, b64 = prepare_image(p)
        out = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert (out.width, out.height) == (200, 100)
        assert ext == "jpeg"

    def test_large_image_downscaled(self, tmp_path):
        img = Image.new("RGB", (2000, 1000), "blue")
        p = tmp_path / "big.jpg"
        _save_jpg(img, p)

        _, b64 = prepare_image(p)
        out = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert max(out.width, out.height) == MAX_SIDE
        assert out.width < 2000

    def test_small_image_not_rescaled(self, tmp_path):
        img = Image.new("RGB", (800, 600), "green")
        p = tmp_path / "small.jpg"
        _save_jpg(img, p)

        _, b64 = prepare_image(p)
        out = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert (out.width, out.height) == (800, 600)

    def test_rgba_transparent_to_white_bg(self, tmp_path):
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))  # 全透明
        p = tmp_path / "alpha.png"
        img.save(p)

        _, b64 = prepare_image(p)
        out = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert out.mode == "RGB"
        assert out.getpixel((50, 50)) == (255, 255, 255)

    def test_output_valid_jpeg_quality(self, tmp_path):
        img = Image.new("RGB", (300, 300), "yellow")
        p = tmp_path / "q.jpg"
        _save_jpg(img, p)

        ext, b64 = prepare_image(p)
        assert ext == "jpeg"
        raw = base64.b64decode(b64)
        assert raw.startswith(b"\xff\xd8")  # JPEG SOI 标记
        assert len(raw) > 0

    def test_output_uses_declared_jpeg_quality(self, tmp_path):
        """回归：输出必须按 JPEG_QUALITY 压缩，而不是悄悄用了更高质量（更贵）"""
        random.seed(0)
        img = Image.new("RGB", (200, 200))
        img.putdata([(random.randrange(256), random.randrange(256), random.randrange(256))
                     for _ in range(200 * 200)])
        p = tmp_path / "noise.png"
        img.save(p)                      # 无需缩图/转色，可与其他质量编码直接比对

        _, b64 = prepare_image(p)
        got = len(base64.b64decode(b64))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        expected = buf.tell()
        assert abs(got - expected) / expected < 0.02

        higher = io.BytesIO()
        img.save(higher, format="JPEG", quality=95)
        assert got < higher.tell()        # 高频图上 q85 应明显小于 q95
