"""PyInstaller 构建脚本"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent


def build():
    specs = sorted(ROOT.glob("*.spec"))
    if not specs:
        print("Build FAILED: 未找到 .spec 文件")
        sys.exit(1)
    spec = specs[0]
    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "--clean", "--noconfirm", str(spec),
    ], check=True)

    exe = ROOT / "dist" / "Kimi图片分类工具.exe"
    if exe.exists():
        size = exe.stat().st_size / (1024 * 1024)
        print(f"\nBuild OK: {exe} ({size:.1f} MB)")
    else:
        print("Build FAILED: EXE not found")
        sys.exit(1)


if __name__ == "__main__":
    build()
