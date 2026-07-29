"""Render CriGent's mark to crigent.ico (multi-size) for the desktop shortcut.

The app itself paints its icon at runtime; this file exists only because Windows
shortcuts need an .ico on disk. Re-run after changing paint_logo().
"""
import io
import sys
from pathlib import Path

from PyQt6.QtCore import QBuffer, QByteArray
from PyQt6.QtWidgets import QApplication
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crigent import paint_logo                                  # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)


def pixmap_to_pil(size: int) -> Image.Image:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.OpenModeFlag.WriteOnly)
    paint_logo(size).save(buf, "PNG")
    buf.close()
    return Image.open(io.BytesIO(bytes(ba))).convert("RGBA")


def main() -> None:
    app = QApplication(sys.argv)                                   # noqa: F841
    out = Path(__file__).resolve().parent / "crigent.ico"
    frames = [pixmap_to_pil(s) for s in SIZES]
    frames[-1].save(out, format="ICO", sizes=[(f.width, f.height) for f in frames])
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
