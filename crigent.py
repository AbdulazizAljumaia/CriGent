"""CriGent — a local AI agent: chat over local Ollama models, with tool use, web
search, reusable skills, a live GPU dashboard and a built-in model manager."""

import html as _html
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from ddgs import DDGS
except ImportError:                                                # noqa: BLE001
    DDGS = None

try:
    from bs4 import BeautifulSoup
except ImportError:                                                # noqa: BLE001
    BeautifulSoup = None
from PyQt6.QtCore import (QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRect,
                          QRectF, QSize, Qt, QThread, QTimer, QUrl, pyqtProperty,
                          pyqtSignal)
from PyQt6.QtGui import (QColor, QDesktopServices, QFont, QFontDatabase,
                         QFontMetrics, QIcon, QLinearGradient, QPainter,
                         QPainterPath, QPen, QPixmap)
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup,
                             QComboBox, QDialog, QFileDialog, QFrame, QGridLayout,
                             QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMainWindow, QMenu,
                             QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                             QScrollArea,
                             QSizePolicy, QStackedWidget, QTextEdit, QVBoxLayout,
                             QWidget)

APP_NAME = "CriGent"
APP_TAGLINE = "A local AI agent — your models, your machine."
DEV_NAME = "Abdulaziz Al Jumaia"
DEV_SITE = "https://crimsonlingua.com"
DEV_LINKEDIN = "https://sa.linkedin.com/in/abdulaziz-al-jumaia"

# Frozen into a one-file exe, __file__ points inside a temp extraction dir that is
# wiped on exit — so anything the user creates has to live somewhere writable and
# persistent instead.
FROZEN = getattr(sys, "frozen", False)


def _user_base() -> Path:
    """Per-user fallback. Never a fixed drive or a fixed user name."""
    return Path(os.environ.get("LOCALAPPDATA") or Path.home())


# Records where the user wants their data. Kept per-user rather than beside the
# program so it survives moving the exe, and so a read-only location still works.
LOCATION_FILE = _user_base() / APP_NAME / "location.json"


def program_dir() -> Path:
    """Folder the program itself lives in — exe when frozen, script otherwise."""
    return Path(sys.executable if FROZEN else __file__).resolve().parent


def is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".crigent_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:                                              # noqa: BLE001
        return False


def free_gb(path: Path) -> float:
    """Free space on whatever volume holds path (walking up if it is new)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1e9
    except OSError:
        return 0.0


def _app_dir() -> Path:
    """Resolution order, all relative — nothing is pinned to a drive or user:
      1. a folder the user picked (location.json)
      2. beside the program, so a USB stick or D:\\ install keeps everything
         together and models do not land on the system drive
      3. the per-user app-data folder, for when the program sits somewhere
         read-only like Program Files
    """
    try:
        if LOCATION_FILE.exists():
            chosen = json.loads(LOCATION_FILE.read_text(encoding="utf-8")).get("data_dir")
            if chosen and is_writable(Path(chosen)):
                return Path(chosen)
    except Exception:                                              # noqa: BLE001
        pass

    if FROZEN:
        beside = program_dir() / f"{APP_NAME}-data"
        if is_writable(beside):
            return beside
        fallback = _user_base() / APP_NAME
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    directory = Path(__file__).resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory


ROOT = _app_dir()


def set_data_dir(path: Path) -> None:
    """Move CriGent's storage somewhere else and rebind every derived path."""
    global ROOT, CHATS_DIR, SKILLS_PATH, PROMPTS_PATH, SETTINGS_PATH, MODELS_DIR
    path.mkdir(parents=True, exist_ok=True)
    LOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCATION_FILE.write_text(json.dumps({"data_dir": str(path)}, indent=2),
                             encoding="utf-8")
    ROOT = path
    CHATS_DIR = ROOT / "chats"
    SKILLS_PATH = ROOT / "skills.json"
    PROMPTS_PATH = ROOT / "prompts.json"
    SETTINGS_PATH = ROOT / "settings.json"
    MODELS_DIR = ROOT / "models"
OLLAMA_DL_URL = ("https://github.com/ollama/ollama/releases/download/"
                 "v0.32.5/ollama-windows-amd64.zip")
OLLAMA_DL_MB = 1458


def models_dir() -> Path:
    """Where Ollama keeps its blobs. Configurable so a fresh install can adopt an
    existing store instead of re-importing many gigabytes of models."""
    try:
        if SETTINGS_PATH.exists():
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8")).get("models_dir")
            if saved and Path(saved).is_dir():
                return Path(saved)
    except Exception:                                              # noqa: BLE001
        pass
    return ROOT / "models"


def _saved_ollama() -> Path | None:
    """A path the user pointed us at in the setup wizard."""
    try:
        if SETTINGS_PATH.exists():
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8")).get("ollama_path")
            if saved and Path(saved).is_file():
                return Path(saved)
    except Exception:                                              # noqa: BLE001
        pass
    return None


def ollama_exe() -> Path | None:
    """Find an Ollama we can drive: one the user picked, then ours, then whatever
    is installed in the usual places."""
    chosen = _saved_ollama()
    if chosen:
        return chosen
    candidates = [ROOT / "ollama" / "ollama.exe"]
    found = shutil.which("ollama")
    if found:
        candidates.append(Path(found))
    for env in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMW6432"):
        base = os.environ.get(env)
        if base:
            candidates.append(Path(base) / "Programs" / "Ollama" / "ollama.exe")
            candidates.append(Path(base) / "Ollama" / "ollama.exe")
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None
HOST = "127.0.0.1:11434"
CHAT_URL = f"http://{HOST}/api/chat"
TAGS_URL = f"http://{HOST}/api/tags"

def model_label(name: str) -> tuple[str, str]:
    """Readable name and chat badge for whatever Ollama reports. Derived from the
    model's own name so the app never assumes a particular set of models."""
    pretty = re.sub(r"[-_.]+", " ", name).strip()
    pretty = " ".join(w if w.isupper() else w.capitalize() for w in pretty.split())
    return (pretty or name), (pretty or name).upper()

# Tool-use protocol: the model asks to run a command with a ```run fenced block;
# the app shows it to the user and only executes on explicit click.
TOOL_RE = re.compile(r"```run\s*\n(.*?)```", re.S | re.I)
MAX_TOOL_ROUNDS = 6
TOOL_SYSTEM_PROMPT = (
    "You can ask to run a PowerShell command on the user's Windows machine. "
    "To do so, reply with a fenced code block using the language tag `run`, containing exactly "
    "one command, e.g.:\n\n```run\nGet-ChildItem C:\\Users\n```\n\n"
    "The user will review and explicitly approve or deny it before anything executes. You will "
    "then be told the command's output (or that it was denied) and can continue. Only use a "
    "`run` block when you actually want that exact command executed right now — use normal "
    "language-tagged blocks (like ```powershell) for examples you are only showing, not asking "
    "to run. Put just one command per `run` block; issue another turn for the next one."
)

# Skill protocol: the model proposes a reusable skill with a ```skill fenced block;
# the app shows it to the user and only saves on explicit click, same as tool-run.
CHATS_DIR = ROOT / "chats"
SKILLS_PATH = ROOT / "skills.json"
SKILL_RE = re.compile(r"```skill\s*\n(.*?)```", re.S | re.I)
SKILL_SYSTEM_PROMPT = (
    "If the user explicitly asks you to save, remember, or create a reusable skill (a named "
    "set of instructions they can reuse later), propose it with a fenced block using the "
    "language tag `skill`, formatted exactly as:\n\n"
    "```skill\nname: <short skill name>\n---\n<the actual instructions>\n```\n\n"
    "The user will review it and click Save before it's added to their skill list. Only propose "
    "a skill when the user is clearly asking for one to be saved — not for ordinary code "
    "examples or one-off answers."
)


def parse_skill_block(raw: str):
    """Parse a ```skill block body into (name, content), or None if malformed."""
    lines = raw.strip("\n").split("\n")
    if not lines or not lines[0].lower().startswith("name:"):
        return None
    name = lines[0].split(":", 1)[1].strip()
    rest = lines[1:]
    if rest and rest[0].strip() == "---":
        rest = rest[1:]
    content = "\n".join(rest).strip()
    if not name or not content:
        return None
    return name, content


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# Web protocol: the model searches with a ```search block or reads a page with a ```fetch
# block. Both are read-only, so — unlike `run` — they execute automatically once the Web
# search checkbox is on; every search/fetch still renders as a visible card in the chat.
SEARCH_RE = re.compile(r"```search\s*\n(.*?)```", re.S | re.I)
FETCH_RE = re.compile(r"```fetch\s*\n(.*?)```", re.S | re.I)
WEB_SYSTEM_PROMPT = (
    "You can search the web or read a specific page to get information beyond your training "
    "data. To search, reply with a fenced block tagged `search` containing just the query:\n\n"
    "```search\nyour query here\n```\n\n"
    "To read a specific URL, reply with a fenced block tagged `fetch` containing just that URL:\n\n"
    "```fetch\nhttps://example.com/page\n```\n\n"
    "Results come back to you automatically and are shown to the user, so no permission step is "
    "needed — just use one block per turn and wait for the result before continuing. Prefer "
    "`search` first to find sources, then `fetch` a promising result for detail."
)

PROMPTS_PATH = ROOT / "prompts.json"
SETTINGS_PATH = ROOT / "settings.json"
MODELS_DIR = ROOT / "models"

# Where the model runs. Ollama takes num_gpu as the count of layers to offload,
# so 0 pins it to CPU and a high number pushes everything it can onto the GPU.
# "auto" sends nothing and lets Ollama decide.
COMPUTE_MODES = [
    ("auto", "Auto", None,
     "Let Ollama decide how much to offload — the usual choice."),
    ("gpu", "GPU (CUDA)", 999,
     "Force as many layers as possible onto the NVIDIA GPU."),
    ("cpu", "CPU", 0,
     "Run entirely on the CPU. Much slower, but works with no usable GPU "
     "and frees VRAM for other apps."),
]
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[=>]")

# The enrichment prompts the app prepends to a request. Editable in the Prompts
# page and persisted to prompts.json; these stay the reset-to-default source.
DEFAULT_PROMPTS = {
    "tools": TOOL_SYSTEM_PROMPT,
    "web": WEB_SYSTEM_PROMPT,
    "skills": SKILL_SYSTEM_PROMPT,
}
PROMPT_META = [
    ("tools", "Tools", "Sent when the Tools toggle is on. Teaches the model to request a "
                       "command with a ```run block."),
    ("web", "Web", "Sent when the Web toggle is on. Teaches the model to search with a "
                   "```search block or read a page with ```fetch."),
    ("skills", "Skills", "Sent when the Skills toggle is on. Teaches the model to propose a "
                         "reusable skill with a ```skill block."),
]


def _is_safe_web_url(url: str) -> bool:
    """Reject non-http(s) schemes and obvious local/private targets before fetching."""
    try:
        parsed = urlparse(url)
    except Exception:                                              # noqa: BLE001
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host or host == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    except ValueError:
        pass                                                       # a hostname, not a literal IP
    return True

# nvidia-smi fields that actually exist. `driver_model` does not, and `power.limit`
# reports [N/A] on this laptop GPU while `enforced.power.limit` works.
GPU_FIELDS = [
    "name", "temperature.gpu", "power.draw", "enforced.power.limit",
    "memory.used", "memory.total", "utilization.gpu", "utilization.memory",
    "clocks.sm", "clocks.max.sm",
]

# One cool-neutral hue family for every surface, a single primary accent, and
# desaturated semantics. Values step evenly so depth reads as elevation, not as
# five competing colours.
C = {
    "bg": "#0e1116",          # window background
    "panel": "#151922",       # sidebars, header, composer
    "panel_hi": "#1c212c",    # cards, inputs, raised surfaces
    "overlay": "#232936",     # hover / pressed states
    "line": "#232936",        # hairline dividers
    "line_str": "#333b4b",    # emphasised borders (focus, active)

    "text": "#e9edf4",        # primary copy
    "dim": "#98a3b5",         # secondary copy
    "faint": "#69748a",       # tertiary, labels, captions

    "accent": "#5b8def",      # primary action
    "accent_hi": "#7ba6f7",   # hover
    "accent_soft": "#1e2942", # accent-tinted surface (user bubble)
    "green": "#4ec9a0",
    "green_soft": "#16302a",
    "amber": "#dda657",
    "amber_soft": "#302819",
    "red": "#e4646d",
    "red_soft": "#331b20",
    "violet": "#9b8cf0",
    "violet_soft": "#242040",
    "code": "#c6d3e6",        # code foreground
}

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


# --------------------------------------------------------------------------- #
#  Brand mark
# --------------------------------------------------------------------------- #

BRAND_A = "#E23D57"        # crimson, after crimsonlingua
BRAND_B = "#A8102F"


def paint_logo(size: int, tile: bool = True) -> QPixmap:
    """CriGent's mark: an open crimson ring cut at the lower right, with a solid
    node sitting in the gap — a 'C' for Crimson and an agent core. Drawn in code
    so it stays crisp at any size instead of shipping bitmaps."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    if tile:
        grad = QLinearGradient(0, 0, size, size)
        grad.setColorAt(0, QColor(BRAND_A))
        grad.setColorAt(1, QColor(BRAND_B))
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, size, size), size * 0.26, size * 0.26)
        p.fillPath(path, grad)
        ink = QColor("#FFFFFF")
        inset = size * 0.28
    else:
        ink = QColor(BRAND_A)
        inset = size * 0.16

    ring = QRectF(inset, inset, size - inset * 2, size - inset * 2)
    radius = ring.width() / 2.0
    stroke = max(2.0, size * 0.105)

    pen = QPen(ink, stroke)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    # A 270° sweep starting at 45° leaves a clean quarter gap on the right —
    # the "C". Qt angles are 1/16°, zero at 3 o'clock, positive anticlockwise.
    p.drawArc(ring, int(45 * 16), int(270 * 16))

    # The node sits dead centre of that gap, one full quarter-turn clear of both
    # arc ends, so it reads as a separate core rather than fusing with the stroke.
    node_r = stroke * 0.72
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(ink)
    p.drawEllipse(QPointF(ring.center().x() + radius, ring.center().y()), node_r, node_r)
    p.end()
    return pm


def app_icon() -> QIcon:
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(paint_logo(s))
    return icon


def install_crash_guard() -> Path:
    """Stop one bad slot from killing the whole app.

    PyQt aborts the process when an exception escapes a slot — no dialog, no
    traceback, the window just vanishes. Replacing sys.excepthook keeps the app
    alive and writes the traceback somewhere we can actually read it.
    """
    log_path = ROOT / "crash.log"

    def handler(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {stamp} =====\n{text}")
        except OSError:
            pass
        app = QApplication.instance()
        if app is not None:
            for widget in app.topLevelWidgets():
                report = getattr(widget, "report_error", None)
                if callable(report):
                    try:
                        report(exc, text)
                    except Exception:                              # noqa: BLE001
                        pass
                    break

    sys.excepthook = handler
    return log_path


def still_running(worker) -> bool:
    """isRunning() on a QThread whose C++ side is gone raises RuntimeError, and
    an exception in a slot is fatal in PyQt. Treat a dead wrapper as finished."""
    if worker is None:
        return False
    try:
        return worker.isRunning()
    except RuntimeError:
        return False


def mono_family() -> str:
    for name in ("Cascadia Mono", "Consolas", "Courier New"):
        if name in QFontDatabase.families():
            return name
    return "monospace"


# Code surfaces must be sized in PIXELS, matching the px sizes declared in the
# stylesheet. A point-sized QFont here would disagree with the QSS `font-size`
# that actually paints the text, and every fixed height would clip its content.
CODE_PX = 12
CODE_PX_SM = 11
UI_FONT = "Segoe UI"
BODY_PX = 14
COLUMN_MAX = 880          # centred reading column
BUBBLE_MAX = 720


def mono_font(family: str, px: int) -> QFont:
    f = QFont(family)
    f.setPixelSize(px)
    return f


def text_px_width(text: str) -> int:
    """Width of the widest line at body size — used to make user bubbles hug
    their content instead of collapsing to QLabel's word-wrap sizeHint."""
    f = QFont(UI_FONT)
    f.setPixelSize(BODY_PX)
    fm = QFontMetrics(f)
    lines = text.splitlines() or [""]
    return max(fm.horizontalAdvance(ln) for ln in lines)


def fit_height(edit, font: QFont, px_pad: int = 30, cap: int = 400) -> int:
    """Height that shows every line of a read-only text box without scrolling.

    Measure with the *explicit* font rather than edit.fontMetrics(): at
    construction the widget hasn't been polished by the stylesheet yet, so its
    own metrics are the app default and every height comes out short.
    px_pad covers the stylesheet's vertical padding (16), the document margin
    Qt adds on both sides (8), and a little slack.
    """
    lines = max(1, edit.document().blockCount())
    return min(int(QFontMetrics(font).lineSpacing() * lines) + px_pad, cap)


# --------------------------------------------------------------------------- #
#  Text rendering
# --------------------------------------------------------------------------- #

CODE_FENCE_RE = re.compile(r"```(\w*)\n?(.*?)(?:```|$)", re.S)


def split_blocks(text: str):
    """Split markdown into an alternating [prose, (lang, code), prose, ...] list.

    Prose is further split on blank lines. That keeps each block small, so a
    streaming reply only has to re-render its final paragraph rather than one
    ever-growing wall of text.
    """
    parts = CODE_FENCE_RE.split(text)
    blocks = []
    i = 0
    while i < len(parts):
        prose = parts[i]
        if prose.strip():
            for para in re.split(r"\n\s*\n", prose):
                if para.strip():
                    blocks.append(("prose", para))
        if i + 2 < len(parts):
            lang, code = parts[i + 1], parts[i + 2]
            if code.strip():
                blocks.append(("code", lang.strip(), code))
        i += 3
    return blocks


def _inline(text: str) -> str:
    s = _html.escape(text)
    s = re.sub(r"`([^`]+)`",
               f'<code style="background:{C["overlay"]};color:{C["code"]};'
               r'border-radius:4px;padding:1px 5px;">\1</code>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?m)^\s*[-*]\s+", "&nbsp;&nbsp;• ", s)
    s = re.sub(r"(?m)^(#{1,4})\s*(.+)$", r"<b>\2</b>", s)
    return s.replace("\n", "<br>")


def split_think(raw: str):
    """Return (reasoning, answer) — Qwen emits <think> blocks we keep collapsed."""
    think = "\n".join(m.strip() for m in re.findall(r"<think>(.*?)</think>", raw, flags=re.S))
    body = re.sub(r"(?s)<think>.*?</think>\s*", "", raw)
    if "<think>" in body and "</think>" not in body:      # still streaming a block
        think = (think + "\n" + body.split("<think>", 1)[1]).strip()
        body = body.split("<think>", 1)[0]
    return think.strip(), body.strip()


# --------------------------------------------------------------------------- #
#  Painted widgets
# --------------------------------------------------------------------------- #

class RingGauge(QWidget):
    """Circular gauge with an animated sweep."""

    def __init__(self, label: str, color: str, unit: str = "%"):
        super().__init__()
        self._value = 0.0
        self._shown = 0.0
        self.label, self.unit = label, unit
        self.color = QColor(color)
        self.caption = ""
        self.setMinimumSize(140, 168)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._anim = QPropertyAnimation(self, b"shown", self)
        self._anim.setDuration(450)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def get_shown(self) -> float:
        return self._shown

    def set_shown(self, v: float):
        self._shown = v
        self.update()

    shown = pyqtProperty(float, fget=get_shown, fset=set_shown)

    def set_value(self, pct: float, caption: str = ""):
        self.caption = caption
        self._value = max(0.0, min(100.0, pct))
        self._anim.stop()
        self._anim.setStartValue(self._shown)
        self._anim.setEndValue(self._value)
        self._anim.start()

    LABEL_BAND = 42        # reserved under the ring so captions never touch the arc

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        area = self.rect().adjusted(8, 8, -8, -self.LABEL_BAND)
        d = max(48, min(area.width(), area.height()))
        ring = QRect(0, 0, d, d)
        ring.moveCenter(QPoint(self.rect().center().x(), area.center().y()))
        stroke = max(7, int(d * 0.09))

        pen = QPen(QColor(C["panel_hi"]), stroke)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(ring, 0, 360 * 16)

        if self._shown > 0.4:
            pen.setColor(self.color)
            p.setPen(pen)
            p.drawArc(ring, 90 * 16, int(-self._shown / 100 * 360 * 16))

        f = self.font()
        f.setPixelSize(max(16, int(d * 0.23)))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(C["text"]))
        p.drawText(ring, Qt.AlignmentFlag.AlignCenter, f"{self._shown:.0f}{self.unit}")

        f.setPixelSize(11)
        f.setBold(False)
        p.setFont(f)
        full = self.rect().width()
        p.setPen(QColor(C["dim"]))
        p.drawText(QRect(self.rect().left(), ring.bottom() + 9, full, 15),
                   Qt.AlignmentFlag.AlignCenter, self.label)
        if self.caption:
            p.setPen(QColor(C["faint"]))
            p.drawText(QRect(self.rect().left(), ring.bottom() + 25, full, 15),
                       Qt.AlignmentFlag.AlignCenter, self.caption)
        p.end()


class Sparkline(QWidget):
    """Filled history graph, newest sample on the right."""

    def __init__(self, color: str, capacity: int = 120):
        super().__init__()
        self.color = QColor(color)
        self.data = deque(maxlen=capacity)
        self.setMinimumHeight(64)

    def push(self, v: float):
        self.data.append(max(0.0, min(100.0, v)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        grid = QPen(QColor(C["line"]), 1, Qt.PenStyle.DotLine)
        p.setPen(grid)
        for frac in (0.25, 0.5, 0.75):
            p.drawLine(0, int(h * frac), w, int(h * frac))

        if len(self.data) < 2:
            p.end()
            return

        n = self.data.maxlen
        step = w / (n - 1)
        pts = [QPointF(w - (len(self.data) - 1 - i) * step, h - (v / 100.0) * (h - 6) - 3)
               for i, v in enumerate(self.data)]

        fill = QPainterPath(QPointF(pts[0].x(), h))
        for pt in pts:
            fill.lineTo(pt)
        fill.lineTo(pts[-1].x(), h)
        fill.closeSubpath()
        grad = QLinearGradient(0, 0, 0, h)
        c0 = QColor(self.color)
        c0.setAlpha(110)
        c1 = QColor(self.color)
        c1.setAlpha(0)
        grad.setColorAt(0, c0)
        grad.setColorAt(1, c1)
        p.fillPath(fill, grad)

        line = QPainterPath(pts[0])
        for pt in pts[1:]:
            line.lineTo(pt)
        p.setPen(QPen(self.color, 2))
        p.drawPath(line)
        p.end()


class Card(QFrame):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("card")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(16, 14, 16, 16)
        self.box.setSpacing(10)
        if title:
            lab = QLabel(title.upper())
            lab.setObjectName("cardTitle")
            self.box.addWidget(lab)


class CodeBlock(QFrame):
    """A fenced code block rendered as its own container with a copy button."""

    MAX_HEIGHT = 420

    def __init__(self, code: str, lang: str, mono: str):
        super().__init__()
        self.setObjectName("codeBlock")
        self._code = code.rstrip("\n")

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = QWidget()
        header.setObjectName("codeHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 6, 8, 6)
        tag = QLabel((lang or "text").lower())
        tag.setObjectName("codeLang")
        hl.addWidget(tag)
        hl.addStretch()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setObjectName("copyBtn")
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.clicked.connect(self._copy)
        hl.addWidget(self.copy_btn)
        v.addWidget(header)

        self.editor = QPlainTextEdit(self._code)
        self.editor.setObjectName("codeBody")
        self.editor.setReadOnly(True)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._font = mono_font(mono, CODE_PX)
        self.editor.setFont(self._font)
        self.editor.setFrameShape(QFrame.Shape.NoFrame)
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        v.addWidget(self.editor)
        self._autosize()

    def _autosize(self):
        self.editor.setFixedHeight(fit_height(self.editor, self._font, cap=self.MAX_HEIGHT))

    def set_code(self, code: str):
        """Update in place while a reply streams, instead of rebuilding the widget."""
        code = code.rstrip("\n")
        if code == self._code:
            return
        self._code = code
        self.editor.setPlainText(code)
        self._autosize()

    def _copy(self):
        QApplication.clipboard().setText(self._code)
        self.copy_btn.setText("Copied")
        # Parent the timer to the button: clearing or switching chats inside this
        # window destroys the widget, and a free-floating singleShot would then
        # fire on a freed C++ object and take the whole process down.
        reset = QTimer(self.copy_btn)
        reset.setSingleShot(True)
        reset.timeout.connect(lambda: self.copy_btn.setText("Copy"))
        reset.start(1100)


# --------------------------------------------------------------------------- #
#  Workers
# --------------------------------------------------------------------------- #

class GpuWorker(QThread):
    sample = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._run = True
        self._active = False

    def stop(self):
        self._run = False

    def set_active(self, active: bool):
        """Only poll while the GPU page is visible - two nvidia-smi processes a
        second is real overhead to pay for a tab nobody is looking at."""
        self._active = active

    def run(self):
        while self._run:
            if self._active:
                self.sample.emit(self._read())
            for _ in range(10):                     # 1s total, but exits promptly
                if not self._run:
                    return
                self.msleep(100)

    def _read(self) -> dict:
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, errors="replace",
                timeout=5, creationflags=NO_WINDOW)
            if r.returncode != 0:
                return {"error": (r.stderr or r.stdout).strip() or "nvidia-smi failed"}
            row = [c.strip() for c in r.stdout.strip().splitlines()[0].split(",")]
            d = dict(zip(GPU_FIELDS, row))

            def num(key, default=0.0):
                try:
                    return float(d.get(key, ""))
                except ValueError:
                    return default

            out = {
                "name": d.get("name", "GPU"),
                "temp": num("temperature.gpu"),
                "power": num("power.draw"),
                "power_max": num("enforced.power.limit", 0.0),
                "mem_used": num("memory.used"),
                "mem_total": num("memory.total", 1.0),
                "util": num("utilization.gpu"),
                "mem_util": num("utilization.memory"),
                "clock": num("clocks.sm"),
                "clock_max": num("clocks.max.sm"),
                "procs": [],
            }
            pr = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, errors="replace",
                timeout=5, creationflags=NO_WINDOW)
            if pr.returncode == 0:
                for line in pr.stdout.strip().splitlines():
                    parts = [c.strip() for c in line.split(",")]
                    if len(parts) >= 3:
                        out["procs"].append((parts[0], Path(parts[1]).name, parts[2]))
            return out
        except FileNotFoundError:
            return {"error": "nvidia-smi not found — is the NVIDIA driver installed?"}
        except Exception as exc:                                  # noqa: BLE001
            return {"error": str(exc)}


class ChatWorker(QThread):
    chunk = pyqtSignal(str)
    done = pyqtSignal(float, int)
    failed = pyqtSignal(str)

    def __init__(self, messages, model: str, num_gpu=None):
        super().__init__()
        self.messages = messages
        self.model = model
        self.num_gpu = num_gpu
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        t0 = time.time()
        tokens = 0
        try:
            body = {"model": self.model, "messages": self.messages, "stream": True}
            if self.num_gpu is not None:
                body["options"] = {"num_gpu": self.num_gpu}
            with requests.post(
                CHAT_URL, json=body, stream=True, timeout=(10, 7200),
            ) as resp:
                if resp.status_code >= 400:
                    # Ollama explains itself in the body; raise_for_status throws
                    # that away and leaves the user staring at a bare status code.
                    try:
                        detail = resp.json().get("error") or resp.text[:300]
                    except Exception:                             # noqa: BLE001
                        detail = resp.text[:300] or "no detail given"
                    self.failed.emit(f"Ollama rejected the request "
                                     f"({resp.status_code}): {detail}")
                    return
                for line in resp.iter_lines():
                    if self._stop:
                        break
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        self.failed.emit(str(data["error"]))
                        return
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        tokens += 1
                        self.chunk.emit(piece)
                    if data.get("done"):
                        break
            self.done.emit(time.time() - t0, tokens)
        except requests.exceptions.ConnectionError:
            self.failed.emit("Cannot reach Ollama on 127.0.0.1:11434.")
        except Exception as exc:                                  # noqa: BLE001
            self.failed.emit(str(exc))


class CommandWorker(QThread):
    """Runs one user-approved PowerShell command off the UI thread."""

    result = pyqtSignal(str, str, int)     # stdout, stderr, returncode

    def __init__(self, command: str):
        super().__init__()
        self.command = command

    def run(self):
        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                 self.command],
                capture_output=True, text=True, errors="replace",
                timeout=120, creationflags=NO_WINDOW)
            self.result.emit(proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired:
            self.result.emit("", "Command timed out after 120s.", -1)
        except Exception as exc:                                  # noqa: BLE001
            self.result.emit("", str(exc), -1)


class SearchWorker(QThread):
    """Runs a web search (DuckDuckGo, via ddgs) off the UI thread."""

    result = pyqtSignal(list, str)     # results, error

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        if DDGS is None:
            self.result.emit([], "The `ddgs` package isn't installed (pip install ddgs).")
            return
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(self.query, max_results=6))
            self.result.emit(results, "")
        except Exception as exc:                                  # noqa: BLE001
            self.result.emit([], str(exc))


class FetchWorker(QThread):
    """Downloads one URL and extracts readable text, off the UI thread."""

    result = pyqtSignal(str, str)      # text, error

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        if not _is_safe_web_url(self.url):
            self.result.emit("", "Refused: not a public http(s) URL.")
            return
        if BeautifulSoup is None:
            self.result.emit("", "The `beautifulsoup4` package isn't installed.")
            return
        try:
            headers = {"User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) {APP_NAME}/1.0"}
            resp = requests.get(self.url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
            self.result.emit("\n".join(lines), "")
        except Exception as exc:                                  # noqa: BLE001
            self.result.emit("", str(exc))


# --------------------------------------------------------------------------- #
#  Chat bubble
# --------------------------------------------------------------------------- #

class Bubble(QWidget):
    def __init__(self, role: str, mono: str, badge: str = "ASSISTANT"):
        super().__init__()
        self.role, self.mono = role, mono
        self.raw = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        who = QLabel("You" if role == "user" else badge.title())
        who.setObjectName("who_user" if role == "user" else "who_bot")
        if role == "user":
            who.hide()          # right alignment already says who wrote it

        self.think_btn = QPushButton("▸ reasoning")
        self.think_btn.setObjectName("thinkBtn")
        self.think_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.think_btn.clicked.connect(self._toggle)
        self.think_btn.hide()

        self.think = QLabel()
        self.think.setObjectName("think")
        self.think.setWordWrap(True)
        self.think.hide()

        self.bubble_frame = QFrame()
        self.bubble_frame.setObjectName("bubble_user" if role == "user" else "bubble_bot")
        self.bubble_frame.setMaximumWidth(BUBBLE_MAX)
        self.bubble_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.body_layout = QVBoxLayout(self.bubble_frame)
        self.body_layout.setContentsMargins(16, 12, 16, 12)
        self.body_layout.setSpacing(8)
        self._prose_name = "proseUser" if role == "user" else "proseBot"
        self._rendered = None          # block list currently on screen

        self.meta = QLabel()
        self.meta.setObjectName("meta")
        self.meta.hide()

        align = Qt.AlignmentFlag.AlignRight if role == "user" else Qt.AlignmentFlag.AlignLeft
        outer.addWidget(who, alignment=align)
        outer.addWidget(self.think_btn, alignment=align)
        outer.addWidget(self.think, alignment=align)

        # A stretch on one side beats an alignment flag here: alignment freezes the
        # frame at its (unreliable) word-wrap sizeHint, which collapses the bubble.
        bubble_row = QHBoxLayout()
        bubble_row.setContentsMargins(0, 0, 0, 0)
        bubble_row.setSpacing(0)
        if role == "user":
            bubble_row.addStretch(1)
            bubble_row.addWidget(self.bubble_frame, 0)
        else:
            bubble_row.addWidget(self.bubble_frame, 1)
            bubble_row.addStretch(0)
        outer.addLayout(bubble_row)
        outer.addWidget(self.meta, alignment=align)

    def _toggle(self):
        vis = not self.think.isVisible()
        self.think.setVisible(vis)
        self.think_btn.setText(("▾ " if vis else "▸ ") + "reasoning")

    def _clear_body(self):
        self._rendered = None
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _hug(self, answer: str, blocks) -> None:
        """Size a user bubble to its content rather than the full column."""
        if any(b[0] == "code" for b in blocks):
            self.bubble_frame.setFixedWidth(BUBBLE_MAX)
            return
        width = min(BUBBLE_MAX, max(140, text_px_width(answer) + 40))
        self.bubble_frame.setFixedWidth(width)

    def _prose_label(self, html: str) -> QLabel:
        lbl = QLabel(html)
        lbl.setObjectName(self._prose_name)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return lbl

    def set_text(self, raw: str):
        self.raw = raw
        reasoning, answer = split_think(raw)
        if reasoning:
            self.think_btn.show()
            self.think.setText(_inline(reasoning))

        blocks = split_blocks(answer) if answer else []
        if not blocks:
            if self._rendered != []:
                self._clear_body()
                self.body_layout.addWidget(self._prose_label("…"))
                self._rendered = []
            return
        if self.role == "user":
            self._hug(answer, blocks)
        self._render(blocks)

    def _render(self, blocks: list):
        """Only touch what actually changed.

        Rebuilding the whole bubble on every streamed chunk is quadratic: a long
        reply spent minutes recreating widgets and froze the UI for over a second
        at a time. While streaming, all blocks but the last are already final, so
        the common prefix is reused and the trailing one is updated in place.
        """
        prev = self._rendered
        if prev is None:                       # placeholder was showing
            self._clear_body()
            prev = []

        same = 0
        while same < len(prev) and same < len(blocks) and prev[same] == blocks[same]:
            same += 1

        # The usual streaming case: everything settled except the final block.
        if (same == len(blocks) - 1 == len(prev) - 1
                and self.body_layout.count() == len(prev)
                and prev[same][0] == blocks[same][0]):
            widget = self.body_layout.itemAt(same).widget()
            block = blocks[same]
            if block[0] == "prose" and isinstance(widget, QLabel):
                widget.setText(_inline(block[1]))
                self._rendered = list(blocks)
                return
            if block[0] == "code" and isinstance(widget, CodeBlock):
                widget.set_code(block[2])
                self._rendered = list(blocks)
                return

        # Otherwise drop the tail that changed and rebuild just that part.
        while self.body_layout.count() > same:
            item = self.body_layout.takeAt(self.body_layout.count() - 1)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        for block in blocks[same:]:
            if block[0] == "prose":
                self.body_layout.addWidget(self._prose_label(_inline(block[1])))
            else:
                _, lang, code = block
                self.body_layout.addWidget(CodeBlock(code, lang, self.mono))
        self._rendered = list(blocks)

    def set_error(self, text: str):
        self._clear_body()
        lbl = self._prose_label(f'<span style="color:{C["red"]}">⚠ {_html.escape(text)}</span>')
        self.body_layout.addWidget(lbl)

    def set_meta(self, text: str):
        self.meta.setText(text)
        self.meta.show()


class ToolCard(QFrame):
    """A command the model wants to run, held for explicit user approval."""

    run_clicked = pyqtSignal()
    deny_clicked = pyqtSignal()

    def __init__(self, command: str, mono: str, auto: bool = False):
        super().__init__()
        self.auto = auto
        self.setObjectName("toolCardAuto" if auto else "toolCard")
        self.setMaximumWidth(760)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 14)
        v.setSpacing(10)

        head_row = QHBoxLayout()
        header = QLabel("⚠  Auto-executing — no confirmation" if auto
                        else "Run command?")
        header.setObjectName("toolHeaderAuto" if auto else "toolHeader")
        head_row.addWidget(header)
        head_row.addStretch()
        self.status_lbl = QLabel("running…" if auto else "")
        self.status_lbl.setObjectName("toolStatus")
        head_row.addWidget(self.status_lbl)
        v.addLayout(head_row)

        self.cmd_box = QPlainTextEdit(command)
        self.cmd_box.setObjectName("toolCmd")
        self.cmd_box.setReadOnly(True)
        self.cmd_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._cmd_font = mono_font(mono, CODE_PX)
        self.cmd_box.setFont(self._cmd_font)
        self.cmd_box.setFrameShape(QFrame.Shape.NoFrame)
        self.cmd_box.setFixedHeight(fit_height(self.cmd_box, self._cmd_font, cap=200))
        v.addWidget(self.cmd_box)

        self.output = QPlainTextEdit()
        self.output.setObjectName("toolOutput")
        self.output.setReadOnly(True)
        self._out_font = mono_font(mono, CODE_PX_SM)
        self.output.setFont(self._out_font)
        self.output.setFrameShape(QFrame.Shape.NoFrame)
        self.output.hide()
        v.addWidget(self.output)

        self.run_btn = None
        self.deny_btn = None
        if not auto:
            row = QHBoxLayout()
            row.addStretch()
            self.deny_btn = QPushButton("Deny")
            self.deny_btn.setObjectName("ghost")
            self.deny_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.deny_btn.clicked.connect(self._on_deny)
            self.run_btn = QPushButton("Run")
            self.run_btn.setObjectName("primary")
            self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.run_btn.clicked.connect(self._on_run)
            row.addWidget(self.deny_btn)
            row.addWidget(self.run_btn)
            v.addLayout(row)

    def _on_run(self):
        self.run_btn.setEnabled(False)
        self.deny_btn.setEnabled(False)
        self.run_btn.setText("Running…")
        self.run_clicked.emit()

    def _on_deny(self):
        self.run_btn.setEnabled(False)
        self.deny_btn.setEnabled(False)
        self.deny_btn.setText("Denied")
        self.deny_clicked.emit()

    def show_result(self, stdout: str, stderr: str, code: int):
        if self.run_btn:
            self.run_btn.setText(f"Ran · exit {code}")
        self.status_lbl.setText(f"exit {code}")
        text = (stdout or "") + (("\n" + stderr) if stderr else "")
        text = text.strip() or "(no output)"
        if len(text) > 4000:
            text = text[:4000] + "\n…(truncated)"
        self.output.setPlainText(text)
        self.output.setFixedHeight(fit_height(self.output, self._out_font, cap=240))
        self.output.show()


class ListRow(QWidget):
    """A sidebar/list row: title that elides to the available width, plus an ×
    that asks for confirmation before deleting. Background stays transparent so
    the list item's own hover/selected styling shows through."""

    delete_requested = pyqtSignal(str)

    # Fixed, because sizeHint() is computed before the stylesheet is applied and
    # would be measured against the default font — leaving rows too short and
    # clipping their text once the real (larger) QSS font kicks in.
    H_ONE_LINE = 40
    H_TWO_LINE = 54

    def __init__(self, key: str, title: str, subtitle: str = ""):
        super().__init__()
        self.key = key
        self.full = title
        self.subtitle = subtitle
        self.setFixedHeight(self.H_TWO_LINE if subtitle else self.H_ONE_LINE)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 6, 4)
        row.setSpacing(6)

        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(1)
        # Ignored width policy is what makes elision work: otherwise the label
        # demands its full text width, the row's sizeHint exceeds the viewport,
        # and the view hands back a rect too wide to ever need eliding.
        self.label = QLabel(title)
        self.label.setObjectName("rowTitle")
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        texts.addWidget(self.label)
        self.sub = None
        if subtitle:
            self.sub = QLabel(subtitle)
            self.sub.setObjectName("rowSub")
            self.sub.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            texts.addWidget(self.sub)
        row.addLayout(texts, 1)

        self.close_btn = QPushButton("×")      # U+00D7 renders reliably in Segoe UI
        self.close_btn.setObjectName("rowClose")
        self.close_btn.setFixedSize(22, 22)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setToolTip("Delete")
        self.close_btn.clicked.connect(lambda: self.delete_requested.emit(self.key))
        row.addWidget(self.close_btn, 0, Qt.AlignmentFlag.AlignVCenter)

    def resizeEvent(self, ev):
        # Elide against the label's real width so the title never runs under the ×.
        fm = self.label.fontMetrics()
        self.label.setText(fm.elidedText(self.full, Qt.TextElideMode.ElideRight,
                                         max(20, self.label.width())))
        if self.sub:
            fm2 = self.sub.fontMetrics()
            self.sub.setText(fm2.elidedText(self.subtitle, Qt.TextElideMode.ElideRight,
                                            max(20, self.sub.width())))
        super().resizeEvent(ev)


class ModelImportWorker(QThread):
    """Writes a Modelfile for a chosen .gguf and runs `ollama create` off the UI thread."""

    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, name: str, gguf: Path, ctx: int = 8192):
        super().__init__()
        self.name = name
        self.gguf = gguf
        self.ctx = ctx

    def run(self):
        exe = ollama_exe()
        if exe is None:
            self.failed.emit("Could not find ollama.exe on this system.")
            return
        try:
            modelfile = ROOT / f"Modelfile.{self.name}"
            modelfile.write_text(
                f"FROM {self.gguf}\n\nPARAMETER num_ctx {self.ctx}\n",
                encoding="utf-8")
            self.progress.emit(f"Wrote {modelfile.name}")
            self.progress.emit("Importing into Ollama — this copies the whole file, "
                               "so a large model takes a few minutes…")

            env = dict(os.environ, OLLAMA_MODELS=str(models_dir()), OLLAMA_HOST=HOST)
            proc = subprocess.Popen(
                [str(exe), "create", self.name, "-f", str(modelfile)],
                cwd=str(exe.parent), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                # The CLI writes UTF-8 spinner glyphs; the Windows locale codec
                # (cp1252) chokes on them and kills the import mid-run.
                text=True, encoding="utf-8", errors="replace",
                creationflags=NO_WINDOW)
            last = ""
            last_emit = 0.0
            for raw in proc.stdout:
                # Strip ANSI escapes and the CLI's Braille spinner glyphs — they
                # are noise in a status label, and non-ASCII here has already
                # tripped console encoders downstream.
                line = ANSI_RE.sub("", raw)
                line = "".join(ch for ch in line if 32 <= ord(ch) < 127).strip()
                if not line:
                    continue
                last = line
                # The CLI redraws a progress spinner using \r, which universal
                # newlines turn into thousands of "lines" a second. Emitting each
                # one floods the UI thread's queued-connection backlog until the
                # process dies, so report at most a few times a second.
                now = time.monotonic()
                if now - last_emit < 0.3:
                    continue
                last_emit = now
                self.progress.emit(line[:160])
            code = proc.wait()
            if code == 0:
                self.finished_ok.emit(self.name)
            else:
                self.failed.emit(f"ollama create exited with code {code}. Last output: {last}")
        except Exception as exc:                                  # noqa: BLE001
            self.failed.emit(str(exc))


class SetupWorker(QThread):
    """Downloads and unpacks Ollama into the app folder. Every message it emits
    describes something it actually just did — the wizard animates the delivery,
    it does not invent the content."""

    step = pyqtSignal(str)          # narration line
    log = pyqtSignal(str)           # raw command/output line
    pct = pyqtSignal(int)           # 0-100, -1 for indeterminate
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, dest: Path):
        super().__init__()
        self.dest = dest
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.dest.mkdir(parents=True, exist_ok=True)
            archive = self.dest / "ollama-windows-amd64.zip"

            self.step.emit("Downloading the Ollama runtime")
            self.log.emit(f"GET {OLLAMA_DL_URL}")
            self.pct.emit(0)

            with requests.get(OLLAMA_DL_URL, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                done = 0
                last = 0.0
                with open(archive, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if self._stop:
                            self.failed.emit("Cancelled.")
                            return
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last > 0.4:
                            last = now
                            if total:
                                self.pct.emit(int(done / total * 100))
                                self.log.emit(
                                    f"{done / 1e6:,.0f} MB of {total / 1e6:,.0f} MB")
                            else:
                                self.pct.emit(-1)
                                self.log.emit(f"{done / 1e6:,.0f} MB")
            self.pct.emit(100)
            self.log.emit(f"saved {archive.name} ({archive.stat().st_size / 1e6:,.0f} MB)")

            self.step.emit("Unpacking the runtime")
            self.pct.emit(-1)
            self.log.emit(f"expand {archive.name} -> {self.dest}")
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                for i, name in enumerate(names):
                    if self._stop:
                        self.failed.emit("Cancelled.")
                        return
                    zf.extract(name, self.dest)
                    if i % 12 == 0:
                        self.pct.emit(int(i / max(1, len(names)) * 100))
                        self.log.emit(name)
            self.pct.emit(100)

            exe = self.dest / "ollama.exe"
            if not exe.is_file():
                self.failed.emit("Unpacked, but ollama.exe is not where expected.")
                return
            self.log.emit(f"ready: {exe}")
            try:
                archive.unlink()
                self.log.emit("removed the downloaded archive")
            except OSError:
                pass
            self.step.emit("Runtime installed")
            self.finished_ok.emit()
        except Exception as exc:                                  # noqa: BLE001
            self.failed.emit(str(exc))


class TypeLine(QLabel):
    """Reveals real text one character at a time. Presentation only — it never
    supplies words of its own."""

    def __init__(self, obj_name: str = "wizStep", interval: int = 14):
        super().__init__()
        self.setObjectName(obj_name)
        self.setWordWrap(True)
        self._full = ""
        self._at = 0
        self._timer = QTimer(self)
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self._tick)

    def type(self, text: str):
        self._full = text
        self._at = 0
        self.setText("")
        self._timer.start()

    def _tick(self):
        self._at += 1
        self.setText(self._full[:self._at])
        if self._at >= len(self._full):
            self._timer.stop()


class Pulse(QWidget):
    """Three dots cycling while a step is in flight."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(34, 10)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(260)
        self._timer.timeout.connect(self._advance)

    def start(self):
        self._timer.start()
        self.show()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _advance(self):
        self._phase = (self._phase + 1) % 3
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            colour = QColor(C["accent"] if i == self._phase else C["line_str"])
            p.setBrush(colour)
            p.drawEllipse(QPointF(5 + i * 12, 5), 3.5, 3.5)
        p.end()


class SetupDialog(QDialog):
    """First-run check. Reuses an existing Ollama if there is one, and never
    downloads anything without an explicit click."""

    def __init__(self, parent=None, qss: str = "", mono: str = "Consolas"):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} setup")
        self.setModal(True)
        self.setMinimumSize(620, 470)
        self.setStyleSheet(qss)
        self.worker = None
        self.ok = False

        v = QVBoxLayout(self)
        v.setContentsMargins(26, 24, 26, 22)
        v.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(14)
        logo = QLabel()
        logo.setPixmap(paint_logo(48))
        logo.setFixedSize(48, 48)
        head.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        name = QLabel(f"Setting up {APP_NAME}")
        name.setObjectName("aboutName")
        titles.addWidget(name)
        self.step = TypeLine("wizStep")
        titles.addWidget(self.step)
        head.addLayout(titles, 1)
        v.addLayout(head)

        self.pulse = Pulse()
        self.pulse.hide()
        v.addWidget(self.pulse, 0, Qt.AlignmentFlag.AlignLeft)

        self.detail = QLabel("")
        self.detail.setObjectName("wizDetail")
        self.detail.setWordWrap(True)
        v.addWidget(self.detail)

        self.bar = QProgressBar()
        self.bar.setObjectName("wizBar")
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.hide()
        v.addWidget(self.bar)

        self.console = QPlainTextEdit()
        self.console.setObjectName("wizConsole")
        self.console.setReadOnly(True)
        self.console.setFont(mono_font(mono, CODE_PX_SM))
        self.console.setFrameShape(QFrame.Shape.NoFrame)
        v.addWidget(self.console, 1)

        self.where = QLabel("")
        self.where.setObjectName("wizWhere")
        self.where.setWordWrap(True)
        v.addWidget(self.where)

        row = QHBoxLayout()
        self.folder_btn = QPushButton("Change folder…")
        self.folder_btn.setObjectName("ghost")
        self.folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.folder_btn.setToolTip("Choose where CriGent keeps Ollama, your models "
                                   "and your chats. Pick a drive with room to spare.")
        self.folder_btn.clicked.connect(self._pick_folder)
        row.addWidget(self.folder_btn)
        self.locate_btn = QPushButton("I already have Ollama…")
        self.locate_btn.setObjectName("ghost")
        self.locate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.locate_btn.setToolTip("Point CriGent at an ollama.exe that is already "
                                   "on this machine instead of downloading one.")
        self.locate_btn.clicked.connect(self._locate)
        row.addWidget(self.locate_btn)
        row.addStretch()
        self.skip_btn = QPushButton("Continue without it")
        self.skip_btn.setObjectName("ghost")
        self.skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skip_btn.clicked.connect(self.reject)
        self.go_btn = QPushButton("Install")
        self.go_btn.setObjectName("primary")
        self.go_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.go_btn.clicked.connect(self._start)
        row.addWidget(self.skip_btn)
        row.addWidget(self.go_btn)
        v.addLayout(row)

        QTimer.singleShot(120, self._probe)

    # -- helpers ---------------------------------------------------------- #
    def _say(self, text: str):
        self.step.type(text)

    def _emit(self, line: str):
        self.console.appendPlainText(line)
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum())

    # -- flow ------------------------------------------------------------- #
    def _probe(self):
        self._say("Checking what is already on this machine")
        self.pulse.start()
        self._show_where()
        self._emit(f"{APP_NAME} data folder: {ROOT}")
        self._emit(f"program folder: {program_dir()}")

        # A server that already answers is the strongest signal there is: we can
        # chat, list and delete models through the API without ever locating the
        # binary. Checking this first stops us pushing a 1.5 GB download at
        # someone whose Ollama is running happily from a path we cannot guess.
        self._emit(f"GET http://{HOST}/api/tags")
        try:
            tags = requests.get(TAGS_URL, timeout=3).json().get("models", [])
            self._emit(f"server responded — {len(tags)} model(s) installed")
            for m in tags[:6]:
                self._emit(f"  {m.get('name')}")
            self.pulse.stop()
            self._say("Ollama is already running — using it")
            self.detail.setText(
                "Nothing to download. CriGent talks to the Ollama already running on "
                f"this machine at {HOST}.")
            self._finish_probe_ok()
            return
        except Exception:                                          # noqa: BLE001
            self._emit("no server answered on that port")

        exe = ollama_exe()
        if exe:
            self._emit(f"found ollama: {exe}")
            try:
                res = subprocess.run([str(exe), "--version"], capture_output=True,
                                     text=True, errors="replace", timeout=15,
                                     creationflags=NO_WINDOW)
                out = (res.stdout or res.stderr).strip().splitlines()
                if out:
                    self._emit(out[-1])
            except Exception as exc:                              # noqa: BLE001
                self._emit(f"version check failed: {exc}")
            self.pulse.stop()
            self._say("Ollama is already installed — using it")
            self.detail.setText(
                "Nothing to download. CriGent will drive the copy already on this "
                "machine and start it when needed.")
            self._finish_probe_ok()
            return

        self.pulse.stop()
        self._emit("no ollama.exe on PATH, in the usual install folders, or saved")
        self._say("Ollama is not installed yet")
        self.detail.setText(
            f"{APP_NAME} needs the Ollama runtime to load models.\n\n"
            f"Already have it somewhere custom? Choose “I already have Ollama…” and "
            f"point at your ollama.exe — nothing is downloaded and the choice is "
            f"remembered.\n\n"
            f"Otherwise Install fetches the official build ({OLLAMA_DL_MB:,} MB) and "
            f"unpacks it into {ROOT / 'ollama'} — no admin rights, nothing added to "
            f"PATH, nothing outside that folder touched.")
        self.go_btn.setEnabled(True)
        self.skip_btn.show()
        self.locate_btn.show()
        self.folder_btn.show()

    def _finish_probe_ok(self):
        """Shared 'we're good, just continue' ending for every success path."""
        self.go_btn.setText("Continue")
        try:
            self.go_btn.clicked.disconnect()
        except TypeError:
            pass                        # _probe can run twice; nothing connected yet
        self.go_btn.clicked.connect(self._accept_existing)
        self.go_btn.setEnabled(True)
        self.skip_btn.hide()
        self.locate_btn.hide()
        self.folder_btn.hide()

    def _show_where(self):
        gb = free_gb(ROOT)
        warn = "  ⚠ that is tight for models" if gb < 25 else ""
        self.where.setText(
            f"Storage folder: {ROOT}\n"
            f"Ollama, your models and your chats all go here — {gb:,.0f} GB free"
            f"{warn}")

    def _pick_folder(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose where CriGent stores everything", str(ROOT.parent))
        if not chosen:
            return
        target = Path(chosen)
        if target.name.lower() != APP_NAME.lower() and not (target / "settings.json").exists():
            target = target / f"{APP_NAME}-data"
        if not is_writable(target):
            self._emit(f"cannot write to {target}")
            return
        set_data_dir(target)
        self._emit(f"storage folder set to {target}")
        self._show_where()
        self._probe()

    def _accept_existing(self):
        self.ok = True
        self.accept()

    def _locate(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ollama.exe", str(Path.home()), "ollama.exe;;All files (*)")
        if not path:
            return
        chosen = Path(path)
        if chosen.name.lower() != "ollama.exe":
            self._emit(f"that is not ollama.exe: {chosen.name}")
            return
        settings = {}
        if SETTINGS_PATH.exists():
            try:
                settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:                                      # noqa: BLE001
                settings = {}
        settings["ollama_path"] = str(chosen)
        try:
            _atomic_write_json(SETTINGS_PATH, settings)
        except OSError as exc:
            self._emit(f"could not save that choice: {exc}")
            return
        self._emit(f"using ollama at: {chosen}")
        self.ok = True
        self._probe()

    def _start(self):
        self.go_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.folder_btn.setEnabled(False)
        self.bar.show()
        self.pulse.start()
        self.worker = SetupWorker(ROOT / "ollama")
        self.worker.step.connect(self._say)
        self.worker.log.connect(self._emit)
        self.worker.pct.connect(self._on_pct)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._fail)
        self.worker.start()

    def _on_pct(self, value: int):
        if value < 0:
            self.bar.setRange(0, 0)                # indeterminate
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(value)

    def _done(self):
        self.pulse.stop()
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self._say("All set")
        self.detail.setText("Ollama is installed. You can add a model from the "
                            "Models page whenever you are ready.")
        self.ok = True
        self.go_btn.setText("Start")
        self.go_btn.setEnabled(True)
        self.go_btn.clicked.disconnect()
        self.go_btn.clicked.connect(self.accept)
        self.skip_btn.hide()

    def _fail(self, err: str):
        self.pulse.stop()
        self.bar.hide()
        self._say("That did not work")
        self.detail.setText(f"{err}\n\nYou can close this and install Ollama "
                            "yourself, then reopen CriGent.")
        self._emit(f"error: {err}")
        self.skip_btn.setEnabled(True)
        self.skip_btn.setText("Close")
        self.go_btn.setEnabled(True)
        self.go_btn.setText("Retry")

    def closeEvent(self, ev):
        if still_running(self.worker):
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(ev)


class SkillCard(QFrame):
    """A skill the model proposes to save, held for explicit user approval."""

    save_clicked = pyqtSignal()
    discard_clicked = pyqtSignal()

    def __init__(self, name: str, content: str, mono: str):
        super().__init__()
        self.setObjectName("skillCard")
        self.setMaximumWidth(760)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 14)
        v.setSpacing(10)

        header = QLabel("Save this skill?")
        header.setObjectName("skillHeader")
        v.addWidget(header)

        name_lbl = QLabel(name)
        name_lbl.setObjectName("skillName")
        v.addWidget(name_lbl)

        self.content_box = QPlainTextEdit(content)
        self.content_box.setObjectName("toolOutput")
        self.content_box.setReadOnly(True)
        self._body_font = mono_font(mono, CODE_PX_SM)   # matches #toolOutput
        self.content_box.setFont(self._body_font)
        self.content_box.setFrameShape(QFrame.Shape.NoFrame)
        self.content_box.setFixedHeight(fit_height(self.content_box, self._body_font, cap=220))
        v.addWidget(self.content_box)

        row = QHBoxLayout()
        row.addStretch()
        self.discard_btn = QPushButton("Discard")
        self.discard_btn.setObjectName("ghost")
        self.discard_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.discard_btn.clicked.connect(self._on_discard)
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("primary")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.clicked.connect(self._on_save)
        row.addWidget(self.discard_btn)
        row.addWidget(self.save_btn)
        v.addLayout(row)

    def _on_save(self):
        self.save_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.save_btn.setText("Saved")
        self.save_clicked.emit()

    def _on_discard(self):
        self.save_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.discard_btn.setText("Discarded")
        self.discard_clicked.emit()


class SkillEditDialog(QDialog):
    """Manual create/edit dialog for a skill, used from the sidebar's + New skill / Edit."""

    def __init__(self, parent, mono: str, qss: str, name: str = "", content: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Edit Skill" if name else "New Skill")
        self.setMinimumSize(480, 380)
        self.setStyleSheet(qss)

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(8)

        v.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit(name)
        self.name_edit.setObjectName("input")
        v.addWidget(self.name_edit)

        v.addWidget(QLabel("Content"))
        self.content_edit = QPlainTextEdit(content)
        self.content_edit.setObjectName("toolCmd")
        self.content_edit.setFont(mono_font(mono, CODE_PX))
        v.addWidget(self.content_edit, 1)

        row = QHBoxLayout()
        row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("ghost")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save")
        save_btn.setObjectName("primary")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        row.addWidget(cancel_btn)
        row.addWidget(save_btn)
        v.addLayout(row)

    def _on_save(self):
        if self.name_edit.text().strip() and self.content_edit.toPlainText().strip():
            self.accept()

    def values(self):
        return self.name_edit.text().strip(), self.content_edit.toPlainText().strip()


class WebCard(QFrame):
    """An in-progress or completed web search/fetch. Read-only, so it auto-runs — this is
    purely a transparency log, not a confirmation gate."""

    def __init__(self, kind: str, query: str, mono: str):
        super().__init__()
        self.setObjectName("webCard")
        self.setMaximumWidth(760)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 14)
        v.setSpacing(8)

        label = "Web search" if kind == "search" else "Reading page"
        head_row = QHBoxLayout()
        header = QLabel(label)
        header.setObjectName("webHeader")
        head_row.addWidget(header)
        head_row.addStretch()
        self.status_lbl = QLabel("running…")
        self.status_lbl.setObjectName("toolStatus")
        head_row.addWidget(self.status_lbl)
        v.addLayout(head_row)

        query_lbl = QLabel(query)
        query_lbl.setObjectName("webQuery")
        query_lbl.setWordWrap(True)
        v.addWidget(query_lbl)

        self.output = QPlainTextEdit()
        self.output.setObjectName("toolOutput")
        self.output.setReadOnly(True)
        self._out_font = mono_font(mono, CODE_PX_SM)
        self.output.setFont(self._out_font)
        self.output.setFrameShape(QFrame.Shape.NoFrame)
        self.output.hide()
        v.addWidget(self.output)

    def show_result(self, text: str, ok: bool = True):
        self.status_lbl.setText("done" if ok else "error")
        if len(text) > 4000:
            text = text[:4000] + "\n…(truncated)"
        self.output.setPlainText(text or "(no results)")
        self.output.setFixedHeight(fit_height(self.output, self._out_font, cap=260))
        self.output.show()


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #

class CriGent(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mono = mono_family()
        self.messages = []
        self.chat_worker = None
        self.bubble = None
        self.buffer = ""
        self._last_flushed = ""
        self.current_model = ""
        self.tool_round = 0
        self._cmd_worker = None

        # Workers must stay referenced until they actually finish. Reassigning
        # self.chat_worker / self._cmd_worker was dropping the last reference to a
        # still-running QThread, which Qt aborts the whole process over.
        self._workers = []

        self.current_chat_id = None
        self.chats_dir = CHATS_DIR
        self.chats_dir.mkdir(exist_ok=True)
        self.skills = self._read_skills_file()
        self.prompts = self._read_prompts_file()
        self.settings = self._read_settings()
        self._import_worker = None

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1400, 820)
        self.setStyleSheet(self._qss())
        self._build()
        self._reload_chat_list()
        self._reload_skill_list()

        self.flush = QTimer(self)                    # batch tokens -> smooth repaints
        self.flush.setInterval(40)
        self.flush.timeout.connect(self._flush)

        # Building the first CodeBlock costs ~2s while Qt loads the monospace
        # font and primes its text layout. Pay that during startup instead of
        # stalling mid-reply the first time the model writes any code.
        QTimer.singleShot(0, self._warm_fonts)

        self.gpu = GpuWorker()
        self.gpu.sample.connect(self._on_gpu)
        self.gpu.start()

        QTimer.singleShot(200, self._check_ollama)

    # -- layout ----------------------------------------------------------- #
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        col = QVBoxLayout(root)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        head = QWidget()
        head.setObjectName("head")
        hb = QHBoxLayout(head)
        hb.setContentsMargins(20, 0, 20, 0)
        hb.setSpacing(12)

        mark = QLabel()
        mark.setObjectName("brandMark")
        mark.setPixmap(paint_logo(28))
        mark.setFixedSize(28, 28)
        hb.addWidget(mark)

        title = QLabel(APP_NAME)
        title.setObjectName("title")
        hb.addWidget(title)
        hb.addSpacing(18)

        # Nav lives in the header rather than a second chrome band below it.
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        nav = QHBoxLayout()
        nav.setSpacing(2)
        for i, name in enumerate(("Chat", "GPU", "Prompts", "Models", "About")):
            btn = QPushButton(name)
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setChecked(i == 0)
            self.nav_group.addButton(btn, i)
            nav.addWidget(btn)
        self.nav_group.idClicked.connect(self._on_nav)
        hb.addLayout(nav)
        hb.addStretch()

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelPicker")
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        hb.addWidget(self.model_combo)

        self.compute_combo = QComboBox()
        self.compute_combo.setObjectName("computePicker")
        self.compute_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        for key, label, _, hint in COMPUTE_MODES:
            self.compute_combo.addItem(label, userData=key)
            self.compute_combo.setItemData(self.compute_combo.count() - 1, hint,
                                           Qt.ItemDataRole.ToolTipRole)
        idx = self.compute_combo.findData(self.settings.get("compute", "auto"))
        self.compute_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.compute_combo.setToolTip(
            "Where the model runs. Changing this makes Ollama reload the model, so "
            "the next reply takes longer to start.")
        self.compute_combo.currentIndexChanged.connect(self._on_compute_changed)
        hb.addWidget(self.compute_combo)

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        hb.addWidget(self.status_dot)
        self.status = QLabel("connecting…")
        self.status.setObjectName("status")
        hb.addWidget(self.status)
        head.setFixedHeight(56)
        col.addWidget(head)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._chat_tab())
        self.stack.addWidget(self._gpu_tab())
        self.stack.addWidget(self._prompts_tab())
        self.stack.addWidget(self._models_tab())
        self.stack.addWidget(self._about_tab())
        col.addWidget(self.stack, 1)

    def _on_nav(self, index: int):
        if hasattr(self, "gpu"):
            self.gpu.set_active(index == 1)
        # Also drive the button state, so programmatic navigation keeps the
        # highlighted tab in sync with the visible page.
        btn = self.nav_group.button(index)
        if btn and not btn.isChecked():
            btn.setChecked(True)
        self.stack.setCurrentIndex(index)
        if index == 3:
            self._reload_model_list()

    def _chat_tab(self) -> QWidget:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._chats_sidebar())
        outer.addWidget(self._chat_center(), 1)
        outer.addWidget(self._skills_sidebar())
        return page

    def _chats_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidebar")
        panel.setFixedWidth(248)
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 16, 12, 12)
        v.setSpacing(12)

        new_btn = QPushButton("New chat")
        new_btn.setObjectName("newBtn")
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.clicked.connect(self._new_chat)
        v.addWidget(new_btn)

        title = QLabel("Recent")
        title.setObjectName("sidebarTitle")
        v.addWidget(title)

        self.chat_list = QListWidget()
        self.chat_list.setObjectName("chatList")
        self.chat_list.setSpacing(1)
        # Rows must fit the sidebar: elide long titles instead of growing the row
        # and forcing a horizontal scrollbar.
        self.chat_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.chat_list.setWordWrap(False)
        self.chat_list.setUniformItemSizes(True)
        self.chat_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_list.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)          # smooth, not row-jumping
        self.chat_list.itemClicked.connect(self._on_chat_item_clicked)
        self.chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self._chat_context_menu)
        v.addWidget(self.chat_list, 1)

        self.chat_empty = QLabel("No saved chats yet.")
        self.chat_empty.setObjectName("listEmpty")
        self.chat_empty.setWordWrap(True)
        v.addWidget(self.chat_empty)
        return panel

    def _skills_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidebarRight")
        panel.setFixedWidth(268)
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 16, 12, 12)
        v.setSpacing(12)

        new_btn = QPushButton("New skill")
        new_btn.setObjectName("newBtn")
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.clicked.connect(self._new_skill)
        v.addWidget(new_btn)

        title = QLabel("Skills")
        title.setObjectName("sidebarTitle")
        v.addWidget(title)

        self.skill_list = QListWidget()
        self.skill_list.setObjectName("skillList")
        self.skill_list.setSpacing(1)
        self.skill_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.skill_list.setWordWrap(False)
        self.skill_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.skill_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.skill_list.itemDoubleClicked.connect(self._on_skill_item_double_clicked)
        self.skill_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.skill_list.customContextMenuRequested.connect(self._skill_context_menu)
        v.addWidget(self.skill_list, 1)

        self.skill_empty = QLabel(
            "No skills yet. Create one, or ask the model to save a skill for you.")
        self.skill_empty.setObjectName("listEmpty")
        self.skill_empty.setWordWrap(True)
        v.addWidget(self.skill_empty)

        hint = QLabel("Tick to apply · double-click to edit")
        hint.setObjectName("sidebarHint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return panel

    def _chat_center(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        holder.setObjectName("chatArea")
        hrow = QHBoxLayout(holder)
        hrow.setContentsMargins(24, 24, 24, 24)
        hrow.setSpacing(0)
        column = QWidget()
        column.setMaximumWidth(COLUMN_MAX)
        hrow.addStretch(1)
        hrow.addWidget(column, 10)
        hrow.addStretch(1)

        self.feed = QVBoxLayout(column)
        self.feed.setContentsMargins(0, 0, 0, 0)
        self.feed.setSpacing(20)
        self.feed.addStretch()
        self.scroll.setWidget(holder)
        v.addWidget(self.scroll, 1)

        self.hint = QWidget()
        hv = QVBoxLayout(self.hint)
        hv.setContentsMargins(0, 90, 0, 0)
        hv.setSpacing(6)
        empty_title = QLabel("What can I help with?")
        empty_title.setObjectName("emptyTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_sub = QLabel("Running locally on your RTX 5080 — nothing leaves this machine "
                           "unless you enable web search.")
        empty_sub.setObjectName("emptySub")
        empty_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_sub.setWordWrap(True)
        hv.addWidget(empty_title)
        hv.addWidget(empty_sub)
        self.feed.insertWidget(0, self.hint)

        bar = QWidget()
        bar.setObjectName("composer")
        bar_row = QHBoxLayout(bar)
        bar_row.setContentsMargins(24, 14, 24, 16)
        bar_row.setSpacing(0)
        inner = QWidget()
        inner.setMaximumWidth(COLUMN_MAX)
        bar_row.addStretch(1)
        bar_row.addWidget(inner, 10)
        bar_row.addStretch(1)
        cb = QVBoxLayout(inner)
        cb.setContentsMargins(0, 0, 0, 0)
        cb.setSpacing(8)

        self.autorun_warn = QLabel(
            "⚠  Auto-run is on — proposed commands execute immediately, with no review step.")
        self.autorun_warn.setObjectName("autorunWarn")
        self.autorun_warn.setWordWrap(True)
        self.autorun_warn.hide()
        cb.addWidget(self.autorun_warn)

        shell = QFrame()
        shell.setObjectName("inputShell")
        sv = QVBoxLayout(shell)
        sv.setContentsMargins(6, 6, 6, 6)
        sv.setSpacing(6)

        self.input = QTextEdit()
        self.input.setObjectName("input")
        self.input.setPlaceholderText(f"Message {APP_NAME}…")
        self.input.setAcceptRichText(False)
        self.input.setFixedHeight(72)
        self.input.installEventFilter(self)
        sv.addWidget(self.input)

        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 2)
        row.setSpacing(6)

        self.tools_check = QPushButton("Tools")
        self.tools_check.setObjectName("pillTools")
        self.tools_check.setCheckable(True)
        self.tools_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tools_check.setToolTip(
            "Lets the model propose PowerShell commands. Nothing runs without your "
            "explicit click on each one.")
        self.tools_check.toggled.connect(self._on_tools_toggled)
        row.addWidget(self.tools_check)

        self.autorun_check = QPushButton("Auto-run")
        self.autorun_check.setObjectName("pillAutorun")
        self.autorun_check.setCheckable(True)
        self.autorun_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.autorun_check.setToolTip(
            "Skips the Run/Deny step entirely — every proposed command executes right away. "
            "Only enable this if you trust what you're about to ask for.")
        self.autorun_check.setEnabled(False)
        self.autorun_check.toggled.connect(self.autorun_warn.setVisible)
        row.addWidget(self.autorun_check)

        self.web_check = QPushButton("Web")
        self.web_check.setObjectName("pillWeb")
        self.web_check.setCheckable(True)
        self.web_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.web_check.setToolTip(
            "Lets the model search the web and read pages automatically — read-only, so there's "
            "no confirmation step, but every search/fetch is shown as a card in the chat.")
        row.addWidget(self.web_check)

        self.skills_check = QPushButton("Skills")
        self.skills_check.setObjectName("pillSkills")
        self.skills_check.setCheckable(True)
        self.skills_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skills_check.setToolTip(
            "Lets the model write and amend skills. It can only propose one — you still "
            "review and click Save before anything is written to your skill list.")
        row.addWidget(self.skills_check)

        row.addStretch()
        self.tokens_lbl = QLabel("")
        self.tokens_lbl.setObjectName("meta")
        row.addWidget(self.tokens_lbl)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._send_or_stop)
        row.addWidget(self.send_btn)
        sv.addLayout(row)

        cb.addWidget(shell)
        self.composer_hint = QLabel("Enter to send · Shift+Enter for a new line")
        self.composer_hint.setObjectName("composerHint")
        self.composer_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cb.addWidget(self.composer_hint)
        v.addWidget(bar)
        return page

    def _gpu_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("chatArea")
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 22, 24, 22)
        v.setSpacing(16)

        self.gpu_name = QLabel("Detecting GPU…")
        self.gpu_name.setObjectName("gpuName")
        v.addWidget(self.gpu_name)

        gauges = Card()
        g = QGridLayout()
        g.setSpacing(6)
        self.g_util = RingGauge("GPU LOAD", C["accent"])
        self.g_mem = RingGauge("VRAM", C["violet"])
        self.g_temp = RingGauge("TEMP", C["green"], unit="°")
        self.g_pow = RingGauge("POWER", C["amber"])
        for i, w in enumerate((self.g_util, self.g_mem, self.g_temp, self.g_pow)):
            g.addWidget(w, 0, i)
        gauges.box.addLayout(g)
        v.addWidget(gauges)

        graphs = QHBoxLayout()
        graphs.setSpacing(16)
        c1 = Card("GPU load · last 2 min")
        self.spark_util = Sparkline(C["accent"])
        c1.box.addWidget(self.spark_util)
        c2 = Card("VRAM · last 2 min")
        self.spark_mem = Sparkline(C["violet"])
        c2.box.addWidget(self.spark_mem)
        graphs.addWidget(c1)
        graphs.addWidget(c2)
        v.addLayout(graphs)

        stats = Card("Details")
        self.detail = QLabel("—")
        self.detail.setObjectName("detail")
        self.detail.setTextFormat(Qt.TextFormat.RichText)
        stats.box.addWidget(self.detail)
        v.addWidget(stats)
        v.addStretch()
        return page

    def _page(self, title: str, subtitle: str):
        """Standard page scaffold: heading + scrollable body column."""
        page = QWidget()
        page.setObjectName("chatArea")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        holder.setObjectName("chatArea")
        hrow = QHBoxLayout(holder)
        hrow.setContentsMargins(28, 26, 28, 28)
        column = QWidget()
        column.setMaximumWidth(COLUMN_MAX)
        hrow.addStretch(1)
        hrow.addWidget(column, 10)
        hrow.addStretch(1)
        body = QVBoxLayout(column)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)

        head = QLabel(title)
        head.setObjectName("pageTitle")
        body.addWidget(head)
        sub = QLabel(subtitle)
        sub.setObjectName("pageSub")
        sub.setWordWrap(True)
        body.addWidget(sub)

        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)
        return page, body

    # -- prompts page ----------------------------------------------------- #
    def _prompts_tab(self) -> QWidget:
        page, body = self._page(
            "Enrichment prompts",
            "These are prepended to your message when the matching toggle is on. Edit them to "
            "change how the model uses tools, the web, and skills.")

        self.prompt_boxes = {}
        for key, label, hint in PROMPT_META:
            card = Card()
            head = QHBoxLayout()
            name = QLabel(label)
            name.setObjectName("skillName")
            head.addWidget(name)
            head.addStretch()
            reset = QPushButton("Reset to default")
            reset.setObjectName("ghostSm")
            reset.setCursor(Qt.CursorShape.PointingHandCursor)
            reset.clicked.connect(lambda _=False, k=key: self._reset_prompt(k))
            head.addWidget(reset)
            card.box.addLayout(head)

            desc = QLabel(hint)
            desc.setObjectName("pageSub")
            desc.setWordWrap(True)
            card.box.addWidget(desc)

            box = QPlainTextEdit(self.prompts.get(key, ""))
            box.setObjectName("promptBox")
            box.setFont(mono_font(self.mono, CODE_PX))
            box.setMinimumHeight(210)
            card.box.addWidget(box)
            self.prompt_boxes[key] = box
            body.addWidget(card)

        actions = QHBoxLayout()
        self.prompt_status = QLabel("")
        self.prompt_status.setObjectName("meta")
        actions.addWidget(self.prompt_status)
        actions.addStretch()
        save = QPushButton("Save prompts")
        save.setObjectName("primary")
        save.setCursor(Qt.CursorShape.PointingHandCursor)
        save.clicked.connect(self._save_prompts)
        actions.addWidget(save)
        body.addLayout(actions)
        body.addStretch()
        return page

    # -- settings --------------------------------------------------------- #
    def _read_settings(self) -> dict:
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:                                      # noqa: BLE001
                pass
        return {}

    def _save_settings(self):
        _atomic_write_json(SETTINGS_PATH, self.settings)

    def _num_gpu(self):
        """Ollama's num_gpu for the selected compute mode, or None for auto."""
        key = self.compute_combo.currentData() if hasattr(self, "compute_combo") else "auto"
        for mode, _, num_gpu, _hint in COMPUTE_MODES:
            if mode == key:
                return num_gpu
        return None

    def _on_compute_changed(self, _index: int):
        key = self.compute_combo.currentData()
        self.settings["compute"] = key
        self._save_settings()
        label = next((l for k, l, _n, _h in COMPUTE_MODES if k == key), key)
        note = QLabel(f"— compute set to {label}. Ollama reloads the model on the "
                      f"next message, so it will be slower to start. —")
        note.setObjectName("hint")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setWordWrap(True)
        self.feed.insertWidget(self.feed.count() - 1, note)
        self._scroll_down()

    def _read_prompts_file(self) -> dict:
        data = dict(DEFAULT_PROMPTS)
        if PROMPTS_PATH.exists():
            try:
                stored = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
                for key in DEFAULT_PROMPTS:
                    if isinstance(stored.get(key), str) and stored[key].strip():
                        data[key] = stored[key]
            except Exception:                                      # noqa: BLE001
                pass
        return data

    def _save_prompts(self):
        for key, box in self.prompt_boxes.items():
            self.prompts[key] = box.toPlainText().strip()
        _atomic_write_json(PROMPTS_PATH, self.prompts)
        self.prompt_status.setText("Saved — applies to your next message.")
        timer = QTimer(self.prompt_status)          # parented: dies with the widget
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.prompt_status.setText(""))
        timer.start(2600)

    def _reset_prompt(self, key: str):
        self.prompt_boxes[key].setPlainText(DEFAULT_PROMPTS[key])
        self.prompt_status.setText(f"{key.title()} reset — not saved yet.")

    # -- about page ------------------------------------------------------- #
    def _about_tab(self) -> QWidget:
        page, body = self._page("About", "")

        hero = Card()
        top = QHBoxLayout()
        top.setSpacing(16)
        logo = QLabel()
        logo.setPixmap(paint_logo(72))
        logo.setFixedSize(72, 72)
        top.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)

        titles = QVBoxLayout()
        titles.setSpacing(4)
        name = QLabel(APP_NAME)
        name.setObjectName("aboutName")
        titles.addWidget(name)
        tag = QLabel(APP_TAGLINE)
        tag.setObjectName("pageSub")
        tag.setWordWrap(True)
        titles.addWidget(tag)
        blurb = QLabel(
            "Chat with local models through Ollama, let the agent run commands and search "
            "the web with your approval, save reusable skills, watch the GPU live, and "
            "import new models straight from a .gguf file.")
        blurb.setObjectName("pageSub")
        blurb.setWordWrap(True)
        titles.addWidget(blurb)
        top.addLayout(titles, 1)
        hero.box.addLayout(top)
        body.addWidget(hero)

        dev = Card("Developer")
        who = QLabel(DEV_NAME)
        who.setObjectName("skillName")
        dev.box.addWidget(who)

        links = QHBoxLayout()
        links.setSpacing(8)
        site = QPushButton("crimsonlingua.com")
        site.setObjectName("linkBtn")
        site.setCursor(Qt.CursorShape.PointingHandCursor)
        site.setToolTip(f"Open {DEV_SITE} in your browser")
        site.clicked.connect(lambda: self._open_url(DEV_SITE))
        links.addWidget(site)

        li = QPushButton("LinkedIn")
        li.setObjectName("linkBtn")
        li.setCursor(Qt.CursorShape.PointingHandCursor)
        li.setToolTip(f"Open {DEV_LINKEDIN} in your browser")
        li.clicked.connect(lambda: self._open_url(DEV_LINKEDIN))
        links.addWidget(li)
        links.addStretch()
        dev.box.addLayout(links)

        self.about_status = QLabel("")
        self.about_status.setObjectName("meta")
        dev.box.addWidget(self.about_status)
        body.addWidget(dev)
        body.addStretch()
        return page

    def _open_url(self, url: str):
        """Hand the link to the user's default browser."""
        opened = QDesktopServices.openUrl(QUrl(url))
        self.about_status.setText(
            f"Opened {url} in your browser." if opened else f"Could not open {url}.")
        timer = QTimer(self.about_status)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.about_status.setText(""))
        timer.start(3000)

    # -- models page ------------------------------------------------------ #
    def _models_tab(self) -> QWidget:
        page, body = self._page(
            "Models",
            "Every model registered with your local Ollama. Add a .gguf from anywhere on this "
            "machine and the app writes the Modelfile and imports it for you.")

        row = QHBoxLayout()
        add = QPushButton("Add model…")
        add.setObjectName("primary")
        add.setCursor(Qt.CursorShape.PointingHandCursor)
        add.clicked.connect(self._add_model)
        row.addWidget(add)
        refresh = QPushButton("Refresh")
        refresh.setObjectName("ghost")
        refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh.clicked.connect(self._reload_model_list)
        row.addWidget(refresh)

        setup = QPushButton("Ollama setup…")
        setup.setObjectName("ghost")
        setup.setCursor(Qt.CursorShape.PointingHandCursor)
        setup.setToolTip("Re-run the setup check, or point CriGent at an ollama.exe "
                         "somewhere else on this machine.")
        setup.clicked.connect(self._run_setup)
        row.addWidget(setup)

        store = QPushButton("Model folder…")
        store.setObjectName("ghost")
        store.setCursor(Qt.CursorShape.PointingHandCursor)
        store.setToolTip("Point CriGent at an existing Ollama model store instead of "
                         "re-importing models into a fresh one.")
        store.clicked.connect(self._pick_models_dir)
        row.addWidget(store)
        row.addStretch()
        body.addLayout(row)

        self.runtime_lbl = QLabel("")
        self.runtime_lbl.setObjectName("pageSub")
        self.runtime_lbl.setWordWrap(True)
        body.addWidget(self.runtime_lbl)

        self.import_status = QLabel("")
        self.import_status.setObjectName("importStatus")
        self.import_status.setWordWrap(True)
        self.import_status.hide()
        body.addWidget(self.import_status)

        card = Card()
        self.model_list = QListWidget()
        self.model_list.setObjectName("modelList")
        self.model_list.setSpacing(2)
        self.model_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.model_list.setWordWrap(False)
        self.model_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.model_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.model_list.setMinimumHeight(260)
        card.box.addWidget(self.model_list)
        body.addWidget(card)
        body.addStretch()
        return page

    def _run_setup(self):
        dlg = SetupDialog(self, qss=self._qss(), mono=self.mono)
        dlg.setWindowIcon(app_icon())
        dlg.exec()
        if dlg.ok:
            self.settings["setup_done"] = True
            self.settings.pop("_skipped", None)
            self._save_settings()
        self._reload_model_list()
        self._check_ollama()

    def _pick_models_dir(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Select the Ollama model folder (the one containing 'blobs')",
            str(models_dir()))
        if not chosen:
            return
        path = Path(chosen)
        if not (path / "blobs").is_dir():
            reply = QMessageBox.question(
                self, "No blobs folder",
                f"{path} does not contain a 'blobs' folder, so it is probably not an "
                "Ollama model store. Use it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.settings["models_dir"] = str(path)
        self._save_settings()
        self.import_status.setText(
            f"Model folder set to {path}. Restart CriGent so Ollama picks it up.")
        self.import_status.show()
        self._reload_model_list()

    def _reload_model_list(self):
        exe = ollama_exe()
        store = models_dir()
        self.runtime_lbl.setText(
            (f"Runtime: {exe}" if exe else
             "Runtime: no ollama.exe located — use “Ollama setup…” to point at one "
             "or install it.") + f"\nModel folder: {store}")
        self.model_list.clear()
        try:
            models = requests.get(TAGS_URL, timeout=3).json().get("models", [])
        except Exception as exc:                                   # noqa: BLE001
            item = QListWidgetItem(f"Cannot reach Ollama — {exc}")
            self.model_list.addItem(item)
            return
        if not models:
            self.model_list.addItem(QListWidgetItem("No models installed yet."))
            return
        for m in sorted(models, key=lambda x: x["name"]):
            raw = m["name"].split(":")[0]
            det = m.get("details", {})
            size_gb = m.get("size", 0) / 1e9
            bits = [b for b in (det.get("parameter_size"), det.get("quantization_level"),
                                f"{size_gb:.1f} GB") if b and b != "unknown"]
            display, _ = model_label(raw)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, raw)
            row = ListRow(raw, display, "  ·  ".join(bits))
            item.setSizeHint(QSize(row.sizeHint().width(), row.height()))
            self.model_list.addItem(item)
            self.model_list.setItemWidget(item, row)
            row.delete_requested.connect(self._delete_model)

    def _add_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a GGUF model file", str(Path.home()),
            "GGUF models (*.gguf);;All files (*)")
        if not path:
            return
        gguf = Path(path)
        suggested = re.sub(r"[^a-z0-9_-]", "", gguf.stem.lower().replace(".", "-"))[:32] or "model"
        name, ok = QInputDialog.getText(
            self, "Name this model",
            "Short name (letters, numbers, dashes):", text=suggested)
        if not ok:
            return
        name = re.sub(r"[^a-z0-9_-]", "", name.strip().lower())
        if not name:
            QMessageBox.warning(self, "Invalid name", "Please use letters, numbers or dashes.")
            return
        if still_running(self._import_worker):
            QMessageBox.information(self, "Import running",
                                    "Another model import is still in progress.")
            return

        self.import_status.setText(f"Importing “{name}” from {gguf.name}…")
        self.import_status.show()
        worker = ModelImportWorker(name, gguf)
        worker.progress.connect(lambda msg: self.import_status.setText(msg))
        worker.finished_ok.connect(self._on_model_imported)
        worker.failed.connect(self._on_model_import_failed)
        self._import_worker = self._track(worker)
        worker.start()

    def _on_model_imported(self, name: str):
        self.import_status.setText(f"“{name}” imported and ready to use.")
        self._reload_model_list()
        self._populate_models()

    def _on_model_import_failed(self, err: str):
        self.import_status.setText(f"Import failed — {err}")

    def _delete_model(self, name: str):
        reply = QMessageBox.question(
            self, "Delete model",
            f"Permanently delete the model “{name}” from Ollama?\n\n"
            "This removes its copy in Ollama's store. The original .gguf file you "
            "imported is left untouched.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._remove_model(name)

    def _remove_model(self, name: str):
        try:
            resp = requests.delete(f"http://{HOST}/api/delete", json={"name": name}, timeout=30)
            resp.raise_for_status()
            self.import_status.setText(f"Deleted “{name}”.")
            self.import_status.show()
        except Exception as exc:                                   # noqa: BLE001
            self.import_status.setText(f"Could not delete “{name}” — {exc}")
            self.import_status.show()
        self._reload_model_list()
        self._populate_models()

    # -- chat ------------------------------------------------------------- #
    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == ev.Type.KeyPress:
            if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                    not (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._send_or_stop()
                return True
        return super().eventFilter(obj, ev)

    def _add_bubble(self, role: str, badge: str = "ASSISTANT") -> Bubble:
        b = Bubble(role, self.mono, badge)
        self.feed.insertWidget(self.feed.count() - 1, b)
        return b

    def _track(self, worker: QThread) -> QThread:
        """Hold a reference until the thread finishes, so it can't be collected
        mid-run (Qt aborts on 'QThread: Destroyed while thread is still running')."""
        self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._untrack(w))
        return worker

    def _untrack(self, worker: QThread):
        if worker in self._workers:
            self._workers.remove(worker)
        # deleteLater destroys the C++ object, but any attribute still pointing
        # at it keeps a dead Python wrapper. Touching that raises RuntimeError,
        # and an exception inside a slot takes the whole app down — which is
        # exactly what happened when sending a second message.
        for attr in ("chat_worker", "_cmd_worker", "_import_worker"):
            if getattr(self, attr, None) is worker:
                setattr(self, attr, None)
        worker.deleteLater()

    def report_error(self, exc: BaseException, detail: str):
        """Called by the crash guard so a caught fault is visible, not silent."""
        try:
            self._notice(f"Something went wrong internally — "
                         f"{type(exc).__name__}: {exc}. CriGent is still running; "
                         f"details were written to {ROOT / 'crash.log'}")
        except Exception:                                          # noqa: BLE001
            pass

    def _warm_fonts(self):
        """Force Qt to load the mono font and lay out a code block once, offscreen."""
        try:
            primer = CodeBlock("warm = True", "python", self.mono)
            primer.editor.document().documentLayout().documentSize()
            primer.deleteLater()
        except Exception:                                          # noqa: BLE001
            pass          # purely an optimisation; never let it break startup

    def _notice(self, text: str):
        """Inline, centred message in the transcript — used for state the user
        needs to act on, rather than a modal."""
        self.hint.hide()
        label = QLabel(text)
        label.setObjectName("hint")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        self.feed.insertWidget(self.feed.count() - 1, label)
        self._scroll_down()

    def _at_bottom(self) -> bool:
        sb = self.scroll.verticalScrollBar()
        return sb.value() >= sb.maximum() - 120

    def _scroll_down(self):
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

    def _send_or_stop(self):
        if still_running(self.chat_worker):
            self.chat_worker.stop()
            return
        text = self.input.toPlainText().strip()
        if not text:
            return

        # Without this the request goes out as {"model": ""} and Ollama answers
        # 400 "model is required" — technically accurate, useless to a newcomer
        # who simply has not imported a model yet.
        if not self.current_model:
            self._notice(
                "No model is selected yet. Open the Models page, choose "
                "“Add model…” and pick a .gguf file — CriGent imports it for you. "
                "It will then appear in the selector at the top.")
            return

        self.hint.hide()
        user = self._add_bubble("user")
        user.set_text(text)
        self.messages.append({"role": "user", "content": text})
        self.input.clear()
        self.tool_round = 0
        self._save_current_chat()
        self._start_worker()

    def _start_worker(self):
        _, badge = model_label(self.current_model)
        self.bubble = self._add_bubble("assistant", badge)
        self.bubble.set_text("")
        self.buffer = ""
        self._last_flushed = ""
        self._scroll_down()

        self.send_btn.setText("Stop")
        self.send_btn.setObjectName("danger")
        self.send_btn.setStyleSheet("")
        self.tokens_lbl.setText("generating…")

        # Each enrichment prompt is opt-in via its toggle, and comes from the
        # user-editable store rather than the module constants.
        sys_parts = []
        if self.tools_check.isChecked():
            sys_parts.append(self.prompts.get("tools", ""))
        if self.web_check.isChecked():
            sys_parts.append(self.prompts.get("web", ""))
        if self.skills_check.isChecked():
            sys_parts.append(self.prompts.get("skills", ""))
        active_skills = self._active_skills_text()
        if active_skills:
            sys_parts.append(active_skills)
        sys_parts = [p for p in sys_parts if p.strip()]

        payload = list(self.messages)
        if sys_parts:
            payload = [{"role": "system", "content": "\n\n".join(sys_parts)}] + payload

        prev = self.chat_worker
        if still_running(prev):
            prev.stop()
            prev.wait(2000)

        self.chat_worker = self._track(
            ChatWorker(payload, self.current_model, self._num_gpu()))
        self.chat_worker.chunk.connect(self._on_chunk)
        self.chat_worker.done.connect(self._on_done)
        self.chat_worker.failed.connect(self._on_failed)
        self.chat_worker.start()
        self.flush.start()

    def _on_chunk(self, piece: str):
        self.buffer += piece

    def _flush(self):
        if not self.bubble:
            return
        if self.buffer == self._last_flushed:
            return                     # nothing new since the last tick
        self._last_flushed = self.buffer
        stick = self._at_bottom()
        self.bubble.set_text(self.buffer)
        if stick:
            self._scroll_down()

    def _finish_ui(self):
        self.flush.stop()
        self._flush()
        self.send_btn.setText("Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.setStyleSheet("")

    def _on_done(self, elapsed: float, tokens: int):
        self._finish_ui()
        if self.buffer:
            self.messages.append({"role": "assistant", "content": self.buffer})
        rate = tokens / elapsed if elapsed > 0 else 0
        stamp = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"
        if self.bubble:
            self.bubble.set_meta(f"⏱ {stamp}  ·  {rate:.1f} tok/s")
        self.tokens_lbl.setText("")
        self.bubble = None
        self._save_current_chat()

        if self.tool_round < MAX_TOOL_ROUNDS:
            tool_match = TOOL_RE.search(self.buffer) if self.tools_check.isChecked() else None
            search_match = SEARCH_RE.search(self.buffer) if self.web_check.isChecked() else None
            fetch_match = FETCH_RE.search(self.buffer) if self.web_check.isChecked() else None

            if tool_match:
                self.tool_round += 1
                self._offer_tool(tool_match.group(1).strip())
            elif search_match:
                self.tool_round += 1
                self._start_search(search_match.group(1).strip())
            elif fetch_match:
                self.tool_round += 1
                self._start_fetch(fetch_match.group(1).strip())
            elif self.skills_check.isChecked():
                skill_match = SKILL_RE.search(self.buffer)
                parsed = parse_skill_block(skill_match.group(1)) if skill_match else None
                if parsed:
                    self.tool_round += 1
                    self._offer_skill(*parsed)

    def _on_failed(self, err: str):
        self._finish_ui()
        if self.bubble:
            self.bubble.set_error(err)
        self.tokens_lbl.setText("")
        self.bubble = None
        self._save_current_chat()

    def _on_tools_toggled(self, checked: bool):
        self.autorun_check.setEnabled(checked)
        if not checked:
            self.autorun_check.setChecked(False)

    def _offer_tool(self, command: str):
        auto = self.autorun_check.isChecked()
        card = ToolCard(command, self.mono, auto=auto)
        self.feed.insertWidget(self.feed.count() - 1, card)
        self._scroll_down()
        if auto:
            self._run_tool(card, command)
        else:
            card.run_clicked.connect(lambda c=card, cmd=command: self._run_tool(c, cmd))
            card.deny_clicked.connect(lambda c=card, cmd=command: self._deny_tool(c, cmd))

    def _run_tool(self, card: ToolCard, command: str):
        worker = CommandWorker(command)
        worker.result.connect(
            lambda out, err, code, c=card, cmd=command: self._tool_result(c, cmd, out, err, code))
        self._cmd_worker = self._track(worker)
        worker.start()

    def _tool_result(self, card: ToolCard, command: str, stdout: str, stderr: str, code: int):
        card.show_result(stdout, stderr, code)
        summary = f"$ {command}\nexit code: {code}"
        if stdout.strip():
            summary += f"\n\nstdout:\n{stdout[:4000]}"
        if stderr.strip():
            summary += f"\n\nstderr:\n{stderr[:4000]}"
        self.messages.append({"role": "user", "content": f"[Tool result]\n{summary}"})
        self._save_current_chat()
        self._scroll_down()
        self._start_worker()

    def _deny_tool(self, card: ToolCard, command: str):
        self.messages.append(
            {"role": "user", "content": f"[User denied running this command: {command}]"})
        self._save_current_chat()
        self._start_worker()

    def _start_search(self, query: str):
        card = WebCard("search", query, self.mono)
        self.feed.insertWidget(self.feed.count() - 1, card)
        self._scroll_down()
        worker = SearchWorker(query)
        worker.result.connect(
            lambda results, err, c=card, q=query: self._search_result(c, q, results, err))
        self._cmd_worker = self._track(worker)
        worker.start()

    def _search_result(self, card: WebCard, query: str, results: list, err: str):
        if err:
            card.show_result(f"Error: {err}", ok=False)
            summary = f'Search failed for "{query}": {err}'
        elif not results:
            card.show_result("(no results)")
            summary = f'Search for "{query}" returned no results.'
        else:
            lines = [f"{i}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')}"
                    for i, r in enumerate(results, 1)]
            text = "\n\n".join(lines)
            card.show_result(text)
            summary = f'Search results for "{query}":\n\n{text[:3500]}'
        self.messages.append({"role": "user", "content": f"[Web search result]\n{summary}"})
        self._save_current_chat()
        self._start_worker()

    def _start_fetch(self, url: str):
        card = WebCard("fetch", url, self.mono)
        self.feed.insertWidget(self.feed.count() - 1, card)
        self._scroll_down()
        worker = FetchWorker(url)
        worker.result.connect(lambda text, err, c=card, u=url: self._fetch_result(c, u, text, err))
        self._cmd_worker = self._track(worker)
        worker.start()

    def _fetch_result(self, card: WebCard, url: str, text: str, err: str):
        if err:
            card.show_result(f"Error: {err}", ok=False)
            summary = f"Fetching {url} failed: {err}"
        else:
            card.show_result(text)
            summary = f"Content fetched from {url}:\n\n{text[:4000]}"
        self.messages.append({"role": "user", "content": f"[Web fetch result]\n{summary}"})
        self._save_current_chat()
        self._start_worker()

    def _offer_skill(self, name: str, content: str):
        card = SkillCard(name, content, self.mono)
        card.save_clicked.connect(
            lambda c=card, n=name, ct=content: self._save_skill_from_card(c, n, ct))
        card.discard_clicked.connect(lambda c=card, n=name: self._discard_skill_card(c, n))
        self.feed.insertWidget(self.feed.count() - 1, card)
        self._scroll_down()

    def _save_skill_from_card(self, card: SkillCard, name: str, content: str):
        self.skills.append({
            "id": uuid.uuid4().hex[:10], "name": name, "content": content,
            "created": time.time(), "updated": time.time(),
        })
        self._save_skills()
        self._reload_skill_list()
        self.messages.append({"role": "user", "content": f"[Skill saved: {name}]"})
        self._save_current_chat()
        self._start_worker()

    def _discard_skill_card(self, card: SkillCard, name: str):
        self.messages.append(
            {"role": "user", "content": f"[User discarded proposed skill: {name}]"})
        self._save_current_chat()
        self._start_worker()

    # -- chat history ------------------------------------------------------ #
    def _new_chat(self):
        if still_running(self.chat_worker):
            self.chat_worker.stop()
        while self.feed.count() > 2:                 # keep hint + stretch
            item = self.feed.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self.messages.clear()
        self.buffer = ""
        self.bubble = None
        self.tool_round = 0
        self.current_chat_id = None
        self.tools_check.setChecked(False)
        self.autorun_check.setChecked(False)
        self.hint.show()
        self.chat_list.clearSelection()

    def _save_current_chat(self):
        if not self.messages:
            return
        if not self.current_chat_id:
            self.current_chat_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        path = self.chats_dir / f"{self.current_chat_id}.json"
        is_new = not path.exists()
        created = time.time()
        existing_title = ""
        if path.exists():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
                created = prior.get("created", created)
                existing_title = prior.get("title", "")
            except Exception:                                      # noqa: BLE001
                pass
        # Keep whatever title the file already has — recomputing it on every save
        # silently reverted any rename the user made.
        if existing_title:
            title = existing_title
        else:
            first_user = next((m["content"] for m in self.messages if m["role"] == "user"),
                              "New chat")
            squashed = " ".join(first_user.split())
            # Stored long; the sidebar elides it to fit, so nothing is lost here.
            title = squashed[:120] or "New chat"
        data = {
            "id": self.current_chat_id, "title": title, "model": self.current_model,
            "created": created, "updated": time.time(), "messages": self.messages,
        }
        _atomic_write_json(path, data)
        if is_new:
            self._reload_chat_list()      # a row appeared; otherwise nothing moved

    def _reload_chat_list(self):
        self.chat_list.clear()
        files = sorted(self.chats_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:                                      # noqa: BLE001
                continue
            title = data.get("title") or "Untitled"
            chat_id = data.get("id") or f.stem
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, chat_id)
            item.setToolTip(title)          # elided in the row; full text on hover
            row = ListRow(chat_id, title)
            item.setSizeHint(QSize(row.sizeHint().width(), row.height()))
            self.chat_list.addItem(item)
            self.chat_list.setItemWidget(item, row)
            row.delete_requested.connect(self._delete_chat)
            if chat_id == self.current_chat_id:
                self.chat_list.setCurrentItem(item)
        self.chat_empty.setVisible(self.chat_list.count() == 0)

    def _on_chat_item_clicked(self, item: QListWidgetItem):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        if chat_id != self.current_chat_id:
            self._load_chat(chat_id)

    def _load_chat(self, chat_id: str):
        path = self.chats_dir / f"{chat_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            return

        if still_running(self.chat_worker):
            self.chat_worker.stop()
        while self.feed.count() > 2:
            item = self.feed.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        self.messages = data.get("messages", [])
        self.current_chat_id = data.get("id", chat_id)
        self.buffer = ""
        self.bubble = None
        self.tool_round = 0
        self.tools_check.setChecked(False)      # don't silently resume tool/auto-run for a
        self.autorun_check.setChecked(False)    # chat that was reopened later

        model = data.get("model")
        if model:
            idx = self.model_combo.findData(model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
                self.current_model = model

        if self.messages:
            self.hint.hide()
            for m in self.messages:
                role = m.get("role", "user")
                if role == "user":
                    b = self._add_bubble("user")
                else:
                    _, badge = model_label(self.current_model)
                    b = self._add_bubble("assistant", badge)
                b.set_text(m.get("content", ""))
        else:
            self.hint.show()

        self._scroll_down()
        self._reload_chat_list()

    def _rename_chat(self, chat_id: str):
        path = self.chats_dir / f"{chat_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            return
        new_title, ok = QInputDialog.getText(self, "Rename chat", "Title:",
                                             text=data.get("title", ""))
        if ok and new_title.strip():
            data["title"] = new_title.strip()
            _atomic_write_json(path, data)
            self._reload_chat_list()

    def _delete_chat(self, chat_id: str):
        title = chat_id
        path = self.chats_dir / f"{chat_id}.json"
        if path.exists():
            try:
                title = json.loads(path.read_text(encoding="utf-8")).get("title", chat_id)
            except Exception:                                      # noqa: BLE001
                pass
        short = (title[:60] + "…") if len(title) > 60 else title
        reply = QMessageBox.question(
            self, "Delete chat",
            f"Permanently delete “{short}”?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._remove_chat(chat_id)

    def _remove_chat(self, chat_id: str):
        path = self.chats_dir / f"{chat_id}.json"
        if path.exists():
            path.unlink()
        if chat_id == self.current_chat_id:
            self._new_chat()
        self._reload_chat_list()

    def _remove_all_chats(self):
        for f in self.chats_dir.glob("*.json"):
            try:
                f.unlink()
            except OSError:
                pass
        self._new_chat()
        self._reload_chat_list()

    def _chat_context_menu(self, pos):
        item = self.chat_list.itemAt(pos)
        if not item:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        rename_act = menu.addAction("Rename")
        delete_act = menu.addAction("Delete")
        action = menu.exec(self.chat_list.mapToGlobal(pos))
        if action == rename_act:
            self._rename_chat(chat_id)
        elif action == delete_act:
            self._delete_chat(chat_id)

    # -- skills -------------------------------------------------------------#
    def _read_skills_file(self) -> list:
        if not SKILLS_PATH.exists():
            return []
        try:
            return json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            return []

    def _save_skills(self):
        _atomic_write_json(SKILLS_PATH, self.skills)

    def _reload_skill_list(self):
        checked_ids = {
            self.skill_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.skill_list.count())
            if self.skill_list.item(i).checkState() == Qt.CheckState.Checked
        }
        self.skill_list.clear()
        for sk in self.skills:
            item = QListWidgetItem(sk["name"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if sk["id"] in checked_ids
                               else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, sk["id"])
            self.skill_list.addItem(item)
        self.skill_empty.setVisible(self.skill_list.count() == 0)

    def _active_skills_text(self) -> str:
        active_ids = {
            self.skill_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.skill_list.count())
            if self.skill_list.item(i).checkState() == Qt.CheckState.Checked
        }
        if not active_ids:
            return ""
        chosen = [sk for sk in self.skills if sk["id"] in active_ids]
        parts = [f"### Skill: {sk['name']}\n{sk['content']}" for sk in chosen]
        return "Apply the following user-selected skill(s):\n\n" + "\n\n".join(parts)

    def _new_skill(self):
        dlg = SkillEditDialog(self, self.mono, self._qss())
        if dlg.exec():
            name, content = dlg.values()
            if name and content:
                self.skills.append({
                    "id": uuid.uuid4().hex[:10], "name": name, "content": content,
                    "created": time.time(), "updated": time.time(),
                })
                self._save_skills()
                self._reload_skill_list()

    def _edit_skill(self, skill_id: str):
        sk = next((s for s in self.skills if s["id"] == skill_id), None)
        if not sk:
            return
        dlg = SkillEditDialog(self, self.mono, self._qss(), sk["name"], sk["content"])
        if dlg.exec():
            name, content = dlg.values()
            if name and content:
                sk["name"], sk["content"], sk["updated"] = name, content, time.time()
                self._save_skills()
                self._reload_skill_list()

    def _delete_skill(self, skill_id: str):
        sk = next((s for s in self.skills if s["id"] == skill_id), None)
        if not sk:
            return
        reply = QMessageBox.question(
            self, "Delete skill", f'Delete the skill "{sk["name"]}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._remove_skill(skill_id)

    def _remove_skill(self, skill_id: str):
        self.skills = [s for s in self.skills if s["id"] != skill_id]
        self._save_skills()
        self._reload_skill_list()

    def _on_skill_item_double_clicked(self, item: QListWidgetItem):
        self._edit_skill(item.data(Qt.ItemDataRole.UserRole))

    def _skill_context_menu(self, pos):
        item = self.skill_list.itemAt(pos)
        if not item:
            return
        skill_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        edit_act = menu.addAction("Edit")
        delete_act = menu.addAction("Delete")
        action = menu.exec(self.skill_list.mapToGlobal(pos))
        if action == edit_act:
            self._edit_skill(skill_id)
        elif action == delete_act:
            self._delete_skill(skill_id)

    # -- ollama ----------------------------------------------------------- #
    def _set_status(self, text: str, color: str):
        self.status.setText(text)
        self.status_dot.setStyleSheet(f"color:{color};")

    def _check_ollama(self):
        try:
            requests.get(TAGS_URL, timeout=2).raise_for_status()
            self._set_status("Connected", C["green"])
            self._populate_models()
            return
        except Exception:                                        # noqa: BLE001
            pass
        self._set_status("Starting Ollama…", C["amber"])
        exe = ollama_exe()
        if exe is not None:
            env = dict(os.environ,
                       OLLAMA_MODELS=str(models_dir()), OLLAMA_HOST=HOST)
            subprocess.Popen([str(exe), "serve"], cwd=str(exe.parent),
                             env=env, creationflags=NO_WINDOW)
            QTimer.singleShot(2500, self._recheck)
        else:
            self._set_status("Ollama not found", C["red"])

    def _recheck(self, tries: int = 12):
        try:
            requests.get(TAGS_URL, timeout=2).raise_for_status()
            self._set_status("Connected", C["green"])
            self._populate_models()
        except Exception:                                        # noqa: BLE001
            if tries:
                QTimer.singleShot(1500, lambda: self._recheck(tries - 1))
            else:
                self._set_status("Ollama offline", C["red"])

    def _populate_models(self):
        """Fill the model picker from /api/tags, preferring known display names."""
        try:
            names = [m["name"].split(":")[0]
                    for m in requests.get(TAGS_URL, timeout=3).json().get("models", [])]
            reachable = True
        except Exception:                                        # noqa: BLE001
            names, reachable = [], False    # server down; leave the picker alone

        if not names:
            if reachable:
                # Server is up but has nothing installed: make that state explicit
                # instead of leaving a stale or empty picker with a model still set.
                self.model_combo.blockSignals(True)
                self.model_combo.clear()
                self.model_combo.addItem("No models installed", userData="")
                self.model_combo.setCurrentIndex(0)
                self.model_combo.blockSignals(False)
                self.current_model = ""
            return

        names.sort(key=lambda n: model_label(n)[0].lower())

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for n in names:
            display, _ = model_label(n)
            self.model_combo.addItem(display, userData=n)
        idx = self.model_combo.findData(self.current_model)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.current_model = self.model_combo.currentData()
        self.model_combo.blockSignals(False)

    def _on_model_changed(self, _index: int):
        model = self.model_combo.currentData()
        if not model or model == self.current_model:
            return
        self.current_model = model
        display, _ = model_label(model)
        note = QLabel(f"— switched to {display}. Ollama may take a moment to "
                      f"swap it into VRAM on the next message. —")
        note.setObjectName("hint")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.insertWidget(self.feed.count() - 1, note)
        self._scroll_down()

    # -- gpu -------------------------------------------------------------- #
    def _on_gpu(self, s: dict):
        if "error" in s:
            self.gpu_name.setText("GPU unavailable")
            self.detail.setText(f'<span style="color:{C["red"]}">{_html.escape(s["error"])}</span>')
            return

        self.gpu_name.setText(s["name"])
        mem_pct = s["mem_used"] / s["mem_total"] * 100 if s["mem_total"] else 0
        pow_pct = s["power"] / s["power_max"] * 100 if s["power_max"] else 0

        self.g_util.set_value(s["util"], f"{s['clock']:.0f} MHz")
        self.g_mem.set_value(mem_pct, f"{s['mem_used'] / 1024:.1f} / {s['mem_total'] / 1024:.1f} GB")
        self.g_temp.set_value(s["temp"], "")
        self.g_temp.color = QColor(C["green"] if s["temp"] < 70 else
                                   C["amber"] if s["temp"] < 83 else C["red"])
        self.g_pow.set_value(pow_pct, f"{s['power']:.0f} / {s['power_max']:.0f} W"
                             if s["power_max"] else f"{s['power']:.0f} W")

        self.spark_util.push(s["util"])
        self.spark_mem.push(mem_pct)

        rows = [
            ("Memory", f"{s['mem_used']:.0f} / {s['mem_total']:.0f} MiB  ({mem_pct:.1f}%)"),
            ("Memory bus", f"{s['mem_util']:.0f}% busy"),
            ("SM clock", f"{s['clock']:.0f} MHz  (max {s['clock_max']:.0f})"),
            ("Board power", f"{s['power']:.1f} W" +
             (f" / {s['power_max']:.0f} W" if s["power_max"] else "")),
            ("Temperature", f"{s['temp']:.0f} °C"),
        ]
        html = "".join(
            f'<tr><td style="color:{C["dim"]};padding:3px 26px 3px 0;">{k}</td>'
            f'<td style="color:{C["text"]};font-family:{self.mono};">{v}</td></tr>'
            for k, v in rows)

        if s["procs"]:
            html += (f'<tr><td colspan="2" style="color:{C["faint"]};padding-top:12px;">'
                     f'GPU PROCESSES</td></tr>')
            for pid, nm, mem in s["procs"]:
                mark = C["green"] if "ollama" in nm.lower() else C["dim"]
                shown = f"{mem} MiB" if mem.isdigit() else "—"
                html += (f'<tr><td style="color:{mark};padding:3px 26px 3px 0;">{_html.escape(nm)}'
                         f'</td><td style="color:{C["dim"]};font-family:{self.mono};">'
                         f'pid {pid} · {shown}</td></tr>')
        self.detail.setText(f"<table cellspacing='0'>{html}</table>")

    # -- shutdown --------------------------------------------------------- #
    def closeEvent(self, ev):
        if still_running(self.chat_worker):
            self.chat_worker.stop()
        for worker in list(self._workers):
            if still_running(worker):
                stop = getattr(worker, "stop", None)
                if stop:
                    stop()
                worker.wait(2000)
        self.gpu.stop()
        self.gpu.wait(1500)
        super().closeEvent(ev)

    # -- style ------------------------------------------------------------ #
    def _qss(self) -> str:
        """Type scale: 11 / 12 / 13 / 14 / 15 / 20.  Weights: 400 / 500 / 600.
        Spacing steps: 4 / 6 / 8 / 12 / 16 / 20 / 24.  Radii: 6 / 8 / 10 / 14."""
        return f"""
        QMainWindow, QWidget {{ background:{C['bg']}; color:{C['text']};
                                font-family:'Segoe UI'; font-size:14px; }}
        /* Without this, every QLabel paints the window colour as an opaque bar
           over whatever panel it sits on. Rules with an ID selector still win. */
        QLabel {{ background:transparent; }}
        QToolTip {{ background:{C['panel_hi']}; color:{C['text']};
                    border:1px solid {C['line_str']}; border-radius:6px; padding:6px 9px;
                    font-size:12px; }}

        /* ---- header ---- */
        #head {{ background:{C['panel']}; border-bottom:1px solid {C['line']}; }}
        #brandMark {{ background:transparent; }}
        #title {{ font-size:15px; font-weight:600; letter-spacing:0.2px; }}
        #statusDot {{ color:{C['faint']}; font-size:9px; }}
        #status {{ color:{C['dim']}; font-size:12px; }}

        #modelPicker {{ background:{C['panel_hi']}; color:{C['text']};
                        border:1px solid {C['line']}; border-radius:8px;
                        padding:7px 12px; font-size:12px; font-weight:500; }}
        #modelPicker:hover {{ background:{C['overlay']}; border-color:{C['line_str']}; }}
        #modelPicker::drop-down {{ border:none; width:20px; }}
        #computePicker {{ background:{C['panel_hi']}; color:{C['dim']};
                          border:1px solid {C['line']}; border-radius:8px;
                          padding:7px 10px; font-size:12px; font-weight:500; }}
        #computePicker:hover {{ background:{C['overlay']}; border-color:{C['line_str']};
                                color:{C['text']}; }}
        #computePicker::drop-down {{ border:none; width:20px; }}
        #computePicker QAbstractItemView {{ background:{C['panel_hi']}; color:{C['text']};
                        border:1px solid {C['line_str']}; border-radius:8px;
                        selection-background-color:{C['accent']}; selection-color:#0b1220;
                        outline:none; padding:4px; }}
        #modelPicker QAbstractItemView {{ background:{C['panel_hi']}; color:{C['text']};
                        border:1px solid {C['line_str']}; border-radius:8px;
                        selection-background-color:{C['accent']}; selection-color:#0b1220;
                        outline:none; padding:4px; }}

        /* ---- tabs ---- */
        /* segmented nav, living in the header instead of a second chrome band */
        #navBtn {{ background:transparent; color:{C['faint']}; border:none;
                   border-radius:8px; padding:7px 15px; font-size:13px; font-weight:500; }}
        #navBtn:hover {{ color:{C['text']}; background:{C['panel_hi']}; }}
        #navBtn:checked {{ color:{C['text']}; background:{C['overlay']}; font-weight:600; }}

        #pageTitle {{ font-size:20px; font-weight:600; }}
        #aboutName {{ font-size:24px; font-weight:600; letter-spacing:0.2px; }}
        #wizStep {{ color:{C['accent']}; font-size:14px; font-weight:500; }}
        #wizWhere {{ color:{C['dim']}; font-size:12px; background:{C['panel_hi']};
                      border:1px solid {C['line']}; border-radius:8px; padding:9px 12px; }}
        #wizDetail {{ color:{C['dim']}; font-size:13px; line-height:150%; }}
        #wizConsole {{ background:{C['bg']}; border:1px solid {C['line']};
                       border-radius:8px; color:{C['faint']};
                       font-family:'{self.mono}'; font-size:{CODE_PX_SM}px;
                       padding:10px 12px; }}
        #wizBar {{ background:{C['panel_hi']}; border:none; border-radius:3px; }}
        #wizBar::chunk {{ background:{C['accent']}; border-radius:3px; }}
        #linkBtn {{ background:{C['panel_hi']}; color:{C['accent']};
                    border:1px solid {C['line_str']}; border-radius:8px;
                    padding:8px 16px; font-size:13px; font-weight:500; }}
        #linkBtn:hover {{ background:{C['overlay']}; color:{C['accent_hi']};
                          border-color:{C['accent']}; }}
        #pageSub {{ color:{C['faint']}; font-size:13px; }}
        #promptBox {{ background:{C['bg']}; border:1px solid {C['line']}; border-radius:8px;
                      color:{C['code']}; font-family:'{self.mono}'; font-size:{CODE_PX}px;
                      padding:10px 12px; selection-background-color:{C['accent']}; }}
        #promptBox:focus {{ border-color:{C['accent']}; }}
        #importStatus {{ color:{C['dim']}; font-size:12px; background:{C['panel_hi']};
                         border:1px solid {C['line']}; border-radius:8px; padding:9px 12px; }}

        /* list rows carrying their own × */
        #rowTitle {{ background:transparent; font-size:13px; }}
        #rowSub {{ background:transparent; color:{C['faint']}; font-size:11px; }}
        #rowClose {{ background:transparent; color:{C['faint']}; border:none;
                     border-radius:10px; font-size:12px; padding:0; }}
        #rowClose:hover {{ background:{C['red']}; color:#1a0b0e; }}
        QListWidget#modelList {{ background:transparent; border:none; outline:none; }}
        QListWidget#modelList::item {{ border-radius:8px; }}
        QListWidget#modelList::item:hover {{ background:{C['panel_hi']}; }}
        QListWidget#modelList::item:selected {{ background:{C['overlay']}; }}

        /* ---- chat feed ---- */
        #chatArea {{ background:{C['bg']}; }}
        #emptyTitle {{ color:{C['text']}; font-size:20px; font-weight:600; }}
        #emptySub {{ color:{C['faint']}; font-size:13px; }}

        #who_user, #who_bot {{ font-size:11px; font-weight:600; }}
        #who_user {{ color:{C['dim']}; }}
        #who_bot  {{ color:{C['dim']}; }}
        #bubble_user {{ background:{C['accent_soft']}; border:1px solid #26355a;
                        border-radius:14px; }}
        #bubble_bot  {{ background:{C['panel']}; border:1px solid {C['line']};
                        border-radius:14px; }}
        #proseUser, #proseBot {{ background:transparent; color:{C['text']};
                                 font-size:{BODY_PX}px; line-height:158%; }}
        #think {{ color:{C['faint']}; font-size:12px; background:{C['panel_hi']};
                  border-left:2px solid {C['line_str']}; padding:10px 14px;
                  border-radius:8px; max-width:760px; }}
        #thinkBtn {{ background:transparent; border:none; color:{C['faint']};
                     font-size:11px; text-align:left; padding:0; }}
        #thinkBtn:hover {{ color:{C['dim']}; }}
        #meta {{ color:{C['faint']}; font-size:11px; }}

        /* ---- code blocks ---- */
        #codeBlock {{ background:{C['panel_hi']}; border:1px solid {C['line']};
                      border-radius:10px; }}
        #codeHeader {{ background:transparent; border-bottom:1px solid {C['line']}; }}
        #codeLang {{ color:{C['faint']}; font-size:11px; font-weight:500; }}
        #copyBtn {{ background:transparent; color:{C['faint']}; border:none;
                    border-radius:6px; padding:3px 9px; font-size:11px; font-weight:500; }}
        #copyBtn:hover {{ color:{C['text']}; background:{C['overlay']}; }}
        /* font-family must be restated here: the QWidget rule above sets
           'Segoe UI' and would otherwise beat the QFont set in code, rendering
           every code block in a proportional face. */
        #codeBody {{ background:transparent; border:none; color:{C['code']};
                     font-family:'{self.mono}'; font-size:{CODE_PX}px;
                     padding:8px 12px; selection-background-color:{C['accent']}; }}

        /* ---- action cards: subtle surface + one accent edge, not alert boxes ---- */
        #toolCard, #skillCard, #webCard, #toolCardAuto {{
            background:{C['panel_hi']}; border:1px solid {C['line']}; border-radius:12px; }}
        #toolCard {{ border-left:3px solid {C['amber']}; }}
        #skillCard {{ border-left:3px solid {C['green']}; }}
        #webCard {{ border-left:3px solid {C['violet']}; }}
        #toolCardAuto {{ background:{C['red_soft']}; border:1px solid {C['red']};
                         border-left:3px solid {C['red']}; }}
        #toolHeader, #skillHeader, #webHeader {{ font-size:12px; font-weight:600; }}
        #toolHeader {{ color:{C['amber']}; }}
        #skillHeader {{ color:{C['green']}; }}
        #webHeader {{ color:{C['violet']}; }}
        #toolHeaderAuto {{ color:{C['red']}; font-size:12px; font-weight:600; }}
        #toolStatus {{ color:{C['faint']}; font-size:11px; }}
        #skillName {{ color:{C['text']}; font-size:14px; font-weight:600; }}
        #webQuery {{ color:{C['dim']}; font-size:12px; }}
        #toolCmd {{ background:{C['bg']}; border:1px solid {C['line']}; border-radius:8px;
                    color:{C['code']}; font-family:'{self.mono}'; font-size:{CODE_PX}px;
                    padding:8px 10px; }}
        #toolOutput {{ background:{C['bg']}; border:1px solid {C['line']}; border-radius:8px;
                       color:{C['dim']}; font-family:'{self.mono}';
                       font-size:{CODE_PX_SM}px; padding:8px 10px; }}

        /* ---- sidebars ---- */
        #sidebar {{ background:{C['panel']}; border-right:1px solid {C['line']}; }}
        #sidebarRight {{ background:{C['panel']}; border-left:1px solid {C['line']}; }}
        #sidebarTitle {{ color:{C['faint']}; font-size:11px; font-weight:600;
                         padding:4px 6px 0 6px; }}
        #sidebarHint {{ color:{C['faint']}; font-size:11px; padding:0 6px; }}
        #listEmpty {{ color:{C['faint']}; font-size:12px; padding:4px 6px 8px 6px; }}
        #newBtn {{ background:{C['panel_hi']}; color:{C['text']}; border:1px solid {C['line_str']};
                   border-radius:8px; padding:9px 12px; text-align:center;
                   font-size:13px; font-weight:500; }}
        #newBtn:hover {{ background:{C['overlay']}; }}
        #ghostSm {{ background:transparent; color:{C['faint']};
                    border:1px solid {C['line']}; border-radius:7px;
                    padding:6px 8px; font-size:11px; font-weight:500; }}
        #ghostSm:hover {{ color:{C['red']}; border-color:{C['red']};
                          background:{C['red_soft']}; }}
        #ghostSm:disabled {{ color:#3f4756; border-color:{C['line']};
                             background:transparent; }}

        QListWidget#chatList, QListWidget#skillList {{ background:transparent; border:none;
                                                       outline:none; }}
        QListWidget#chatList::item, QListWidget#skillList::item {{
            color:{C['dim']}; padding:8px 10px; border-radius:8px; }}
        QListWidget#chatList::item:hover, QListWidget#skillList::item:hover {{
            background:{C['panel_hi']}; color:{C['text']}; }}
        QListWidget#chatList::item:selected {{ background:{C['overlay']}; color:{C['text']}; }}
        QListWidget#skillList::item:selected {{ background:{C['panel_hi']}; color:{C['text']}; }}
        QListWidget#skillList::indicator {{ width:14px; height:14px; border-radius:4px;
                                            border:1px solid {C['line_str']};
                                            background:{C['bg']}; margin-right:2px; }}
        QListWidget#skillList::indicator:checked {{ background:{C['green']};
                                                    border-color:{C['green']}; }}

        /* ---- composer ---- */
        #composer {{ background:{C['panel']}; border-top:1px solid {C['line']}; }}
        #inputShell {{ background:{C['panel_hi']}; border:1px solid {C['line_str']};
                       border-radius:14px; }}
        #input {{ background:transparent; border:none; padding:8px 10px;
                  color:{C['text']}; font-size:14px;
                  selection-background-color:{C['accent']}; }}
        #composerHint {{ color:{C['faint']}; font-size:11px; }}

        /* toggle pills — off reads as quiet chrome, on reads as clearly armed */
        #pillSkills:checked {{ background:{C['green_soft']}; color:{C['green']};
                               border-color:{C['green']}; }}
        #pillTools, #pillAutorun, #pillWeb, #pillSkills {{
            background:transparent; color:{C['faint']}; border:1px solid {C['line_str']};
            border-radius:14px; padding:5px 14px; font-size:12px; font-weight:500; }}
        #pillTools:hover, #pillAutorun:hover, #pillWeb:hover, #pillSkills:hover {{
            color:{C['text']}; background:{C['overlay']}; }}
        #pillTools:checked {{ background:{C['amber_soft']}; color:{C['amber']};
                              border-color:{C['amber']}; }}
        #pillAutorun:checked {{ background:{C['red_soft']}; color:{C['red']};
                                border-color:{C['red']}; }}
        #pillWeb:checked {{ background:{C['violet_soft']}; color:{C['violet']};
                            border-color:{C['violet']}; }}
        #pillAutorun:disabled {{ color:#4a5464; border-color:{C['line']}; }}
        #autorunWarn {{ color:{C['red']}; font-size:12px; font-weight:500;
                        background:{C['red_soft']}; border:1px solid {C['red']};
                        border-radius:8px; padding:8px 12px; }}

        /* ---- buttons ---- */
        QPushButton#primary {{ background:{C['accent']}; color:#0b1220; border:none;
                               border-radius:10px; padding:8px 20px;
                               font-size:13px; font-weight:600; }}
        QPushButton#primary:hover {{ background:{C['accent_hi']}; }}
        QPushButton#primary:disabled {{ background:{C['overlay']}; color:{C['faint']}; }}
        QPushButton#danger {{ background:{C['red']}; color:#1a0b0e; border:none;
                              border-radius:10px; padding:8px 20px;
                              font-size:13px; font-weight:600; }}
        QPushButton#ghost {{ background:transparent; color:{C['dim']};
                             border:1px solid {C['line_str']}; border-radius:10px;
                             padding:8px 16px; font-size:13px; font-weight:500; }}
        QPushButton#ghost:hover {{ color:{C['text']}; background:{C['overlay']}; }}
        QPushButton#ghost:disabled {{ color:{C['faint']}; border-color:{C['line']}; }}

        /* ---- gpu tab ---- */
        #card {{ background:{C['panel']}; border:1px solid {C['line']}; border-radius:14px; }}
        #cardTitle {{ color:{C['faint']}; font-size:11px; font-weight:600; }}
        #gpuName {{ font-size:15px; font-weight:600; }}
        #detail {{ font-size:13px; }}

        /* ---- dialogs ---- */
        QDialog {{ background:{C['bg']}; }}
        QDialog QLabel {{ color:{C['dim']}; font-size:12px; }}
        QLineEdit#input {{ background:{C['panel_hi']}; border:1px solid {C['line_str']};
                           border-radius:8px; padding:9px 12px; color:{C['text']};
                           font-size:13px; }}
        QLineEdit#input:focus {{ border-color:{C['accent']}; }}
        QMenu {{ background:{C['panel_hi']}; border:1px solid {C['line_str']};
                 border-radius:8px; padding:4px; }}
        QMenu::item {{ padding:7px 22px 7px 14px; border-radius:6px;
                       color:{C['dim']}; font-size:13px; }}
        QMenu::item:selected {{ background:{C['overlay']}; color:{C['text']}; }}
        QMessageBox {{ background:{C['bg']}; }}
        QMessageBox QLabel {{ color:{C['text']}; font-size:13px; }}
        QMessageBox QPushButton {{ background:{C['panel_hi']}; color:{C['text']};
                                   border:1px solid {C['line_str']}; border-radius:8px;
                                   padding:7px 18px; font-size:13px; min-width:64px; }}
        QMessageBox QPushButton:hover {{ background:{C['overlay']}; }}

        /* ---- scrollbars ---- */
        QScrollBar:vertical {{ background:transparent; width:11px; margin:2px; }}
        QScrollBar::handle:vertical {{ background:{C['line_str']}; border-radius:5px;
                                       min-height:36px; }}
        QScrollBar::handle:vertical:hover {{ background:{C['faint']}; }}
        QScrollBar:horizontal {{ background:transparent; height:11px; margin:2px; }}
        QScrollBar::handle:horizontal {{ background:{C['line_str']}; border-radius:5px;
                                         min-width:36px; }}
        QScrollBar::handle:horizontal:hover {{ background:{C['faint']}; }}
        QScrollBar::handle:vertical:disabled, QScrollBar::handle:horizontal:disabled {{
            background:transparent; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background:none; }}
        """


def main():
    app = QApplication(sys.argv)
    install_crash_guard()
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(app_icon())
    app.setFont(QFont("Segoe UI", 10))

    # First run (or any run with no Ollama yet): show the setup check before the
    # main window, so a fresh machine gets a guided start rather than a dead app.
    settings = {}
    if SETTINGS_PATH.exists():
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            settings = {}
    if ollama_exe() is None or not settings.get("setup_done"):
        probe = CriGent.__new__(CriGent)                # only need its stylesheet
        probe.mono = mono_family()
        dlg = SetupDialog(qss=CriGent._qss(probe), mono=probe.mono)
        dlg.setWindowIcon(app_icon())
        dlg.exec()
        if dlg.ok:
            settings["setup_done"] = True
            try:
                _atomic_write_json(SETTINGS_PATH, settings)
            except OSError:
                pass

    win = CriGent()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
