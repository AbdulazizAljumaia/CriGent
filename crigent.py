"""CriGent — a local AI agent: chat over local Ollama models, with tool use, web
search, reusable skills, a live GPU dashboard and a built-in model manager."""

import html as _html
import ctypes
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
from datetime import date, timedelta
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
                         QPainterPath, QPen, QPixmap, QTextCursor)
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
    global USAGE_PATH, CRASHES_PATH
    path.mkdir(parents=True, exist_ok=True)
    LOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCATION_FILE.write_text(json.dumps({"data_dir": str(path)}, indent=2),
                             encoding="utf-8")
    ROOT = path
    CHATS_DIR = ROOT / "chats"
    SKILLS_PATH = ROOT / "skills.json"
    PROMPTS_PATH = ROOT / "prompts.json"
    SETTINGS_PATH = ROOT / "settings.json"
    USAGE_PATH = ROOT / "usage.json"
    CRASHES_PATH = ROOT / "crashes.json"
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
# The plain completion endpoint. Used only as a last resort when /api/chat's
# parser rejects a model's output — it returns raw text and runs no chat-message
# parsing, so the format the model drifted from cannot reject the reply.
GEN_URL = f"http://{HOST}/api/generate"
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
MAX_TOOL_ROUNDS = 12    # a checklist of any size needs room to finish
# Sent whenever any capability is on. Without it, models fall back on their
# training and answer "I can't browse the web / run commands", refusing work the
# app is perfectly able to do.
AGENT_SYSTEM_PROMPT = (
    "You are running inside CriGent, a desktop application on the user's own Windows "
    "computer. It gives you real, working abilities, described below.\n\n"
    "These are not hypothetical and this is not role-play. When you write one of the "
    "fenced blocks described below, CriGent genuinely performs that action and gives you "
    "the result in the next message. Set aside any assumption from your training that you "
    "cannot browse the internet, run commands, or interact with a computer — in this "
    "application, through these blocks, you can.\n\n"
    "How to work here:\n"
    "- Act, do not offer. Never end your turn by proposing to use an ability and waiting "
    "for the user to say yes. If you are about to write \"I can search for that if you "
    "like\", \"let me know and I'll look it up\", or \"if that sounds helpful\", then do it "
    "instead, in this same reply.\n"
    "- Never say you are unable to search the web, fetch a page, run a command, or inspect "
    "this computer. If an ability below covers what is needed, use it. Saying you cannot is "
    "simply wrong here, and unhelpful.\n"
    "- You genuinely cannot recall live information such as current prices, stock, product "
    "listings or URLs from memory — so look them up rather than apologising for not knowing "
    "them.\n"
    "- Do not ask permission. The app handles approval where it is needed.\n"
    "- Use at most one block per reply, then stop and wait. Never invent, guess or "
    "pre-write the result — it will be given to you, and you continue from there.\n"
    "- If none of these abilities are relevant, simply answer normally.\n\n"
    "Example of what NOT to do:\n"
    "  \"I'm unable to pull live product pages. I could search for you — just let me know "
    "which brand you prefer and I'll run the search.\"\n\n"
    "Do this instead:\n"
    "  \"Looking that up now.\n"
    "  ```search\n"
    "  Kingston Fury 32GB DDR5 4800 laptop memory price\n"
    "  ```\"\n\n"
    "## Finish the whole job\n\n"
    "When a request needs more than one step, open your first reply with a "
    "<tasks> checklist — one line per step, each starting ☐. Then work through "
    "it. In every following reply, repeat the checklist first, updated: ✓ on "
    "each step that succeeded, ✗ with a one-line reason on any step that "
    "genuinely cannot be done, ☐ on what remains.\n\n"
    "Do not stop while any ☐ remains and your abilities can still make "
    "progress — write the next action block instead of ending your turn. A "
    "reply that ends with unticked boxes and no action block is a failure "
    "unless the user has told you to stop.\n\n"
    "When every step is ✓ or ✗, close with a short conclusion: what was done, "
    "what was not and why, and anything the user should do next."
)

# Sent on its own toggle rather than as part of the agent preamble, because the
# layout applies to every reply — an ordinary chat with no tools switched on
# still wants its reasoning collapsed and its code in a container.
FORMAT_SYSTEM_PROMPT = (
    "## Structuring your reply\n\n"
    "Wrap each kind of content in its own tag so it is displayed properly. Use "
    "only these, and always close them:\n\n"
    "<reasoning>Your thinking. Shown collapsed, so work through the problem "
    "here rather than in the answer. One thought per line.</reasoning>\n"
    "<instructions>Steps the user should follow, in order.</instructions>\n"
    "<code lang=\"python\">source code, with the language named</code>\n"
    "<math>Equations and derivations.</math>\n"
    "<tasks>The checklist for a multi-step job. One task per line, each "
    "starting with ☐ (to do), ✓ (done) or ✗ (cannot be done — say why on the "
    "same line).</tasks>\n"
    "<text>Ordinary prose. Optional — untagged text is treated as prose.</text>\n\n"
    "Put your reasoning in <reasoning> and your conclusion outside it, so the "
    "user sees the answer first and can open the thinking if they want it. Do "
    "not nest tags.\n\n"
    "**Close <reasoning> before you do anything else.** An unclosed tag makes "
    "the rest of your reply count as thinking, and thinking is never acted on.\n\n"
    "Action blocks (```run, ```search, ```fetch, ```skill) are fenced blocks, "
    "not tags. Put them at the very end of the reply, outside every tag — one "
    "inside <reasoning> is ignored and you will never be sent its output.\n\n"
    "Working belongs in a tag, not loose in the sentence. An equation, a "
    "derivation or step-by-step arithmetic goes in <math>; a sequence of steps "
    "to follow goes in <instructions>; anything runnable goes in <code>.\n\n"
    "Short conversational replies need no tags at all — do not wrap a one-line "
    "answer in <text> just to have used a tag."
)

TOOL_SYSTEM_PROMPT = (
    "## Running commands\n\n"
    "You can run PowerShell commands on this machine. To do so, end your reply with a "
    "fenced block tagged `run` containing exactly one command:\n\n"
    "```run\nGet-ChildItem C:\\Users\n```\n\n"
    "Then stop. The user sees that exact command and approves or denies it, and you are "
    "given its output (or told it was denied) so you can carry on.\n\n"
    "- One command per block. Ask again on your next turn for the next step.\n"
    "- The block goes at the end of the reply, outside every tag. A command "
    "left inside <reasoning> is not run, and no output comes back.\n"
    "- Use `run` only for a command you want executed now. To show an example the user is "
    "not meant to run, use an ordinary ```powershell block instead.\n"
    "- Prefer reading over changing things, and say plainly what a command will do before "
    "proposing anything that deletes, overwrites or installs."
)

# Skill protocol: the model proposes a reusable skill with a ```skill fenced block;
# the app shows it to the user and only saves on explicit click, same as tool-run.
CHATS_DIR = ROOT / "chats"
SKILLS_PATH = ROOT / "skills.json"
SKILL_RE = re.compile(r"```skill\s*\n(.*?)```", re.S | re.I)
SKILL_SYSTEM_PROMPT = (
    "## Saving skills\n\n"
    "The user can keep reusable instructions, called skills, and switch them on for later "
    "conversations. You can write one for them.\n\n"
    "When they ask you to save, remember, or create a skill, end your reply with:\n\n"
    "```skill\nname: <short skill name>\n---\n<the instructions>\n```\n\n"
    "Then stop. They review it and decide whether to save it.\n\n"
    "**If the instructions contain a fenced code block, wrap the whole skill in four "
    "backticks** (````skill … ````) so the inner fences cannot be mistaken for the end.\n\n"
    "### Naming\n\n"
    "- Two to four words, describing the job, not the library: \"Web scraping\", not "
    "\"requests scraping\".\n"
    "- Sentence case, no trailing punctuation, no version numbers.\n"
    "- Check the spelling before you write it — the name is what the user picks from a "
    "list later, and a typo there is permanent until they rename it.\n\n"
    "### Writing the instructions\n\n"
    "- Address the assistant that will read it, in the second person: \"You write Python "
    "scrapers…\". It is a standing instruction, not a description of one.\n"
    "- Open with one line saying **when it applies**, so it is obvious whether to follow "
    "it for a given question.\n"
    "- Then the rules themselves: short `##` headings, and bullets or numbered steps under "
    "each. Concrete and testable — \"time out after 10s\" beats \"handle errors well\".\n"
    "- Include a worked template only if the user would otherwise retype it every time; "
    "put it in a fenced block with the language named, and keep it minimal.\n"
    "- Say what to avoid as well as what to do, where a mistake is likely.\n"
    "- Self-contained: it must still make sense weeks later, to someone who never saw this "
    "conversation. No \"as we discussed\", no references to this chat.\n"
    "- Aim for 10–40 lines. If it is longer, it is probably two skills.\n\n"
    "Only propose a skill when the user has actually asked for one to be saved — not for "
    "ordinary answers, explanations or code examples."
)


FENCE_RE = re.compile(r"^(`{3,})[ \t]*([A-Za-z0-9_+.#-]*)[ \t]*$")


def extract_block(text: str, tag: str):
    """Body of a ```<tag> fenced block, or None.

    A non-greedy `(.*?)``` ` regex stops at the first closing fence it meets.
    For a skill holding a code template that is the template's *own* opening
    fence, so the skill was silently saved cut off at that point. This walks
    the lines instead, stepping over nested fences, and also accepts an outer
    fence of four or more backticks — the standard way to wrap code in code.
    """
    lines = text.splitlines()
    start = ticks = None
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line.strip())
        if m and m.group(2).lower() == tag.lower():
            ticks, start = len(m.group(1)), i + 1
            break
    if start is None:
        return None

    inner = False                      # inside a nested fence
    for j in range(start, len(lines)):
        m = FENCE_RE.match(lines[j].strip())
        if not m:
            continue
        run, lang = len(m.group(1)), m.group(2)
        if not inner and not lang and run >= ticks:
            return "\n".join(lines[start:j])
        if not inner and lang:
            inner = True               # ```python … opens a nested block
        elif inner and not lang:
            inner = False              # …``` closes it
    # Unclosed: either still streaming, or the model forgot the last fence.
    # Returning what there is beats discarding the whole block.
    return "\n".join(lines[start:])


def _mask_examples(text: str) -> str:
    """Blank out fenced blocks and <code> tags, keeping every offset.

    So that a command *shown* as an example — ```` ```powershell <run>… ```` or
    a <code> block explaining the protocol — is not mistaken for one the model
    is asking to have executed.
    """
    chars = list(text)
    for m in re.finditer(r"(?s)<code(?:\s+[^<>]*)?>(.*?)(?:</code\s*>|$)", text, re.I):
        for i in range(m.start(1), m.end(1)):
            chars[i] = " "
    pos, in_fence = 0, False
    for line in text.splitlines(keepends=True):
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
        elif in_fence:
            for i in range(pos, pos + len(line)):
                chars[i] = " "
        pos += len(line)
    return "".join(chars)


def extract_action(text: str, tag: str):
    """A run/search/fetch/skill block in either form the models actually emit.

    The protocol asks for a ```run fence. But the same reply also teaches the
    model <reasoning>, <code> and <tasks> tags, and it generalises: it writes
    <run>whoami</run> and then stops, waiting for output. Nothing matched the
    fence, so no command was ever offered, and the conversation deadlocked
    until the user typed "carry on" — repeatedly, to no effect.

    Accepting both is the fix. The fenced form is preferred when both appear.
    """
    body = extract_block(text, tag)
    if body is not None and body.strip():
        return body
    masked = _mask_examples(text)
    m = re.compile(rf"<{tag}(?:\s+[^<>]*)?>", re.I).search(masked)
    if not m:
        return body
    close = re.compile(rf"</{tag}\s*>", re.I).search(masked, m.end())
    inner = text[m.end():close.start()] if close else text[m.end():]
    # Unclosed and running to the end of the reply: take the first line only,
    # rather than feeding trailing prose to PowerShell.
    if not close and tag == "run":
        inner = inner.strip().split("\n")[0]
    return inner if inner.strip() else body


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


MAX_CRASHES = 200        # newest kept; the file must not grow without bound


def crash_logging_enabled() -> bool:
    """Read the switch from disk rather than a live object: the crash handler
    runs when things are already going wrong, so it must not depend on the
    window still being in one piece."""
    try:
        return bool(json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                    .get("log_crashes", True))
    except Exception:                                              # noqa: BLE001
        return True


def load_crashes() -> list:
    try:
        data = json.loads(CRASHES_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:                                              # noqa: BLE001
        return []


def record_crash(kind: str, message: str, trace: str) -> None:
    try:
        crashes = load_crashes()
        crashes.insert(0, {
            "id": uuid.uuid4().hex[:10],
            "at": time.time(),
            "kind": kind,
            "message": message.strip()[:400],
            "trace": trace,
        })
        _atomic_write_json(CRASHES_PATH, crashes[:MAX_CRASHES])
    except Exception:                                              # noqa: BLE001
        pass                       # a failure to log must never mask the crash


def load_usage() -> dict:
    """Token history, bucketed by calendar day.

    Per-day totals rather than per-reply records: a year of heavy use is a few
    hundred small entries, so the file never needs pruning and reading it for
    the Usage page stays instant.
    """
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:                                              # noqa: BLE001
        return {}


def usage_totals(usage: dict, days: int | None = None, today: date | None = None):
    """Sum the buckets covering the last `days` days (None = everything)."""
    today = today or date.today()
    out = {"in": 0, "out": 0, "replies": 0, "seconds": 0.0}
    for key, rec in usage.items():
        if days is not None:
            try:
                day = date.fromisoformat(key)
            except ValueError:
                continue
            if not 0 <= (today - day).days < days:
                continue
        out["in"] += rec.get("in", 0)
        out["out"] += rec.get("out", 0)
        out["replies"] += rec.get("replies", 0)
        out["seconds"] += rec.get("seconds", 0.0)
    out["total"] = out["in"] + out["out"]
    return out


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
    "## Searching the web\n\n"
    "You have live web access here. You are not limited to your training data, and you do "
    "not need to speculate about current information or tell the user your knowledge has a "
    "cut-off — look it up instead.\n\n"
    "To search, end your reply with:\n\n"
    "```search\nyour query here\n```\n\n"
    "To read a particular page:\n\n"
    "```fetch\nhttps://example.com/page\n```\n\n"
    "Then stop and wait. Results are fetched automatically, shown to the user, and given "
    "back to you — no approval step is involved.\n\n"
    "- Search first to find sources, then fetch a promising result when you need detail.\n"
    "- Reach for this whenever the answer depends on current facts, specific documentation, "
    "version numbers, prices, news, or anything you are not confident about.\n"
    "- Cite what you found, and say so if the sources disagree or look unreliable."
)

PROMPTS_PATH = ROOT / "prompts.json"
SETTINGS_PATH = ROOT / "settings.json"
USAGE_PATH = ROOT / "usage.json"
CRASHES_PATH = ROOT / "crashes.json"
MODELS_DIR = ROOT / "models"

# Nav order. Pages are looked up by name everywhere else: inserting one here
# used to silently shift the hard-coded indices that drive per-page refreshes.
PAGES = ("Chat", "Track", "Usage", "Prompts", "Skills", "Crashes", "Models", "About")


def page_index(name: str) -> int:
    return PAGES.index(name)

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
    "format": FORMAT_SYSTEM_PROMPT,
    "agent": AGENT_SYSTEM_PROMPT,
    "tools": TOOL_SYSTEM_PROMPT,
    "web": WEB_SYSTEM_PROMPT,
    "skills": SKILL_SYSTEM_PROMPT,
}
PROMPT_META = [
    ("format", "Layout", "Sent with every message. Asks the model to tag reasoning, "
                         "instructions, code and maths so each one gets its own container "
                         "in the reply. Clear this box to switch it off."),
    ("agent", "Agent", "Sent before the others whenever any toggle is on. Tells the model "
                       "its abilities here are real, so it stops replying that it cannot "
                       "browse the web or run commands."),
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
    "cyan": "#57bcd9",        # system RAM, so it reads apart from TEMP

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
        if crash_logging_enabled():
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"\n===== {stamp} =====\n{text}")
            except OSError:
                pass
            record_crash(exc_type.__name__, str(exc) or exc_type.__name__, text)
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

# Content tags the model is asked to wrap its output in, so each kind of content
# gets its own container instead of one undifferentiated wall of prose. Matched
# unclosed too, so a container appears as soon as the tag opens while streaming.
TAG_NAMES = ("reasoning", "instructions", "math", "code", "text", "tasks")
ACTION_NAMES = ("run", "search", "fetch", "skill")

TAG_OPEN_RE = re.compile(
    r"<(reasoning|instructions|math|code|text|tasks|run|search|fetch|skill)"
    r"(?:\s+lang=[\"']?([\w+#.-]+)[\"']?)?\s*>", re.I)
# What an *unclosed* tag gives way to: the next tag that opens. Content tags
# only — deliberately NOT the action tags. If the model never closed
# <reasoning>, a ```run after it is as likely to be a command it was turning
# over in its head as one it wants carried out, and guessing wrong executes
# something nobody approved. It stays inside the thinking, where it cannot run,
# and _nudge_if_stranded asks the model to close the tag and send it again.
TAG_BOUNDARY_RE = re.compile(
    r"<(?:reasoning|instructions|math|code|text|tasks)(?:\s+[^<>]*)?>", re.I)
ORPHAN_CLOSE_RE = re.compile(
    r"</(?:reasoning|instructions|math|code|text|tasks)\s*>", re.I)

# Models that keep their thinking in the content stream delimit it with their own
# control tokens rather than a tag we asked for. Left alone, `<|END_THINKING|>`
# printed literally in the reply *and* left the <reasoning> before it unclosed.
CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_THINK_OPEN = re.compile(r"^<\|(?:start|begin)[_ ]?(?:of[_ ])?think", re.I)
_THINK_CLOSE = re.compile(r"^<\|end[_ ]?(?:of[_ ])?think", re.I)


def normalize_control_tokens(text: str) -> str:
    """Rewrite a model's private thinking delimiters as <reasoning> tags.

    `<|START_THINKING|>` / `<|END_THINKING|>` and friends are how several models
    mark thinking that arrives in the ordinary content stream. They are not
    anything the user should see, and an `<|END_THINKING|>` where a
    `</reasoning>` was expected left the tag open — which used to swallow the
    rest of the reply. Anything else in `<|…|>` form is a control token too, so
    it is dropped rather than displayed.
    """
    # <think> is the same idea in tag form; folding it in here means unclosed
    # thinking is handled in exactly one place, by iter_tags.
    if "<think" in text or "</think" in text:
        text = re.sub(r"(?i)<think\s*>", "<reasoning>", text)
        text = re.sub(r"(?i)</think\s*>", "</reasoning>", text)
    if "<|" not in text:
        return text                      # the overwhelmingly common case

    def sub(m):
        tok = m.group(0)
        if _THINK_OPEN.match(tok):
            return "<reasoning>"
        if _THINK_CLOSE.match(tok):
            return "</reasoning>"
        return ""

    return CONTROL_TOKEN_RE.sub(sub, text)


def iter_tags(text: str):
    """Yield (kind, lang, body, start, end) for each content tag, in order.

    A tag with a closing partner ends there. A tag **without** one ends at the
    next tag that opens — not at the end of the reply.

    That second rule is the whole point. Models forget `</reasoning>` regularly,
    and treating the rest of the message as thinking meant the answer, the
    checklist and any action block after it were all counted as thinking:
    nothing was rendered outside the collapsed panel, and because commands
    inside reasoning are deliberately never executed, the run block was
    silently dropped. The model then sat waiting for output that was never
    coming, and the user had to keep typing "carry on".
    """
    pos = 0
    while True:
        m = TAG_OPEN_RE.search(text, pos)
        if not m:
            return
        kind = m.group(1).lower()
        lang = (m.group(2) or "").strip()
        close = re.compile(rf"</{kind}\s*>", re.I).search(text, m.end())
        if close:
            body, end = text[m.end():close.start()], close.end()
        else:
            nxt = TAG_BOUNDARY_RE.search(text, m.end())
            stop = nxt.start() if nxt else len(text)
            body, end = text[m.end():stop], stop
        yield kind, lang, body, m.start(), end
        pos = end


TAG_LABELS = {
    "reasoning": "Reasoning",
    "instructions": "Instructions",
    "math": "Maths",
    "tasks": "Tasks",
    "text": "",
    "code": "",
}


def split_blocks(text: str):
    """Split a reply into typed blocks the UI can render in separate containers.

    Returns tuples: ("prose", text) | ("code", lang, body) |
    ("reasoning"|"instructions"|"math", body).

    Prose is further split on blank lines so a streaming reply only re-renders
    its final paragraph rather than an ever-growing wall of text.
    """
    text = normalize_control_tokens(text)
    blocks = []
    pos = 0
    for kind, lang, body, start, end in iter_tags(text):
        before = text[pos:start]
        if before.strip():
            blocks.extend(_split_fenced(before))
        if body.strip():
            if kind == "code":
                blocks.append(("code", lang, body))
            elif kind in ACTION_NAMES:
                # Shown the same as the ```run fence it should have been, so a
                # model that reaches for the tag form still reads correctly.
                blocks.append(("code", kind, body.strip("\n")))
            elif kind == "text":
                blocks.extend(_split_fenced(body))
            elif kind == "reasoning" and blocks and blocks[-1][0] == "reasoning":
                # One panel, not one per tag. A model that closes and reopens
                # <reasoning> mid-thought produced a stack of half-empty boxes.
                blocks[-1] = ("reasoning", blocks[-1][1].rstrip() + "\n" + body.strip())
            else:
                blocks.append((kind, body))
        pos = end
    rest = text[pos:]
    if rest.strip():
        blocks.extend(_split_fenced(rest))
    return blocks


def _split_fenced(text: str):
    """Untagged content: markdown prose with ``` fenced code."""
    # A closing tag whose opener was never matched is markup, not prose.
    text = ORPHAN_CLOSE_RE.sub("", text)
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


def _strip_kind(text: str, wanted: str) -> str:
    out, pos = [], 0
    for kind, _lang, _body, start, end in iter_tags(text):
        if kind != wanted:
            continue
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def strip_reasoning(text: str) -> str:
    """The answer without any thinking — for deciding what to *act* on.

    Shares iter_tags' unclosed-tag rule, so a missing `</reasoning>` no longer
    erases the entire answer along with the action block in it.
    """
    return _strip_kind(normalize_control_tokens(text), "reasoning")


def reasoning_text(text: str) -> str:
    """Only the thinking — used to tell a stranded action block apart from none."""
    text = normalize_control_tokens(text)
    return "\n".join(body for kind, _l, body, _s, _e in iter_tags(text)
                     if kind == "reasoning")


def split_think(raw: str):
    """Return (reasoning, answer).

    <think> is now rewritten to <reasoning> by normalize_control_tokens, so
    split_blocks renders it through the ordinary tag path and this returns no
    separate reasoning. Kept because callers want the answer without thinking.
    """
    return "", strip_reasoning(raw).strip()


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

class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def read_ram() -> dict:
    """System memory, straight from the Windows API.

    ctypes rather than psutil: the whole point of the portable build is that it
    carries no dependency the user did not ask for, and this is three fields.
    """
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {}
        gb = 1024 ** 3
        total = status.ullTotalPhys / gb
        avail = status.ullAvailPhys / gb
        return {"ram_total": total, "ram_used": total - avail,
                "ram_pct": float(status.dwMemoryLoad),
                "swap_total": status.ullTotalPageFile / gb,
                "swap_used": (status.ullTotalPageFile - status.ullAvailPageFile) / gb}
    except Exception:                                              # noqa: BLE001
        return {}


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
                return {"error": (r.stderr or r.stdout).strip() or "nvidia-smi failed",
                        **read_ram()}
            row = [c.strip() for c in r.stdout.strip().splitlines()[0].split(",")]
            d = dict(zip(GPU_FIELDS, row))

            def num(key, default=0.0):
                try:
                    return float(d.get(key, ""))
                except ValueError:
                    return default

            out = {
                **read_ram(),
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


def is_parser_error(detail: str) -> bool:
    """Ollama runs a grammar parser over a thinking/tool model's output. Some
    fine-tunes drift from the format their base model declares, and the parse
    fails with a 500 even though generation itself was fine."""
    low = (detail or "").lower()
    return ("peg-native" in low
            or "does not match the expected" in low
            or "unable to parse" in low and "format" in low
            or "failed to parse" in low and "format" in low)


def flatten_for_generate(messages):
    """Render a conversation as (system, prompt) for /api/generate.

    That endpoint takes one prompt rather than a message list, so roles are
    labelled in the text instead. Only used on the parser-error fallback path,
    where a slightly plainer prompt is a fair price for getting an answer at
    all instead of a dead turn.
    """
    system = "\n\n".join(m.get("content", "") for m in messages
                         if m.get("role") == "system").strip()
    lines = []
    for m in messages:
        if m.get("role") == "system":
            continue
        who = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {m.get('content', '')}")
    lines.append("Assistant:")
    return system, "\n\n".join(lines)


class ChatWorker(QThread):
    chunk = pyqtSignal(str)
    done = pyqtSignal(float, int, int)      # seconds, output tokens, prompt tokens
    failed = pyqtSignal(str)
    notice = pyqtSignal(str)

    # How long the stream may go silent before the model counts as stalled.
    # Covers a cold load of a large model; a healthy reply never pauses this long.
    STALL_S = 300

    def __init__(self, messages, model: str, num_gpu=None, think=None):
        super().__init__()
        self.messages = messages
        self.model = model
        self.num_gpu = num_gpu
        self.think = think
        self._stop = False
        self._streamed = False

    def stop(self):
        self._stop = True

    def run(self):
        outcome, detail = self._attempt(self.think)
        if outcome == "parser-error" and self.think is not False:
            # Retrying without thinking skips the parser that just rejected the
            # output. Only safe because nothing was streamed to the user yet.
            self.notice.emit(
                "This model's output did not match the format Ollama expected, "
                "so the reply was retried with thinking turned off.")
            outcome, detail = self._attempt(False)
        if outcome == "parser-error" and not self._stop:
            # Last resort: the plain completion endpoint. /api/chat runs a
            # grammar parser over the reply and 500s when the model's output
            # drifts from the format its template declares; /api/generate does
            # no such parsing, so the same generation comes back fine. This is
            # what turns the error from a dead turn into an answer.
            self.notice.emit(
                "Ollama still could not parse the reply, so it was retried "
                "through the plain completion endpoint.")
            outcome, detail = self._attempt(False, endpoint="generate")
        if outcome == "error" and not self._streamed and not self._stop:
            # Nothing reached the user, so a silent retry cannot repeat text.
            # One attempt: a second identical failure is a real problem.
            self.notice.emit("The model did not respond — retrying…")
            outcome, detail = self._attempt(self.think)
        if outcome == "error":
            self.failed.emit(detail)
        elif outcome == "parser-error":
            self.failed.emit(
                "Ollama could not parse this model's output.\n\n"
                "The model generated a reply, but it did not match the format "
                "Ollama expects for this architecture — common with merged or "
                "fine-tuned builds. Retrying without thinking, and again "
                "through the plain completion endpoint, did not help either. "
                "Try another model, or a different build of this one.\n\n"
                f"Ollama said: {detail}")

    def _attempt(self, think, endpoint="chat"):
        """Returns (outcome, detail). Emits chunks/done itself on success."""
        t0 = time.time()
        tokens = 0
        streamed = False
        self._streamed = False
        thinking_open = False
        final = {}
        try:
            if endpoint == "generate":
                system, prompt = flatten_for_generate(self.messages)
                body = {"model": self.model, "prompt": prompt, "stream": True}
                if system:
                    body["system"] = system
                url = GEN_URL
            else:
                body = {"model": self.model, "messages": self.messages,
                        "stream": True}
                if think is not None:
                    body["think"] = think
                url = CHAT_URL
            if self.num_gpu is not None:
                body["options"] = {"num_gpu": self.num_gpu}
            with requests.post(
                url, json=body, stream=True, timeout=(10, self.STALL_S),
            ) as resp:
                if resp.status_code >= 400:
                    # Ollama explains itself in the body; raise_for_status throws
                    # that away and leaves the user staring at a bare status code.
                    try:
                        detail = resp.json().get("error") or resp.text[:300]
                    except Exception:                             # noqa: BLE001
                        detail = resp.text[:300] or "no detail given"
                    detail = str(detail)
                    if is_parser_error(detail):
                        return "parser-error", detail
                    return "error", (f"Ollama rejected the request "
                                     f"({resp.status_code}): {detail}")
                for line in resp.iter_lines():
                    if self._stop:
                        break
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        detail = str(data["error"])
                        if is_parser_error(detail):
                            # A parse failure mid-stream can only be retried
                            # while nothing has reached the user; a retry after
                            # that would repeat text already on screen.
                            if not streamed:
                                return "parser-error", detail
                            # Text did arrive, so generation itself worked and
                            # only Ollama's parse of it failed. Keeping the
                            # answer beats throwing it away over that — and it
                            # lets any run/search block in it still be acted on.
                            self.notice.emit(
                                "Ollama could not parse the end of this reply, "
                                "so it is shown as far as it got.")
                            break
                        return "error", detail
                    msg = data.get("message", {})
                    # Native thinking arrives in its own field, not in content.
                    # Ignoring it meant the reasoning was never shown or saved —
                    # wrap it in the same <reasoning> tag models write inline,
                    # and everything downstream (panel, save, reload) just works.
                    tpiece = msg.get("thinking", "")
                    if tpiece:
                        tokens += 1
                        streamed = self._streamed = True
                        if not thinking_open:
                            thinking_open = True
                            self.chunk.emit("<reasoning>")
                        self.chunk.emit(tpiece)
                    # /api/generate returns its text as "response"; /api/chat
                    # nests it under "message".
                    piece = msg.get("content", "") or data.get("response", "")
                    if piece:
                        tokens += 1
                        streamed = self._streamed = True
                        if thinking_open:
                            thinking_open = False
                            self.chunk.emit("</reasoning>")
                        self.chunk.emit(piece)
                    if data.get("done"):
                        final = data
                        break
            # Closed here rather than in the done branch so a reply cut short
            # by a parse error or a stop still ends its reasoning tag.
            if thinking_open:
                self.chunk.emit("</reasoning>")
            # Ollama reports the true counts on the final chunk. Counting
            # chunks, as this used to, only approximates the output and says
            # nothing at all about what the prompt cost.
            out_tokens = int(final.get("eval_count") or tokens)
            in_tokens = int(final.get("prompt_eval_count") or 0)
            self.done.emit(time.time() - t0, out_tokens, in_tokens)
            return "ok", ""
        except requests.exceptions.Timeout:
            mins = self.STALL_S // 60
            return "error", (f"The model stopped responding — no output for "
                             f"{mins} minutes. It may be out of memory or wedged; "
                             f"try again, or switch model or compute mode.")
        except requests.exceptions.ConnectionError:
            if streamed:
                return "error", ("The connection to Ollama dropped mid-reply. "
                                 "The partial answer is kept above — Regenerate "
                                 "to try the turn again.")
            return "error", "Cannot reach Ollama on 127.0.0.1:11434."
        except Exception as exc:                                  # noqa: BLE001
            return "error", str(exc)


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

def auto_title(messages: list) -> str:
    """A chat's title when the user has not named it: its opening question."""
    first_user = next((m["content"] for m in messages if m.get("role") == "user"),
                      "New chat")
    squashed = " ".join(first_user.split())
    # Stored long; the sidebar elides it to fit, so nothing is lost here.
    return squashed[:120] or "New chat"


class Bubble(QWidget):
    regenerate = pyqtSignal()
    edit_prompt = pyqtSignal()

    def __init__(self, role: str, mono: str, badge: str = "ASSISTANT"):
        super().__init__()
        self.role, self.mono = role, mono
        self.raw = ""
        self.msg_index = None          # position in the chat's message list

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        who = QLabel("You" if role == "user" else badge.title())
        who.setObjectName("who_user" if role == "user" else "who_bot")
        if role == "user":
            who.hide()          # right alignment already says who wrote it

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

        # Per-message actions. Hidden until the turn is settled, so they never
        # appear on half-streamed text; shown by the window once it knows where
        # this bubble sits in the conversation.
        self.actions = QWidget()
        act = QHBoxLayout(self.actions)
        act.setContentsMargins(0, 0, 0, 0)
        act.setSpacing(4)
        if role == "user":
            act.addStretch(1)
        self.copy_btn = self._action("Copy", self._copy_all)
        act.addWidget(self.copy_btn)
        if role == "user":
            self.edit_btn = self._action("Edit", self.edit_prompt.emit,
                                         "Put this prompt back in the box and "
                                         "drop everything after it")
            act.addWidget(self.edit_btn)
        else:
            self.regen_btn = self._action("Regenerate", self.regenerate.emit,
                                          "Answer this turn again")
            act.addWidget(self.regen_btn)
        if role != "user":
            act.addStretch(1)
        self.actions.hide()
        outer.addWidget(self.actions)

        outer.addWidget(self.meta, alignment=align)

    def _action(self, text: str, slot, tip: str = "") -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("msgAction")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tip or text)
        btn.clicked.connect(slot)
        return btn

    def _copy_all(self):
        _, answer = split_think(self.raw)
        QApplication.clipboard().setText((answer or self.raw).strip())
        self.copy_btn.setText("Copied")
        # A parented QTimer dies with the widget. The context-argument form of
        # singleShot does not exist in this PyQt6 build — passing it raises
        # inside the click slot, which aborts the whole process.
        timer = QTimer(self.copy_btn)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.copy_btn.setText("Copy"))
        timer.start(1200)

    def set_index(self, index: int):
        """Anchor this bubble to a message, which enables its actions."""
        self.msg_index = index
        self.actions.show()

    def offer_retry(self):
        """Actions for a failed turn: it has no message to anchor to, so the
        regenerate signal is wired by the window to a retry instead."""
        if hasattr(self, "regen_btn"):
            self.regen_btn.setText("Retry")
            self.regen_btn.setToolTip("Try this turn again")
        self.actions.show()

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
        # <think> and <|…THINKING…|> are rewritten to <reasoning> on the way in,
        # so thinking reaches the same panel as an explicit tag by the same
        # path — and sits where the model put it rather than always on top.
        blocks = split_blocks(raw) if raw.strip() else []
        answer = strip_reasoning(raw).strip()
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

        # The usual streaming case: everything is settled except one block that
        # is still growing. Update that one widget instead of rebuilding. Which
        # block it is depends on the reply: the answer trails the reasoning, but
        # a <think> trace keeps growing while the answer is already on screen.
        if (len(prev) == len(blocks) and same < len(blocks)
                and self.body_layout.count() == len(prev)
                and prev[same][0] == blocks[same][0]
                and prev[same + 1:] == blocks[same + 1:]):
            widget = self.body_layout.itemAt(same).widget()
            block = blocks[same]
            if block[0] == "prose" and isinstance(widget, QLabel):
                widget.setText(_inline(block[1]))
                self._rendered = list(blocks)
                return
            if (block[0] == "code" and isinstance(widget, CodeBlock)
                    and block[1] == prev[same][1]):     # same language
                widget.set_code(block[2])
                self._rendered = list(blocks)
                return
            if block[0] == "reasoning" and isinstance(widget, ReasoningPanel):
                widget.set_text(block[1])
                self._rendered = list(blocks)
                return
            if (block[0] in ("instructions", "math", "tasks")
                    and isinstance(widget, LabelledBlock)):
                widget.set_body(block[1])
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
            self.body_layout.addWidget(self._build(block))
        self._rendered = list(blocks)

    def _build(self, block):
        kind = block[0]
        if kind == "prose":
            return self._prose_label(_inline(block[1]))
        if kind == "code":
            _, lang, code = block
            return CodeBlock(code, lang, self.mono)
        if kind == "reasoning":
            panel = ReasoningPanel(self.mono)
            panel.set_text(block[1])
            return panel
        return LabelledBlock(kind, block[1], self.mono)

    def set_error(self, text: str):
        self._clear_body()
        lbl = self._prose_label(f'<span style="color:{C["red"]}">⚠ {_html.escape(text)}</span>')
        self.body_layout.addWidget(lbl)

    def set_meta(self, text: str):
        self.meta.setText(text)
        self.meta.show()


class ReasoningPanel(QFrame):
    """The model's thinking. Collapsed it shows only the newest line, so a long
    trace never buries the answer; expanded it shows the whole thing, scrolled to
    the latest line as it arrives."""

    MAX_OPEN_HEIGHT = 260

    def __init__(self, mono: str):
        super().__init__()
        self.setObjectName("reasonPanel")
        self.setMaximumWidth(BUBBLE_MAX)
        self._lines = []
        self._open = False

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.toggle = QPushButton("▸  Reasoning")
        self.toggle.setObjectName("reasonToggle")
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.clicked.connect(self.toggle_open)
        head.addWidget(self.toggle)
        self.count_lbl = QLabel("")
        self.count_lbl.setObjectName("reasonCount")
        head.addWidget(self.count_lbl)
        head.addStretch()
        v.addLayout(head)

        # collapsed: just the line currently being thought. Plain text, not
        # Qt's auto-detection — thinking is full of angle brackets (`<PID>`,
        # `a < b`), and guessed rich text swallows them silently.
        self.latest = QLabel("")
        self.latest.setObjectName("reasonLatest")
        self.latest.setTextFormat(Qt.TextFormat.PlainText)
        self.latest.setWordWrap(True)
        v.addWidget(self.latest)

        # expanded: the full trace
        self.full = QPlainTextEdit()
        self.full.setObjectName("reasonFull")
        self.full.setReadOnly(True)
        self.full.setFont(mono_font(mono, CODE_PX_SM))
        self.full.setFrameShape(QFrame.Shape.NoFrame)
        self.full.hide()
        v.addWidget(self.full)

    def toggle_open(self):
        self._open = not self._open
        self.toggle.setText(("▾  " if self._open else "▸  ") + "Reasoning")
        self.full.setVisible(self._open)
        self.latest.setVisible(not self._open)
        if self._open:
            self._scroll_to_end()

    def set_text(self, text: str):
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        if lines == self._lines:
            return
        self._lines = lines
        self.count_lbl.setText(f"{len(lines)} line{'s' if len(lines) != 1 else ''}")
        self.latest.setText(lines[-1] if lines else "…")
        self.full.setPlainText("\n".join(lines))
        fm = self.full.fontMetrics()
        self.full.setFixedHeight(
            min(int(fm.lineSpacing() * max(1, len(lines))) + 20, self.MAX_OPEN_HEIGHT))
        if self._open:
            self._scroll_to_end()

    def _scroll_to_end(self):
        bar = self.full.verticalScrollBar()
        bar.setValue(bar.maximum())          # always resting on the newest line


class LabelledBlock(QFrame):
    """Instructions and maths: a titled container so they read as their own thing."""

    def __init__(self, kind: str, body: str, mono: str):
        super().__init__()
        self.kind = kind
        self.setObjectName({"math": "mathBlock",
                            "tasks": "tasksBlock"}.get(kind, "instrBlock"))
        self.setMaximumWidth(BUBBLE_MAX)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(6)

        title = QLabel(TAG_LABELS.get(kind, kind.title()))
        title.setObjectName({"math": "mathTitle",
                             "tasks": "tasksTitle"}.get(kind, "instrTitle"))
        v.addWidget(title)

        self.body = QLabel()
        # Equations only line up in a monospaced face, and a stylesheet
        # font-family beats setFont, so maths needs its own styled name.
        self.body.setObjectName("mathBody" if kind == "math" else "blockBody")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if kind == "math":
            self.body.setFont(mono_font(mono, CODE_PX))
        v.addWidget(self.body)
        self.set_body(body)

    def set_body(self, body: str):
        self._raw = body
        if self.kind == "tasks":
            self.body.setText(self._checklist(body))
        else:
            self.body.setText(_inline(body.strip()))

    @staticmethod
    def _checklist(body: str) -> str:
        """Checklist lines with their state glyph coloured: ✓ done, ✗ cannot
        be done, ☐ still to do."""
        out = []
        for raw_line in body.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            colour = None
            if line[0] in "✓✔☑":
                colour = C["green"]
            elif line[0] in "✗✘✖":
                colour = C["red"]
            elif line[0] in "☐□○-":
                colour = C["faint"]
            text = _html.escape(line[1:].strip()) if colour else _html.escape(line)
            glyph = line[0] if colour else ""
            if colour:
                out.append(f'<span style="color:{colour};font-weight:600;">'
                           f'{glyph}</span>&nbsp; {text}')
            else:
                out.append(text)
        return "<br>".join(out)


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


class Spinner(QWidget):
    """Small rotating arc, shown on a chat that is still generating."""

    def __init__(self, colour: str, size: int = 14):
        super().__init__()
        self.colour = QColor(colour)
        self.setFixedSize(size, size)
        self._angle = 0
        self._timer = QTimer(self)              # parented: dies with the widget
        self._timer.setInterval(70)
        self._timer.timeout.connect(self._step)
        self.hide()

    def start(self):
        if not self._timer.isActive():
            self._timer.start()
        self.show()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _step(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        pen = QPen(QColor(C["line_str"]), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 0, 360 * 16)
        pen.setColor(self.colour)
        p.setPen(pen)
        p.drawArc(rect, -self._angle * 16, 100 * 16)
        p.end()


class BusyStrip(QFrame):
    """Spinner and status shown inside the composer while a reply is coming in.

    It floats over the input rather than taking layout space, so the box does not
    jump as generation starts and stops. It anchors itself to the parent's bottom
    left and follows every resize.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("busyStrip")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        h = QHBoxLayout(self)
        h.setContentsMargins(9, 4, 11, 4)
        h.setSpacing(8)
        self.spinner = Spinner(C["accent"], 13)
        h.addWidget(self.spinner)
        self.label = QLabel("Generating…")
        self.label.setObjectName("busyLabel")
        h.addWidget(self.label)
        parent.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() == ev.Type.Resize:
            self._reposition()
        return False

    def _reposition(self):
        parent = self.parent()
        if parent:
            self.adjustSize()
            self.move(8, max(0, parent.height() - self.height() - 8))

    def start(self, text: str = "Generating…"):
        self.label.setText(text)
        self.spinner.start()
        self._reposition()
        self.show()
        self.raise_()

    def stop(self):
        self.spinner.stop()
        self.hide()


class UsageBars(QWidget):
    """Tokens per day for the last N days. Painted rather than composed from
    widgets: one bar per day is a lot of widgets to lay out for a static chart."""

    def __init__(self, days: int = 30):
        super().__init__()
        self.days = days
        self.values = []                   # oldest → newest
        self.labels = []
        self.setMinimumHeight(150)

    def set_data(self, usage: dict, today: date | None = None):
        today = today or date.today()
        self.values, self.labels = [], []
        for i in range(self.days - 1, -1, -1):
            day = today - timedelta(days=i)
            rec = usage.get(day.isoformat(), {})
            self.values.append(rec.get("in", 0) + rec.get("out", 0))
            self.labels.append(day)
        self.update()

    def paintEvent(self, _):
        if not self.values:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        band = 18                                   # room for the date labels
        top, bottom = 4, self.height() - band
        usable = max(1, bottom - top)
        peak = max(self.values) or 1
        gap = 3
        slot = self.width() / len(self.values)
        width = max(2.0, slot - gap)

        p.setPen(QPen(QColor(C["line"]), 1))
        p.drawLine(0, bottom, self.width(), bottom)

        for i, value in enumerate(self.values):
            x = i * slot
            height = 0 if not value else max(2, round(usable * value / peak))
            colour = QColor(C["accent"]) if value else QColor(C["line"])
            p.fillRect(QRectF(x, bottom - height, width, height), colour)

        # Only the ends and the middle are labelled — 30 dates never fit. Each
        # label is kept inside the widget; a rect starting off-canvas simply
        # does not draw, which silently lost the first and last dates.
        p.setPen(QColor(C["faint"]))
        font = p.font()
        font.setPixelSize(10)
        p.setFont(font)
        mid = len(self.values) // 2
        band_top = bottom + 2
        places = (
            (0, QRectF(0, band_top, 70, band), Qt.AlignmentFlag.AlignLeft),
            (mid, QRectF(mid * slot - 35, band_top, 70, band),
             Qt.AlignmentFlag.AlignHCenter),
            (len(self.values) - 1, QRectF(self.width() - 70, band_top, 70, band),
             Qt.AlignmentFlag.AlignRight),
        )
        for idx, rect, align in places:
            text = self.labels[idx].strftime("%d %b")
            p.drawText(rect, int(align | Qt.AlignmentFlag.AlignVCenter), text)
        p.end()


class CrashGroup(QFrame):
    """One error type, collapsed. Opening it lists the days it happened on;
    opening a day shows each traceback."""

    delete_requested = pyqtSignal(str)

    def __init__(self, kind: str, records: list, mono: str):
        super().__init__()
        self.kind = kind
        self.mono = mono
        self.records = records
        self.setObjectName("crashGroup")
        self._open = False

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(10)
        self.toggle = QPushButton(f"▸  {kind}")
        self.toggle.setObjectName("crashToggle")
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.clicked.connect(self._toggle)
        head.addWidget(self.toggle)

        count = QLabel(f"{len(records)}×")
        count.setObjectName("crashCount")
        head.addWidget(count)

        newest = max(r.get("at", 0) for r in records)
        when = QLabel("last " + time.strftime("%d %b %Y, %H:%M", time.localtime(newest)))
        when.setObjectName("crashWhen")
        head.addWidget(when)
        head.addStretch()

        close = QPushButton("×")
        close.setObjectName("rowClose")
        close.setFixedSize(22, 22)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setToolTip(f"Delete every {kind} crash")
        close.clicked.connect(lambda: self.delete_requested.emit(self.kind))
        head.addWidget(close)
        v.addLayout(head)

        # The most recent message, so the group says something while collapsed.
        latest = QLabel(records[0].get("message", ""))
        latest.setObjectName("crashLatest")
        latest.setTextFormat(Qt.TextFormat.PlainText)
        latest.setWordWrap(True)
        v.addWidget(latest)

        self.days = QWidget()
        days_box = QVBoxLayout(self.days)
        days_box.setContentsMargins(0, 4, 0, 0)
        days_box.setSpacing(6)
        by_day = {}
        for rec in records:
            day = time.strftime("%Y-%m-%d", time.localtime(rec.get("at", 0)))
            by_day.setdefault(day, []).append(rec)
        for day in sorted(by_day, reverse=True):
            days_box.addWidget(CrashDay(day, by_day[day], mono))
        self.days.hide()
        v.addWidget(self.days)

    def _toggle(self):
        self._open = not self._open
        self.toggle.setText(("▾  " if self._open else "▸  ") + self.kind)
        self.days.setVisible(self._open)


class CrashDay(QFrame):
    """One day's crashes of a given kind, collapsed to a date line."""

    def __init__(self, day: str, records: list, mono: str):
        super().__init__()
        self.setObjectName("crashDay")
        self._open = False
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 8)
        v.setSpacing(6)

        pretty = time.strftime("%A %d %B %Y", time.strptime(day, "%Y-%m-%d"))
        self.toggle = QPushButton(f"▸  {pretty}   ({len(records)})")
        self.toggle.setObjectName("crashDayToggle")
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.clicked.connect(self._toggle)
        v.addWidget(self.toggle)

        self.body = QWidget()
        box = QVBoxLayout(self.body)
        box.setContentsMargins(0, 2, 0, 0)
        box.setSpacing(8)
        for rec in records:
            stamp = time.strftime("%H:%M:%S", time.localtime(rec.get("at", 0)))
            line = QLabel(f"{stamp}  ·  {rec.get('message', '')}")
            line.setObjectName("crashMsg")
            line.setTextFormat(Qt.TextFormat.PlainText)
            line.setWordWrap(True)
            box.addWidget(line)
            text = rec.get("trace", "")
            trace = QPlainTextEdit(text)
            trace.setObjectName("toolOutput")
            trace.setReadOnly(True)
            font = mono_font(mono, CODE_PX_SM)
            trace.setFont(font)
            trace.setFrameShape(QFrame.Shape.NoFrame)
            # Sized to the traceback, capped — a two-line error should not get
            # the same tall empty box as a deep stack.
            line_h = QFontMetrics(font).lineSpacing()
            lines = max(1, len(text.splitlines()))
            # + document margins and the frame, or the last line is clipped.
            trace.setFixedHeight(min(240, lines * line_h + 26))
            box.addWidget(trace)
        self.body.hide()
        v.addWidget(self.body)
        self._pretty = pretty
        self._n = len(records)

    def _toggle(self):
        self._open = not self._open
        self.toggle.setText(("▾  " if self._open else "▸  ")
                            + f"{self._pretty}   ({self._n})")
        self.body.setVisible(self._open)


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
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        texts.addWidget(self.label)
        self.sub = None
        if subtitle:
            self.sub = QLabel(subtitle)
            self.sub.setObjectName("rowSub")
            self.sub.setTextFormat(Qt.TextFormat.PlainText)
            self.sub.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            texts.addWidget(self.sub)
        row.addLayout(texts, 1)

        self.spinner = Spinner(C["green"])
        self.spinner.setToolTip("Still generating — you can browse other chats "
                                "while this finishes")
        row.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignVCenter)

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
        name_lbl.setTextFormat(Qt.TextFormat.PlainText)
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
        query_lbl.setTextFormat(Qt.TextFormat.PlainText)
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
        self.gens = {}                 # chat_id -> live generation state
        self.pending_actions = {}      # chat_id -> proposal awaiting approval
        self.current_model = ""
        self.tool_round = 0
        self._cmd_worker = None

        # Workers must stay referenced until they actually finish. Reassigning
        # self.chat_worker / self._cmd_worker was dropping the last reference to a
        # still-running QThread, which Qt aborts the whole process over.
        self._workers = []

        self.current_chat_id = None
        # A reply belongs to the chat it was asked in, not to whatever is on
        # screen. These follow the generation so the user can browse away.
        self.chats_dir = CHATS_DIR
        self.chats_dir.mkdir(exist_ok=True)
        self.skills = self._read_skills_file()
        self.prompts = self._read_prompts_file()
        self.settings = self._read_settings()
        self.usage = load_usage()
        self._import_worker = None

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1400, 820)
        self.setStyleSheet(self._qss())
        self._build()
        self._reload_chat_list()
        self._reload_skill_list()
        self._reload_skill_page()
        self._refresh_usage()
        self._reload_crashes()

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
        for i, name in enumerate(PAGES):
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
        # Built in PAGES order — the two must stay in step, which is why
        # everything downstream asks for a page by name rather than by number.
        builders = {
            "Chat": self._chat_tab, "Track": self._gpu_tab, "Usage": self._usage_tab,
            "Prompts": self._prompts_tab, "Skills": self._skills_tab,
            "Crashes": self._crashes_tab, "Models": self._models_tab,
            "About": self._about_tab,
        }
        for name in PAGES:
            self.stack.addWidget(builders[name]())
        col.addWidget(self.stack, 1)

    def _on_nav(self, index: int):
        if hasattr(self, "gpu"):
            self.gpu.set_active(index == page_index("Track"))
        # Also drive the button state, so programmatic navigation keeps the
        # highlighted tab in sync with the visible page.
        btn = self.nav_group.button(index)
        if btn and not btn.isChecked():
            btn.setChecked(True)
        self.stack.setCurrentIndex(index)
        if index == page_index("Usage"):
            self._refresh_usage()
        elif index == page_index("Skills"):
            self._reload_skill_page()
        elif index == page_index("Crashes"):
            self._reload_crashes()
        elif index == page_index("Models"):
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

        # Sits over the bottom-left of the input while a reply is generating, so
        # the wait is visible right where you are typing. Typing is still allowed
        # — the next question can be queued up while this one finishes.
        self.busy = BusyStrip(self.input)
        self.busy.hide()

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

        # Web and Skills are read-only, so remembering them is safe and stops the
        # app silently losing its abilities on every launch. Tools and Auto-run
        # deliberately stay off until asked for: they execute things.
        self.web_check.setChecked(bool(self.settings.get("web_enabled", False)))
        self.skills_check.setChecked(bool(self.settings.get("skills_enabled", False)))
        self.web_check.toggled.connect(
            lambda on: self._remember_toggle("web_enabled", on))
        self.skills_check.toggled.connect(
            lambda on: self._remember_toggle("skills_enabled", on))

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
        self.g_ram = RingGauge("RAM", C["cyan"])
        self.g_temp = RingGauge("TEMP", C["green"], unit="°")
        self.g_pow = RingGauge("POWER", C["amber"])
        for i, w in enumerate((self.g_util, self.g_mem, self.g_ram,
                               self.g_temp, self.g_pow)):
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
        c3 = Card("System RAM · last 2 min")
        self.spark_ram = Sparkline(C["cyan"])
        c3.box.addWidget(self.spark_ram)
        graphs.addWidget(c1)
        graphs.addWidget(c2)
        graphs.addWidget(c3)
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
    # -- usage page -------------------------------------------------------- #
    def _record_usage(self, out_tokens: int, in_tokens: int, seconds: float):
        """Add one reply to today's bucket."""
        if not (out_tokens or in_tokens):
            return
        key = date.today().isoformat()
        self.usage = load_usage()          # re-read: another window may have written
        rec = self.usage.setdefault(
            key, {"in": 0, "out": 0, "replies": 0, "seconds": 0.0, "models": {}})
        rec["in"] += int(in_tokens)
        rec["out"] += int(out_tokens)
        rec["replies"] += 1
        rec["seconds"] = round(rec.get("seconds", 0.0) + seconds, 1)
        model = self.current_model or "unknown"
        rec.setdefault("models", {})
        rec["models"][model] = rec["models"].get(model, 0) + int(out_tokens) + int(in_tokens)
        try:
            _atomic_write_json(USAGE_PATH, self.usage)
        except Exception:                                          # noqa: BLE001
            pass                                                   # never break a reply
        if self.stack.currentIndex() == page_index("Usage"):
            self._refresh_usage()

    def _usage_tab(self) -> QWidget:
        page, body = self._page(
            "Token usage",
            "How much these models have generated on this machine. Counted from what "
            "Ollama reports for each reply — prompt tokens are what CriGent sent, "
            "output tokens what the model wrote back. Nothing leaves this computer.")

        self.usage_cards = {}
        totals = Card("Totals")
        grid = QHBoxLayout()
        grid.setSpacing(10)
        for key, label in (("today", "Today"), ("week", "Last 7 days"),
                           ("month", "Last 30 days"), ("all", "All time")):
            box = QFrame()
            box.setObjectName("statBox")
            v = QVBoxLayout(box)
            v.setContentsMargins(14, 12, 14, 12)
            v.setSpacing(2)
            cap = QLabel(label)
            cap.setObjectName("statCap")
            value = QLabel("—")
            value.setObjectName("statValue")
            sub = QLabel("")
            sub.setObjectName("statSub")
            sub.setWordWrap(True)
            v.addWidget(cap)
            v.addWidget(value)
            v.addWidget(sub)
            grid.addWidget(box, 1)
            self.usage_cards[key] = (value, sub)
        totals.box.addLayout(grid)
        body.addWidget(totals)

        chart = Card("Last 30 days")
        self.usage_bars = UsageBars(30)
        chart.box.addWidget(self.usage_bars)
        body.addWidget(chart)

        self.usage_models_card = Card("By model · last 30 days")
        self.usage_models = QLabel("—")
        self.usage_models.setObjectName("pageSub")
        self.usage_models.setWordWrap(True)
        self.usage_models_card.box.addWidget(self.usage_models)
        body.addWidget(self.usage_models_card)

        body.addStretch()
        return page

    def _refresh_usage(self):
        self.usage = load_usage()
        today = date.today()
        spans = {"today": 1, "week": 7, "month": 30, "all": None}
        for key, days in spans.items():
            value, sub = self.usage_cards[key]
            t = usage_totals(self.usage, days, today)
            value.setText(f"{t['total']:,}")
            if t["replies"]:
                mins = t["seconds"] / 60
                spent = f"{mins:.0f} min" if mins >= 1 else f"{t['seconds']:.0f}s"
                value_sub = (f"{t['replies']:,} repl{'y' if t['replies'] == 1 else 'ies'} · "
                             f"{t['in']:,} in / {t['out']:,} out · {spent} generating")
            else:
                value_sub = "nothing yet"
            sub.setText(value_sub)

        self.usage_bars.set_data(self.usage, today)

        per_model = {}
        for key, rec in self.usage.items():
            try:
                if (today - date.fromisoformat(key)).days >= 30:
                    continue
            except ValueError:
                continue
            for name, n in (rec.get("models") or {}).items():
                per_model[name] = per_model.get(name, 0) + n
        if per_model:
            ranked = sorted(per_model.items(), key=lambda kv: -kv[1])
            total = sum(per_model.values()) or 1
            lines = [f"<b>{_html.escape(model_label(name)[0])}</b> — "
                     f"{n:,} tokens ({n * 100 // total}%)"
                     for name, n in ranked]
            self.usage_models.setText("<br>".join(lines))
        else:
            self.usage_models.setText("No replies recorded in the last 30 days.")

    # -- crashes page ------------------------------------------------------ #
    def _crashes_tab(self) -> QWidget:
        page, body = self._page(
            "Crash log",
            "If something goes wrong inside CriGent, the error is written here instead "
            "of vanishing with the window. Grouped by error, then by the day it "
            "happened. Nothing is sent anywhere — it stays in this folder.")

        bar = Card()
        row = QHBoxLayout()
        self.crash_toggle = QPushButton("Recording crashes")
        self.crash_toggle.setObjectName("pillWeb")
        self.crash_toggle.setCheckable(True)
        self.crash_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.crash_toggle.setChecked(bool(self.settings.get("log_crashes", True)))
        self.crash_toggle.setToolTip(
            "Turn this off and crashes stop being written down. Anything already "
            "recorded is kept until you delete it.")
        self.crash_toggle.toggled.connect(self._on_crash_logging_toggled)
        row.addWidget(self.crash_toggle)

        self.crash_count = QLabel("")
        self.crash_count.setObjectName("meta")
        row.addWidget(self.crash_count)
        row.addStretch()

        self.crash_where = QLabel("")
        self.crash_where.setObjectName("meta")
        self.crash_where.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.crash_where)

        open_btn = QPushButton("Open folder")
        open_btn.setObjectName("ghostSm")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT))))
        row.addWidget(open_btn)

        clear_btn = QPushButton("Clear all")
        clear_btn.setObjectName("ghostSm")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_crashes)
        row.addWidget(clear_btn)
        bar.box.addLayout(row)
        body.addWidget(bar)

        self.crash_holder = QVBoxLayout()
        self.crash_holder.setSpacing(10)
        body.addLayout(self.crash_holder)

        self.crash_empty = QLabel("No crashes recorded. That is the idea.")
        self.crash_empty.setObjectName("pageSub")
        body.addWidget(self.crash_empty)
        body.addStretch()
        return page

    def _on_crash_logging_toggled(self, on: bool):
        self.crash_toggle.setText("Recording crashes" if on else "Not recording")
        self.settings["log_crashes"] = bool(on)
        self._save_settings()

    def _reload_crashes(self):
        if not hasattr(self, "crash_holder"):
            return
        while self.crash_holder.count():
            item = self.crash_holder.takeAt(0)
            widget = item.widget()
            if widget:
                # setParent(None) as well as deleteLater(): taking a widget out
                # of a layout does not unparent it, so it keeps painting over
                # the rebuilt list until the event loop gets round to deleting.
                widget.setParent(None)
                widget.deleteLater()

        crashes = load_crashes()
        self.crash_where.setText(str(CRASHES_PATH))
        self.crash_where.setToolTip(
            f"Structured log: {CRASHES_PATH}\nPlain text: {ROOT / 'crash.log'}")
        self.crash_count.setText(
            f"{len(crashes)} recorded" if crashes else "")
        self.crash_empty.setVisible(not crashes)

        # Grouped by error name, then by day — the two things you actually scan
        # for: "what is breaking" and "is it still breaking".
        by_kind = {}
        for rec in crashes:
            by_kind.setdefault(rec.get("kind", "Error"), []).append(rec)
        for kind, group in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            panel = CrashGroup(kind, group, self.mono)
            panel.delete_requested.connect(self._delete_crash_group)
            self.crash_holder.addWidget(panel)

    def _delete_crash_group(self, kind: str):
        reply = QMessageBox.question(
            self, "Delete crashes",
            f'Delete every recorded "{kind}" crash?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._remove_crashes(lambda rec: rec.get("kind", "Error") != kind)

    def _clear_crashes(self):
        if not load_crashes():
            return
        reply = QMessageBox.question(
            self, "Clear crash log", "Delete every recorded crash?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._remove_crashes(lambda rec: False)

    def _remove_crashes(self, keep):
        """Rewrite the log with only the records `keep` approves."""
        _atomic_write_json(CRASHES_PATH, [r for r in load_crashes() if keep(r)])
        self._reload_crashes()

    # -- skills page ------------------------------------------------------- #
    def _skills_tab(self) -> QWidget:
        page, body = self._page(
            "Skills",
            "Reusable instructions you can switch on for a conversation. Pick one to "
            "read or edit it here; tick it in the panel beside the chat to apply it.")

        split = QHBoxLayout()
        split.setSpacing(14)

        # left: the list, with an × per row like every other list in the app
        left = Card()
        left.setMaximumWidth(300)
        new_btn = QPushButton("New skill")
        new_btn.setObjectName("ghost")
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.clicked.connect(self._new_skill_on_page)
        left.box.addWidget(new_btn)

        self.skill_page_list = QListWidget()
        self.skill_page_list.setObjectName("plainList")
        self.skill_page_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.skill_page_list.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.skill_page_list.currentItemChanged.connect(
            lambda cur, _prev: self._show_skill(
                cur.data(Qt.ItemDataRole.UserRole) if cur else None))
        left.box.addWidget(self.skill_page_list, 1)

        self.skill_page_empty = QLabel(
            "No skills yet. Create one here, or ask the model to write one for you.")
        self.skill_page_empty.setObjectName("pageSub")
        self.skill_page_empty.setWordWrap(True)
        left.box.addWidget(self.skill_page_empty)
        split.addWidget(left)

        # right: name, content, save
        right = Card()
        self.skill_form = QWidget()
        form = QVBoxLayout(self.skill_form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        name_row = QHBoxLayout()
        name_lab = QLabel("Name")
        name_lab.setObjectName("cardTitle")
        name_row.addWidget(name_lab)
        name_row.addStretch()
        self.skill_meta = QLabel("")
        self.skill_meta.setObjectName("meta")
        name_row.addWidget(self.skill_meta)
        form.addLayout(name_row)

        self.skill_name_box = QLineEdit()
        self.skill_name_box.setObjectName("input")
        self.skill_name_box.setPlaceholderText("Web scraping")
        self.skill_name_box.textEdited.connect(self._on_skill_edited)
        form.addWidget(self.skill_name_box)

        content_lab = QLabel("Instructions")
        content_lab.setObjectName("cardTitle")
        form.addWidget(content_lab)
        self.skill_content_box = QPlainTextEdit()
        self.skill_content_box.setObjectName("promptBox")
        self.skill_content_box.setFont(mono_font(self.mono, CODE_PX))
        self.skill_content_box.setMinimumHeight(300)
        self.skill_content_box.textChanged.connect(self._on_skill_edited)
        form.addWidget(self.skill_content_box, 1)

        actions = QHBoxLayout()
        self.skill_save_btn = QPushButton("Save skill")
        self.skill_save_btn.setObjectName("primary")
        self.skill_save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skill_save_btn.clicked.connect(self._save_skill_edits)
        actions.addWidget(self.skill_save_btn)
        self.skill_revert_btn = QPushButton("Revert")
        self.skill_revert_btn.setObjectName("ghost")
        self.skill_revert_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skill_revert_btn.setToolTip("Discard unsaved edits to this skill")
        self.skill_revert_btn.clicked.connect(
            lambda: self._show_skill(self.editing_skill_id, force=True))
        actions.addWidget(self.skill_revert_btn)
        self.skill_status = QLabel("")
        self.skill_status.setObjectName("meta")
        self.skill_status.setWordWrap(True)
        actions.addWidget(self.skill_status)
        actions.addStretch()
        form.addLayout(actions)

        right.box.addWidget(self.skill_form)
        self.skill_placeholder = QLabel(
            "Select a skill on the left, or create one.")
        self.skill_placeholder.setObjectName("pageSub")
        right.box.addWidget(self.skill_placeholder)
        split.addWidget(right, 1)

        body.addLayout(split, 1)
        self.editing_skill_id = None
        self._editor_loaded_id = None      # what the boxes are actually showing
        return page

    def _reload_skill_page(self):
        """Refill the list, keeping the selection where it can be kept."""
        if not hasattr(self, "skill_page_list"):
            return
        want = self.editing_skill_id
        self.skill_page_list.blockSignals(True)
        self.skill_page_list.clear()
        for sk in self.skills:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, sk["id"])
            item.setToolTip(sk["name"])
            lines = len(sk.get("content", "").splitlines())
            row = ListRow(sk["id"], sk["name"],
                          f"{lines} line{'' if lines == 1 else 's'}")
            item.setSizeHint(QSize(row.sizeHint().width(), row.height()))
            self.skill_page_list.addItem(item)
            self.skill_page_list.setItemWidget(item, row)
            row.delete_requested.connect(self._delete_skill)
            if sk["id"] == want:
                self.skill_page_list.setCurrentItem(item)
        self.skill_page_list.blockSignals(False)
        self.skill_page_empty.setVisible(not self.skills)

        if not self.skills:
            self._show_skill(None)
            return
        current = self.skill_page_list.currentItem()
        if current is None:
            self.skill_page_list.setCurrentRow(0)          # emits, loads the first
            return
        # Restoring the selection above was done with signals blocked, so the
        # editor was left showing whatever it had. Reload it only when it is
        # actually showing something else — otherwise switching tabs with
        # unsaved edits would silently discard them.
        selected = current.data(Qt.ItemDataRole.UserRole)
        if self._editor_loaded_id != selected:
            self._show_skill(selected, force=True)

    def _show_skill(self, skill_id, force: bool = False):
        """Load a skill into the editor. Unsaved edits are confirmed first, so
        clicking another row cannot quietly discard your work."""
        if (not force and self.editing_skill_id and skill_id != self.editing_skill_id
                and self._skill_is_dirty()):
            keep = QMessageBox.question(
                self, "Unsaved changes",
                "Save your changes to this skill before leaving it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if keep == QMessageBox.StandardButton.Yes:
                self._save_skill_edits(quiet=True)

        sk = next((s for s in self.skills if s["id"] == skill_id), None)
        self.editing_skill_id = sk["id"] if sk else None
        self._editor_loaded_id = self.editing_skill_id
        self.skill_form.setVisible(sk is not None)
        self.skill_placeholder.setVisible(sk is None)
        self.skill_status.setText("")
        if not sk:
            return
        self.skill_name_box.blockSignals(True)
        self.skill_name_box.setText(sk["name"])
        self.skill_name_box.blockSignals(False)
        self.skill_content_box.blockSignals(True)
        self.skill_content_box.setPlainText(sk.get("content", ""))
        self.skill_content_box.blockSignals(False)
        self.skill_meta.setText(
            f"updated {time.strftime('%d %b %Y, %H:%M', time.localtime(sk.get('updated', 0)))}"
            if sk.get("updated") else "")
        self._set_skill_dirty(False)

    def _skill_is_dirty(self) -> bool:
        sk = next((s for s in self.skills if s["id"] == self.editing_skill_id), None)
        if not sk:
            return False
        return (self.skill_name_box.text() != sk["name"]
                or self.skill_content_box.toPlainText() != sk.get("content", ""))

    def _set_skill_dirty(self, dirty: bool):
        self.skill_save_btn.setEnabled(dirty)
        self.skill_revert_btn.setEnabled(dirty)

    def _on_skill_edited(self):
        self._set_skill_dirty(self._skill_is_dirty())

    def _save_skill_edits(self, quiet: bool = False):
        sk = next((s for s in self.skills if s["id"] == self.editing_skill_id), None)
        if not sk:
            return
        name = self.skill_name_box.text().strip()
        content = self.skill_content_box.toPlainText().strip()
        if not name or not content:
            self.skill_status.setText("A skill needs both a name and instructions.")
            return
        sk["name"], sk["content"], sk["updated"] = name, content, time.time()
        self._save_skills()
        self._reload_skill_list()        # the chat-side ticks show the new name
        self._reload_skill_page()
        self._set_skill_dirty(False)
        if not quiet:
            self.skill_status.setText("Saved.")

    def _new_skill_on_page(self):
        self.skills.append({
            "id": uuid.uuid4().hex[:10], "name": "New skill",
            "content": "", "created": time.time(), "updated": time.time(),
        })
        self._save_skills()
        self.editing_skill_id = self.skills[-1]["id"]
        self._reload_skill_list()
        self._reload_skill_page()
        self.skill_name_box.setFocus()
        self.skill_name_box.selectAll()

    def _prompts_tab(self) -> QWidget:
        page, body = self._page(
            "Enrichment prompts",
            "These are prepended to your message — Layout every time, the rest when the "
            "matching toggle is on. Edit them to change how the model lays out a reply and "
            "how it uses tools, the web, and skills.")

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
        restore_all = QPushButton("Restore all defaults")
        restore_all.setObjectName("ghost")
        restore_all.setCursor(Qt.CursorShape.PointingHandCursor)
        restore_all.setToolTip("Put every prompt back to the wording CriGent ships with.")
        restore_all.clicked.connect(self._restore_all_prompts)
        actions.addWidget(restore_all)
        self.prompt_status = QLabel("")
        self.prompt_status.setObjectName("meta")
        self.prompt_status.setWordWrap(True)
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
        # Store only what the user actually changed. Anything still at the default
        # is left out, so improved wording in a later version reaches people who
        # never customised, instead of pinning them to old text forever.
        customised = {k: v for k, v in self.prompts.items()
                      if v.strip() != DEFAULT_PROMPTS.get(k, "").strip()}
        _atomic_write_json(PROMPTS_PATH, customised)
        kept = len(customised)
        self.prompt_status.setText(
            f"Saved — applies to your next message. "
            f"({kept} customised, {len(DEFAULT_PROMPTS) - kept} following the defaults.)")
        timer = QTimer(self.prompt_status)          # parented: dies with the widget
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.prompt_status.setText(""))
        timer.start(2600)

    def _reset_prompt(self, key: str):
        self.prompt_boxes[key].setPlainText(DEFAULT_PROMPTS[key])
        self.prompt_status.setText(
            f"{key.title()} restored to the default — click “Save prompts” to keep it.")

    def _restore_all_prompts(self):
        changed = [k for k, box in self.prompt_boxes.items()
                   if box.toPlainText().strip() != DEFAULT_PROMPTS.get(k, "").strip()]
        if not changed:
            self.prompt_status.setText("Every prompt is already at its default.")
            return
        reply = QMessageBox.question(
            self, "Restore all defaults",
            f"Replace {len(changed)} edited prompt(s) with the originals?\n\n"
            "Your current wording will be discarded.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._apply_default_prompts()

    def _apply_default_prompts(self):
        """Put every prompt back to the shipped wording and forget the overrides."""
        for key, box in self.prompt_boxes.items():
            box.setPlainText(DEFAULT_PROMPTS[key])
        self.prompts = dict(DEFAULT_PROMPTS)
        try:
            if PROMPTS_PATH.exists():
                PROMPTS_PATH.unlink()      # nothing customised left to store
        except OSError:
            pass
        self.prompt_status.setText("All prompts restored to the originals and saved.")

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

    def _add_bubble(self, role: str, badge: str = "ASSISTANT",
                    index: int | None = None) -> Bubble:
        b = Bubble(role, self.mono, badge)
        if index is not None:
            b.set_index(index)
            b.regenerate.connect(lambda w=b: self._regenerate(w))
            b.edit_prompt.connect(lambda w=b: self._edit_prompt(w))
        self.feed.insertWidget(self.feed.count() - 1, b)
        return b

    def _bubbles(self) -> list:
        return [self.feed.itemAt(i).widget() for i in range(self.feed.count())
                if isinstance(self.feed.itemAt(i).widget(), Bubble)]

    def _rewind_to(self, index: int) -> bool:
        """Drop every message from `index` on, and the bubbles showing them.

        Both regenerating an answer and editing a prompt mean "go back to this
        point and take a different path", so they share this.
        """
        live = self._current_gen()
        if live and still_running(live["worker"]):
            self._notice("Wait for the current reply to finish, or press Stop.")
            return False
        if not 0 <= index < len(self.messages):
            return False
        # Editing the opening question would otherwise leave the sidebar showing
        # the old one. Only drop the stored title if it is still the auto one —
        # a title the user typed themselves stays put.
        path = self.chats_dir / f"{self.current_chat_id}.json"
        if index == 0 and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("title", "") == auto_title(self.messages):
                    data["title"] = ""
                    _atomic_write_json(path, data)
            except Exception:                                      # noqa: BLE001
                pass

        # Held so a regenerate that produces nothing can put them back.
        self._removed_turns = self.messages[index:]
        del self.messages[index:]
        # Rebuild rather than pluck bubbles: tool and web cards carry no message
        # index, and removing only Bubbles left them orphaned on screen.
        self._rebuild_feed()
        self._save_current_chat()
        self._reload_chat_list()
        return True

    def _regenerate(self, bubble: Bubble):
        """Ask the same question again, discarding this answer — but keep it
        recoverable until the replacement actually arrives."""
        index = bubble.msg_index
        if index is None or not self._rewind_to(index):
            return
        self.tool_round = 0
        self._begin_generation("Regenerating…",
                               restore=getattr(self, "_removed_turns", None))

    def _edit_prompt(self, bubble: Bubble):
        """Load a sent prompt back into the composer instead of asking anew."""
        index = bubble.msg_index
        if index is None:
            return
        text = self.messages[index].get("content", "") if index < len(self.messages) else ""
        if not self._rewind_to(index):
            return
        self.input.setPlainText(text)
        self.input.moveCursor(QTextCursor.MoveOperation.End)
        self.input.setFocus()
        if not self.messages:
            self.hint.show()

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

    def _refresh_busy_chats(self):
        """Spin the green marker on every chat that is still generating — there
        can be more than one, since replies run in parallel."""
        for i in range(self.chat_list.count()):
            row = self.chat_list.itemWidget(self.chat_list.item(i))
            if not isinstance(row, ListRow):
                continue
            if row.key in self.gens:
                row.spinner.start()
            else:
                row.spinner.stop()

    def _notice_for_chat(self, chat_id: str, err: str):
        """A reply failed in a chat the user is not looking at — record it there
        so the message is not lost, and say so if they are nearby."""
        path = self.chats_dir / f"{chat_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("messages", []).append(
                {"role": "assistant", "content": f"⚠ {err}"})
            _atomic_write_json(path, data)
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

    def _later(self, ms: int, fn):
        """A delayed call that dies with the window. singleShot with a lambda
        has no receiver, so it outlives a destroyed widget and aborts."""
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(fn)
        timer.start(ms)

    def _scroll_down(self):
        self._later(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

    # -- generations -------------------------------------------------------
    # Keyed by chat id. Each entry is {worker, buffer, messages, flushed}. A
    # single slot meant starting a reply anywhere cancelled the one already
    # running, so a chat you had walked away from never got its answer.
    def _gen(self, chat_id):
        return self.gens.get(chat_id)

    def _current_gen(self):
        return self.gens.get(self.current_chat_id)

    def _set_generating(self, busy: bool, what: str = "Generating…"):
        """One switch for the whole busy state of the composer."""
        self.send_btn.setText("Stop" if busy else "Send")
        self.send_btn.setObjectName("danger" if busy else "primary")
        self.send_btn.setStyleSheet("")          # forces a restyle from the QSS
        if busy:
            self.busy.start(what)
        else:
            self.busy.stop()

    def _send_or_stop(self):
        live = self._current_gen()
        if live and still_running(live["worker"]):
            live["worker"].stop()
            return
        text = self.input.toPlainText().strip()
        if not text:
            return

        if not self._have_model():
            return

        self.hint.hide()
        user = self._add_bubble("user", index=len(self.messages))
        user.set_text(text)
        self.messages.append({"role": "user", "content": text})
        self.input.clear()
        self.tool_round = 0
        self._save_current_chat()
        self._begin_generation()

    def _have_model(self) -> bool:
        # Without this the request goes out as {"model": ""} and Ollama answers
        # 400 "model is required" — technically accurate, useless to a newcomer
        # who simply has not imported a model yet.
        if self.current_model:
            return True
        self._notice(
            "No model is selected yet. Open the Models page, choose "
            "“Add model…” and pick a .gguf file — CriGent imports it for you. "
            "It will then appear in the selector at the top.")
        return False

    def _begin_generation(self, what: str = "Generating…", restore=None):
        """Answer the conversation as it currently stands. `restore` holds
        turns a rewind removed, returned to the chat if this attempt fails
        before producing anything."""
        if not self._have_model():
            return
        self._busy_text = what
        self.gens[self.current_chat_id] = {
            "worker": None,
            "buffer": "",
            "flushed": "",
            "messages": self.messages,      # same list object, keeps growing
            "restore": restore or [],
        }
        self._start_worker(self.current_chat_id)

    def _start_worker(self, chat_id: str):
        gen = self.gens.get(chat_id)
        if gen is None:
            return
        gen["buffer"] = ""
        gen["flushed"] = ""

        if chat_id == self.current_chat_id:
            _, badge = model_label(self.current_model)
            self.bubble = self._add_bubble("assistant", badge)
            self.bubble.set_text("")
            self._scroll_down()
            self._set_generating(True, getattr(self, "_busy_text", "Generating…"))
            self.tokens_lbl.setText("")     # the spinner already says this
        self._refresh_busy_chats()

        # Each enrichment prompt is opt-in via its toggle, and comes from the
        # user-editable store rather than the module constants.
        sys_parts = []
        if (self.tools_check.isChecked() or self.web_check.isChecked()
                or self.skills_check.isChecked()):
            # Goes first: without it, models answer "I can't do that".
            sys_parts.append(self.prompts.get("agent", ""))
        if self.tools_check.isChecked():
            sys_parts.append(self.prompts.get("tools", ""))
        if self.web_check.isChecked():
            sys_parts.append(self.prompts.get("web", ""))
        if self.skills_check.isChecked():
            sys_parts.append(self.prompts.get("skills", ""))
        active_skills = self._active_skills_text()
        if active_skills:
            sys_parts.append(active_skills)
        # Layout applies to every reply, so it is not behind a toggle. An empty
        # box is how you turn it off, and the filter below honours that.
        sys_parts.append(self.prompts.get("format", ""))
        sys_parts = [p for p in sys_parts if p.strip()]

        payload = list(gen["messages"])
        if sys_parts:
            payload = [{"role": "system", "content": "\n\n".join(sys_parts)}] + payload

        # Deliberately does not touch other chats' workers. Replies run in
        # parallel, one per conversation.
        worker = self._track(ChatWorker(payload, self.current_model, self._num_gpu()))
        gen["worker"] = worker
        self.chat_worker = worker            # the most recent, for close-down
        worker.chunk.connect(lambda t, c=chat_id: self._on_chunk(c, t))
        worker.done.connect(
            lambda e, n, p, c=chat_id: self._on_done(c, e, n, p))
        worker.failed.connect(lambda m, c=chat_id: self._on_failed(c, m))
        worker.notice.connect(self._notice)
        worker.start()
        self.flush.start()

    def _on_chunk(self, chat_id: str, piece: str):
        gen = self.gens.get(chat_id)
        if gen is not None:
            gen["buffer"] += piece

    def _flush(self):
        # Only the visible chat is drawn. Other conversations keep accumulating
        # into their own buffers and are shown when the user goes back to them.
        gen = self._current_gen()
        if gen is None or not self.bubble:
            return
        if gen["buffer"] == gen["flushed"]:
            return                     # nothing new since the last tick
        gen["flushed"] = gen["buffer"]
        stick = self._at_bottom()
        self.bubble.set_text(gen["buffer"])
        if stick:
            self._scroll_down()

    def _finish_ui(self, chat_id: str):
        if not self.gens:
            self.flush.stop()
        if chat_id == self.current_chat_id:
            self._flush()
            self._set_generating(False)
            self.tokens_lbl.setText("")
        self._refresh_busy_chats()

    def _on_done(self, chat_id: str, elapsed: float, tokens: int,
                 prompt_tokens: int = 0):
        gen = self.gens.get(chat_id)
        if gen is None:
            return
        text = gen["buffer"]
        visible = chat_id == self.current_chat_id
        if visible:
            self._flush()

        if text:
            gen["messages"].append({"role": "assistant", "content": text})
        self._record_usage(tokens, prompt_tokens, elapsed)
        rate = tokens / elapsed if elapsed > 0 else 0
        stamp = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"
        if visible and self.bubble:
            self.bubble.set_meta(
                f"⏱ {stamp}  ·  {rate:.1f} tok/s  ·  "
                f"{prompt_tokens + tokens:,} tokens")
            # The turn is settled, so it can now be copied or regenerated.
            if text:
                self.bubble.set_index(len(gen["messages"]) - 1)
                self.bubble.regenerate.connect(
                    lambda w=self.bubble: self._regenerate(w))
            self.bubble = None
        # Always written to the chat that asked, which may not be on screen.
        self._save_chat(chat_id, gen["messages"])
        self.gens.pop(chat_id, None)
        self._finish_ui(chat_id)

        if visible:
            self._hint_if_refused(text)

        if self.tool_round < MAX_TOOL_ROUNDS:
            # Scan only the answer. A command the model merely thought about
            # while reasoning must never execute — with Auto-run on, matching
            # against the full buffer would run it.
            acted = strip_reasoning(text)
            command = extract_action(acted, "run") if self.tools_check.isChecked() else None
            query = extract_action(acted, "search") if self.web_check.isChecked() else None
            url = extract_action(acted, "fetch") if self.web_check.isChecked() else None

            if command and command.strip():
                self.tool_round += 1
                self._offer_tool(chat_id, command.strip())
            elif query and query.strip():
                self.tool_round += 1
                self._start_search(chat_id, query.strip())
            elif url and url.strip():
                self.tool_round += 1
                self._start_fetch(chat_id, url.strip())
            else:
                body = (extract_action(acted, "skill")
                        if self.skills_check.isChecked() else None)
                parsed = parse_skill_block(body) if body else None
                if parsed:
                    self.tool_round += 1
                    self._offer_skill(chat_id, *parsed)
                else:
                    self._nudge_if_stranded(chat_id, text)

    def _nudge_if_stranded(self, chat_id: str, text: str):
        """Rescue a turn whose only action block was left inside the thinking.

        Nothing inside <reasoning> is ever executed — a command the model merely
        considered must not run. But the model does not know its block was
        ignored: it stops and waits for output that will never arrive, and the
        conversation is stuck until the user notices. Telling it what happened
        costs one round and it re-sends the block properly.
        """
        think = reasoning_text(text)
        if not think.strip():
            return
        for tag, on in (("run", self.tools_check.isChecked()),
                        ("search", self.web_check.isChecked()),
                        ("fetch", self.web_check.isChecked()),
                        ("skill", self.skills_check.isChecked())):
            body = extract_action(think, tag) if on else None
            if body and body.strip():
                self.tool_round += 1
                self._continue(chat_id, (
                    f"[CriGent] Your `{tag}` block was inside your reasoning, so it was "
                    f"not carried out — nothing inside <reasoning> is ever executed, and "
                    f"you will never receive output for it. Close the reasoning tag first, "
                    f"then put the block on its own at the very end of your reply:\n\n"
                    f"```{tag}\n{body.strip().splitlines()[0]}\n```"))
                return

    def _on_failed(self, chat_id: str, err: str):
        gen = self.gens.get(chat_id)

        # A regenerate that produced nothing must not cost the old answer:
        # put the turns the rewind removed straight back.
        restored = False
        if gen and gen.get("restore") and not gen["buffer"]:
            gen["messages"].extend(gen["restore"])
            restored = True

        if chat_id == self.current_chat_id and self.bubble:
            if restored:
                # The reload below rebuilds the feed from the restored messages.
                self.bubble = None
            elif gen and gen["buffer"]:
                # Keep what did arrive instead of wiping it with an error box.
                self._flush()
                self.bubble.set_meta("⚠ interrupted — the reply above is partial")
                self.bubble.offer_retry()
                self.bubble.regenerate.connect(
                    lambda w=self.bubble: self._retry_turn(w))
                self._notice(err)
                self.bubble = None
            else:
                self.bubble.set_error(err)
                self.bubble.offer_retry()
                self.bubble.regenerate.connect(
                    lambda w=self.bubble: self._retry_turn(w))
                self.bubble = None
        elif gen is not None:
            self._notice_for_chat(chat_id, err)
        if gen is not None:
            self._save_chat(chat_id, gen["messages"])
        self.gens.pop(chat_id, None)
        self._finish_ui(chat_id)
        self._save_current_chat()
        if restored and chat_id == self.current_chat_id:
            # Rebuild, not _load_chat: reload would reset the capability
            # toggles and kill an agent run the user is in the middle of.
            self._rebuild_feed()
            self._scroll_down()
            self._notice(f"{err}\n\nNothing was changed — the previous answer "
                         f"was kept.")

    def _retry_turn(self, bubble: Bubble):
        """Try the last turn again after a failure. The failed bubble goes; the
        conversation itself was never advanced, so generating just works."""
        if self._current_gen() is not None:
            return
        self.feed.removeWidget(bubble)
        bubble.setParent(None)
        bubble.deleteLater()
        self._begin_generation("Retrying…")

    # "I'm unable to browse the web" and friends. Cheap to detect, and worth
    # detecting: the usual cause is simply that the capability is switched off.
    REFUSAL_RE = re.compile(
        r"(can'?t|cannot|unable to|don'?t have|do not have|no ability to|"
        r"not able to)[^.\n]{0,60}"
        r"(search|browse|internet|web|online|live|real[- ]time|fetch|url|link)",
        re.I)
    CMD_REFUSAL_RE = re.compile(
        r"(can'?t|cannot|unable to|don'?t have|not able to)[^.\n]{0,60}"
        r"(run|execute|command|powershell|terminal|shell|your (?:computer|machine|system))",
        re.I)

    def _hint_if_refused(self, text: str):
        """If the model said it couldn't do something the app can do, explain why."""
        if not text:
            return
        if self.REFUSAL_RE.search(text) and not self.web_check.isChecked():
            self._notice("The model said it cannot search or open links — that is because "
                         "the Web toggle is off, so it genuinely has no web access right "
                         "now. Switch on “Web” below the message box and ask again.")
            return
        if self.CMD_REFUSAL_RE.search(text) and not self.tools_check.isChecked():
            self._notice("The model said it cannot run commands — the Tools toggle is off, "
                         "so it has no way to. Switch on “Tools” below the message box and "
                         "ask again.")

    def _remember_toggle(self, key: str, on: bool):
        self.settings[key] = bool(on)
        self._save_settings()

    def _on_tools_toggled(self, checked: bool):
        self.autorun_check.setEnabled(checked)
        if not checked:
            self.autorun_check.setChecked(False)

    def _continue(self, chat_id: str, text: str):
        """Feed a tool/search/skill result back into the chat that asked, and
        carry on generating there — even if the user has moved elsewhere."""
        gen = self.gens.get(chat_id)
        messages = gen["messages"] if gen else (
            self.messages if chat_id == self.current_chat_id else None)
        if messages is None:
            return
        messages.append({"role": "user", "content": text})
        if gen is None:
            self.gens[chat_id] = {"worker": None, "buffer": "",
                                  "flushed": "", "messages": messages}
        self._save_chat(chat_id, messages)
        if chat_id == self.current_chat_id:
            self._scroll_down()
        self._start_worker(chat_id)

    def _offer_tool(self, chat_id: str, command: str):
        auto = self.autorun_check.isChecked()
        visible = chat_id == self.current_chat_id
        if not visible and not auto:
            # The card used to land in whichever chat was on screen — the wrong
            # one — and vanish when its own chat was reopened. Held instead,
            # and offered when that chat is next opened.
            self.pending_actions[chat_id] = ("tool", command)
            return
        card = ToolCard(command, self.mono, auto=auto)
        if visible:
            self.feed.insertWidget(self.feed.count() - 1, card)
            self._scroll_down()
        if auto:
            self._run_tool(chat_id, card, command)
        else:
            card.run_clicked.connect(
                lambda c=card, cmd=command, k=chat_id: self._run_tool(k, c, cmd))
            card.deny_clicked.connect(
                lambda c=card, cmd=command, k=chat_id: self._deny_tool(k, c, cmd))

    def _run_tool(self, chat_id: str, card: ToolCard, command: str):
        worker = CommandWorker(command)
        worker.result.connect(
            lambda out, err, code, c=card, cmd=command, k=chat_id:
            self._tool_result(k, c, cmd, out, err, code))
        self._cmd_worker = self._track(worker)
        worker.start()

    def _tool_result(self, chat_id: str, card: ToolCard, command: str,
                     stdout: str, stderr: str, code: int):
        card.show_result(stdout, stderr, code)
        summary = f"$ {command}\nexit code: {code}"
        if stdout.strip():
            summary += f"\n\nstdout:\n{stdout[:4000]}"
        if stderr.strip():
            summary += f"\n\nstderr:\n{stderr[:4000]}"
        self._continue(chat_id, f"[Tool result]\n{summary}")

    def _deny_tool(self, chat_id: str, card: ToolCard, command: str):
        self._continue(chat_id, f"[User denied running this command: {command}]")

    def _start_search(self, chat_id: str, query: str):
        card = WebCard("search", query, self.mono)
        if chat_id == self.current_chat_id:
            self.feed.insertWidget(self.feed.count() - 1, card)
            self._scroll_down()
        worker = SearchWorker(query)
        worker.result.connect(
            lambda results, err, c=card, q=query, k=chat_id:
            self._search_result(k, c, q, results, err))
        self._cmd_worker = self._track(worker)
        worker.start()

    def _search_result(self, chat_id: str, card: WebCard, query: str,
                       results: list, err: str):
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
        self._continue(chat_id, f"[Web search result]\n{summary}")

    def _start_fetch(self, chat_id: str, url: str):
        card = WebCard("fetch", url, self.mono)
        if chat_id == self.current_chat_id:
            self.feed.insertWidget(self.feed.count() - 1, card)
            self._scroll_down()
        worker = FetchWorker(url)
        worker.result.connect(
            lambda text, err, c=card, u=url, k=chat_id: self._fetch_result(k, c, u, text, err))
        self._cmd_worker = self._track(worker)
        worker.start()

    def _fetch_result(self, chat_id: str, card: WebCard, url: str, text: str, err: str):
        if err:
            card.show_result(f"Error: {err}", ok=False)
            summary = f"Fetching {url} failed: {err}"
        else:
            card.show_result(text)
            summary = f"Content fetched from {url}:\n\n{text[:4000]}"
        self._continue(chat_id, f"[Web fetch result]\n{summary}")

    def _offer_skill(self, chat_id: str, name: str, content: str):
        if chat_id != self.current_chat_id:
            self.pending_actions[chat_id] = ("skill", (name, content))
            return
        card = SkillCard(name, content, self.mono)
        card.save_clicked.connect(
            lambda c=card, n=name, ct=content, k=chat_id:
            self._save_skill_from_card(k, c, n, ct))
        card.discard_clicked.connect(
            lambda c=card, n=name, k=chat_id: self._discard_skill_card(k, c, n))
        self.feed.insertWidget(self.feed.count() - 1, card)
        self._scroll_down()

    def _save_skill_from_card(self, chat_id: str, card: SkillCard, name: str, content: str):
        self.skills.append({
            "id": uuid.uuid4().hex[:10], "name": name, "content": content,
            "created": time.time(), "updated": time.time(),
        })
        self._save_skills()
        self._reload_skill_list()
        self._reload_skill_page()
        self._continue(chat_id, f"[Skill saved: {name}]")

    def _discard_skill_card(self, chat_id: str, card: SkillCard, name: str):
        self._continue(chat_id, f"[User discarded proposed skill: {name}]")

    # -- chat history ------------------------------------------------------ #
    def _new_chat(self):
        self.bubble = None          # generation continues in its own chat
        self._set_generating(False)
        while self.feed.count() > 2:                 # keep hint + stretch
            item = self.feed.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        # Rebind rather than clear(): a running reply holds this exact list as
        # gen_messages, and emptying it would delete the question it answers.
        self.messages = []
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
        self._save_chat(self.current_chat_id, self.messages)

    def _save_chat(self, chat_id: str, messages: list):
        """Persist any chat by id — the one on screen may not be the one that
        just finished generating."""
        if not messages:
            return
        path = self.chats_dir / f"{chat_id}.json"
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
            title = auto_title(messages)
        data = {
            "id": chat_id, "title": title, "model": self.current_model,
            "created": created, "updated": time.time(), "messages": messages,
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
            if chat_id in self.gens:
                row.spinner.start()
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

        # Deliberately does NOT stop the worker: a reply keeps running in the
        # chat it belongs to while the user reads or writes somewhere else.
        self.bubble = None
        while self.feed.count() > 2:
            item = self.feed.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        self.messages = data.get("messages", [])
        self.current_chat_id = data.get("id", chat_id)
        # self.buffer is owned by the in-flight reply; clearing it here would
        # throw away text that has already streamed in for another chat.
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

        self._rebuild_feed()

        # A command proposed while this chat was off screen is offered now,
        # instead of having appeared inside whatever chat was open at the time.
        pend = self.pending_actions.pop(self.current_chat_id, None)
        if pend is not None:
            kind, payload = pend
            if kind == "tool":
                self._offer_tool(self.current_chat_id, payload)
            elif kind == "skill":
                self._offer_skill(self.current_chat_id, *payload)

        # If this chat is the one still generating, re-attach so the rest of the
        # reply keeps streaming into view instead of arriving invisibly.
        self._set_generating(False)
        gen = self.gens.get(self.current_chat_id)
        if gen is not None:
            _, badge = model_label(self.current_model)
            self.bubble = self._add_bubble("assistant", badge)
            gen["flushed"] = ""
            self.bubble.set_text(gen["buffer"] or "…")
            self.hint.hide()
            self._set_generating(True)

        self._scroll_down()
        self._reload_chat_list()

    def _rebuild_feed(self):
        """Redraw the feed from self.messages. Unlike _load_chat this leaves
        the capability toggles alone, so a rewind mid-agent-run does not
        silently switch the tools off. Also drops any orphaned tool/web cards,
        which are not messages and would otherwise survive the rewind."""
        self.bubble = None
        while self.feed.count() > 2:
            item = self.feed.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        if self.messages:
            self.hint.hide()
            for i, m in enumerate(self.messages):
                role = m.get("role", "user")
                if role == "user":
                    b = self._add_bubble("user", index=i)
                else:
                    _, badge = model_label(self.current_model)
                    b = self._add_bubble("assistant", badge, index=i)
                b.set_text(m.get("content", ""))
        else:
            self.hint.show()

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
        self.pending_actions.pop(chat_id, None)
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
                self._reload_skill_page()

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
                self._reload_skill_page()

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
        if getattr(self, "editing_skill_id", None) == skill_id:
            # Drop the editor's claim first, or reloading the page would try to
            # reselect a skill that no longer exists.
            self.editing_skill_id = None
        self._reload_skill_list()
        self._reload_skill_page()

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
                self._later(1500, lambda: self._recheck(tries - 1))
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
    def _show_ram(self, s: dict):
        if not s.get("ram_total"):
            return
        self.g_ram.set_value(s["ram_pct"],
                             f"{s['ram_used']:.1f} / {s['ram_total']:.1f} GB")
        self.spark_ram.push(s["ram_pct"])

    def _ram_rows_html(self, s: dict) -> str:
        if not s.get("ram_total"):
            return ""
        rows = [("System RAM", f"{s['ram_used']:.1f} / {s['ram_total']:.1f} GB "
                               f"({s['ram_pct']:.0f}%)"),
                ("Commit charge", f"{s['swap_used']:.1f} / {s['swap_total']:.1f} GB")]
        body = "".join(
            f'<tr><td style="color:{C["dim"]};padding:3px 26px 3px 0;">{k}</td>'
            f'<td style="color:{C["text"]};font-family:{self.mono};">{v}</td></tr>'
            for k, v in rows)
        return f"<table cellspacing='0'>{body}</table>"

    def _on_gpu(self, s: dict):
        # RAM comes from the OS, not nvidia-smi, so it keeps updating on a
        # machine with no NVIDIA card — the page is Track, not GPU.
        self._show_ram(s)
        if "error" in s:
            self.gpu_name.setText("GPU unavailable")
            self.detail.setText(
                f'<span style="color:{C["red"]}">{_html.escape(s["error"])}</span>'
                + self._ram_rows_html(s))
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
            ("VRAM", f"{s['mem_used']:.0f} / {s['mem_total']:.0f} MiB  ({mem_pct:.1f}%)"),
            ("Memory bus", f"{s['mem_util']:.0f}% busy"),
            ("SM clock", f"{s['clock']:.0f} MHz  (max {s['clock_max']:.0f})"),
            ("Board power", f"{s['power']:.1f} W" +
             (f" / {s['power_max']:.0f} W" if s["power_max"] else "")),
            ("Temperature", f"{s['temp']:.0f} °C"),
        ]
        if s.get("ram_total"):
            rows += [
                ("System RAM", f"{s['ram_used']:.1f} / {s['ram_total']:.1f} GB "
                               f"({s['ram_pct']:.0f}%)"),
                ("Commit charge", f"{s['swap_used']:.1f} / {s['swap_total']:.1f} GB"),
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
        /* reasoning: collapsed to its newest line, expandable to the full trace */
        #reasonPanel {{ background:{C['panel_hi']}; border:1px solid {C['line']};
                        border-left:3px solid {C['faint']}; border-radius:10px; }}
        #reasonToggle {{ background:transparent; border:none; color:{C['dim']};
                         font-size:12px; font-weight:600; text-align:left; padding:0; }}
        #reasonToggle:hover {{ color:{C['text']}; }}
        #reasonCount {{ color:{C['faint']}; font-size:11px; }}
        #reasonLatest {{ color:{C['faint']}; font-size:12px; font-style:italic; }}
        #reasonFull {{ background:{C['bg']}; border:1px solid {C['line']};
                       border-radius:8px; color:{C['dim']};
                       font-family:'{self.mono}'; font-size:{CODE_PX_SM}px;
                       padding:8px 10px; }}

        /* instructions and maths get their own titled containers */
        #instrBlock {{ background:{C['panel_hi']}; border:1px solid {C['line']};
                       border-left:3px solid {C['accent']}; border-radius:10px; }}
        #instrTitle {{ color:{C['accent']}; font-size:11px; font-weight:700;
                       letter-spacing:0.6px; }}
        #crashGroup {{ background:{C['panel']}; border:1px solid {C['line']};
                       border-left:3px solid {C['red']}; border-radius:10px; }}
        #crashToggle {{ background:transparent; border:none; color:{C['text']};
                        font-size:13px; font-weight:600; text-align:left; padding:0; }}
        #crashToggle:hover {{ color:{C['accent_hi']}; }}
        #crashCount {{ color:{C['red']}; font-size:11px; font-weight:700; }}
        #crashWhen {{ color:{C['faint']}; font-size:11px; }}
        #crashLatest {{ color:{C['dim']}; font-size:12px; }}
        #crashDay {{ background:{C['panel_hi']}; border:1px solid {C['line']};
                     border-radius:8px; }}
        #crashDayToggle {{ background:transparent; border:none; color:{C['dim']};
                           font-size:12px; text-align:left; padding:0; }}
        #crashDayToggle:hover {{ color:{C['text']}; }}
        #crashMsg {{ color:{C['text']}; font-size:12px; }}
        #statBox {{ background:{C['panel_hi']}; border:1px solid {C['line']};
                    border-radius:10px; }}
        #statCap {{ color:{C['faint']}; font-size:11px; font-weight:600;
                    letter-spacing:0.5px; }}
        #statValue {{ color:{C['text']}; font-size:26px; font-weight:600; }}
        #statSub {{ color:{C['faint']}; font-size:11px; }}
        #tasksBlock {{ background:{C['panel_hi']}; border:1px solid {C['line']};
                       border-left:3px solid {C['green']}; border-radius:10px; }}
        #tasksTitle {{ color:{C['green']}; font-size:11px; font-weight:700;
                       letter-spacing:0.6px; }}
        #mathBlock {{ background:{C['panel_hi']}; border:1px solid {C['line']};
                      border-left:3px solid {C['violet']}; border-radius:10px; }}
        #mathTitle {{ color:{C['violet']}; font-size:11px; font-weight:700;
                      letter-spacing:0.6px; }}
        #blockBody {{ background:transparent; color:{C['text']}; line-height:155%; }}
        /* per-message actions: quiet until you go looking for them */
        #msgAction {{ background:transparent; border:1px solid transparent;
                      border-radius:7px; color:{C['faint']}; font-size:11px;
                      padding:3px 9px; }}
        #msgAction:hover {{ background:{C['panel_hi']}; border-color:{C['line']};
                            color:{C['text']}; }}
        #busyStrip {{ background:{C['panel_hi']}; border:1px solid {C['line']};
                      border-radius:12px; }}
        #busyLabel {{ color:{C['dim']}; font-size:11px; font-weight:600; }}
        #mathBody {{ background:transparent; color:{C['text']};
                     font-family:'{self.mono}'; font-size:{CODE_PX}px; line-height:165%; }}
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

        QListWidget#chatList, QListWidget#skillList, QListWidget#plainList {{
            background:transparent; border:none; outline:none; }}
        QListWidget#chatList::item, QListWidget#skillList::item,
        QListWidget#plainList::item {{
            color:{C['dim']}; padding:8px 10px; border-radius:8px; }}
        QListWidget#chatList::item:hover, QListWidget#skillList::item:hover,
        QListWidget#plainList::item:hover {{
            background:{C['panel_hi']}; color:{C['text']}; }}
        QListWidget#chatList::item:selected, QListWidget#plainList::item:selected {{
            background:{C['overlay']}; color:{C['text']}; }}
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
