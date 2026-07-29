# CriGent

**A local AI agent for Windows.** Chat with models that run entirely on your own
machine, let the agent run commands and search the web with your approval, save
reusable skills, watch your GPU live, and import new models from a `.gguf` file
without touching a terminal.

No account, no API key, no telemetry. Nothing leaves the machine unless you
switch web search on.

![CriGent](docs/screenshot.png)

---

## Download

Grab `CriGent.exe` from the [latest release](../../releases/latest) and run it.
It is a single portable file — no installer, no admin rights.

## First run

CriGent needs the [Ollama](https://ollama.com) runtime to load models. On first
launch it checks, in order:

1. **Is Ollama already running?** → uses it, nothing to download.
2. **Is `ollama.exe` installed somewhere normal?** → uses it.
3. **Neither?** → you choose: point CriGent at your own `ollama.exe`, or let it
   fetch the official build.

Nothing is ever downloaded without you clicking **Install**.

## Where your data lives

Everything goes in a `CriGent-data` folder **next to the exe**. Put CriGent on a
`D:` drive or a USB stick and the runtime, your models and your chats all stay on
that drive, off the system disk. If the exe sits somewhere unwritable
(Program Files, a read-only share) it falls back to `%LOCALAPPDATA%\CriGent`.

```
CriGent-data/
  ollama/         the runtime, only if CriGent installed it
  models/         your models — tens of GB
  chats/          one JSON file per conversation
  skills.json     your saved skills
  prompts.json    your edited enrichment prompts
  settings.json   runtime path, model folder, compute mode
```

The setup screen shows the folder it will use and how much space is free, and
lets you change it **before** anything downloads.

Already have an Ollama model store? Don't re-import — **Models → Model folder…**
points CriGent at the existing one.

---

## What it does

| | |
|---|---|
| **Chat** | Streamed replies, real code blocks with a copy button, per-message timing and tokens/sec. Conversations are saved automatically and listed in the sidebar. |
| **Models** | See everything installed with size and quantisation, delete with one click, or add any `.gguf` — CriGent writes the Modelfile and imports it for you. Switch models mid-conversation. |
| **Tools** | The model can propose a PowerShell command. You see the exact command and click **Run** or **Deny**; its output is fed back so it can continue. |
| **Web** | Search and read pages when it needs information beyond its training data. Read-only, and every search is shown in the chat. |
| **Skills** | Reusable instructions you tick on and off per message. Write them yourself, or ask the model to draft one — you approve before it is saved. |
| **Prompts** | The enrichment prompts behind Tools, Web and Skills are yours to edit. |
| **GPU** | Live load, VRAM, temperature, power, clocks and per-process usage. |
| **Compute** | Force GPU (CUDA), force CPU, or leave it to Ollama. |

### ⚠️ About Auto-run

**Tools** asks permission for every command. **Auto-run** removes that step —
commands execute the moment the model proposes them, with no review.

Local models are not safety-trained the way hosted assistants are, and if the
agent reads content you did not write (a file, a web page, pasted text), a hidden
instruction inside it can be acted on like a genuine request. Enable Auto-run
only when you are confident about what you are asking for.

---

## Running from source

```bash
git clone https://github.com/AbdulazizAljumaia/CriGent.git
cd CriGent
pip install PyQt6 requests beautifulsoup4 ddgs
python crigent.py
```

Build the portable exe:

```bash
pip install pyinstaller
python -m PyInstaller CriGent.spec --noconfirm
```

Regenerate the icon after changing `paint_logo()`:

```bash
python make_icon.py
```

---

## Licence

CriGent is licensed under the **GNU General Public License v3.0** — see
[`LICENSE`](LICENSE). This is required: CriGent's interface uses PyQt6, which is
GPL-3.0-only.

The author additionally asks that CriGent be used only in ways consistent with
Islamic principles. That request is set out in [`COVENANT.md`](COVENANT.md), and
it is stated there plainly as a matter of conscience rather than a licence
condition, because the GPL does not permit added use restrictions.

---

## Author

**Abdulaziz Al Jumaia**
[crimsonlingua.com](https://crimsonlingua.com) ·
[LinkedIn](https://sa.linkedin.com/in/abdulaziz-al-jumaia)
