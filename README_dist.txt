CriGent — a local AI agent
Developer: Abdulaziz Al Jumaia  ·  crimsonlingua.com

WHAT THIS IS
  A single portable program. There is nothing to install and no admin rights
  are needed. Copy CriGent.exe anywhere — Desktop, a USB stick, another PC —
  and double-click it.

FIRST RUN
  CriGent needs the Ollama runtime to actually load models. On first launch it
  checks, in this order:
     1. Is Ollama already running on this machine?  -> uses it, nothing to do.
     2. Is ollama.exe installed somewhere normal?   -> uses it.
     3. Neither?  -> you choose:
          "I already have Ollama..."  point it at your own ollama.exe
          "Install"                   downloads the official build (~1.46 GB)
                                      into your CriGent folder
  Nothing is ever downloaded without you clicking Install.

WHERE EVERYTHING IS STORED
  By default CriGent keeps everything in a "CriGent-data" folder NEXT TO
  CriGent.exe. Put the exe on a D: drive or a USB stick and all of it -- the
  Ollama runtime, your models, your chats -- stays together on that drive.

     CriGent-data\
        ollama\         the Ollama runtime (only if CriGent installed it)
        models\         your models -- this is the big one, tens of GB
        chats\          one JSON file per conversation
        skills.json     your saved skills
        prompts.json    your edited enrichment prompts
        settings.json   your choices (runtime path, model folder, compute mode)

  If the exe sits somewhere unwritable (Program Files, a read-only share) it
  falls back to %LOCALAPPDATA%\CriGent instead.

  The setup screen shows the folder it is about to use and how much space is
  free on that drive, and lets you change it with "Change folder..." BEFORE
  anything is downloaded. Models are large -- point it at a drive with room.

  Delete the data folder to reset CriGent to a completely fresh state.

ALREADY HAVE MODELS?
  Models are large. If you already have an Ollama model store, do NOT re-import
  them — open the Models page and use "Model folder..." to point CriGent at the
  existing folder (the one containing a "blobs" subfolder), then restart.

ADDING A MODEL
  Models page -> "Add model..." -> pick any .gguf file. CriGent writes the
  Modelfile and imports it for you. Large models take a few minutes and need
  roughly as much free disk space as the file itself.

GPU / CPU
  The selector in the top bar (next to the model list) forces GPU, forces CPU,
  or leaves it to Ollama. Changing it makes Ollama reload the model, so the
  next reply takes longer to start.

A NOTE ON THE AGENT FEATURES
  "Tools" lets the model propose PowerShell commands. You approve each one
  before it runs. "Auto-run" removes that approval step — with it on, commands
  execute immediately and without review. Only enable it if you trust what you
  are about to ask for.
