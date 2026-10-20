# Splash

<p align="center">
  <img src="frontend/src/assets/splash-logo.jpg" alt="Splash — Prompt Enrichment" width="440" />
</p>

**A local prompt studio with diffusion models, evidence, and reversible decisions.**

Splash helps turn an initial idea into a clearer instruction for an LLM or agent. Each prompt lives in its own project, with an immutable main prompt, an editable working draft, context, suggested edits, and a history of every promotion. A model can propose a change; only you can make it the main prompt.

This release includes real diffusion inference, a persistent MCP tool host, and a complete local project workflow. Model execution has been measured on an RTX 2070, an RTX 5090, and a CPU. **Enrichment quality is still experimental:** models can repeat text, introduce assumptions, or miss semantic constraints. There is no custom-trained Splash checkpoint, and execution success is not a quality certification. See [measured results](#measured-hardware-and-quality) and [remaining model work](MODEL_ROADMAP.md).

## Start here

You need **Python 3.11 or later**, **Node.js 22.12 or later with npm**, and a current browser. Initial setup installs the editor, builds the frontend, and **installs PyTorch and the model adapters automatically** for the detected hardware. It verifies imports and a small device calculation before reporting success. Model-runtime packages can be large; an internet connection is required for packages not already cached. Model checkpoints remain separate downloads. Use `--profile editor` only if you intentionally want to skip local inference dependencies.

From the repository directory on Linux or macOS:

```sh
python3 scripts/setup.py --dev
python3 scripts/start.py
```

On Windows, use PowerShell:

```powershell
py -3 scripts/setup.py --dev
py -3 scripts/start.py
```

Open **http://127.0.0.1:43827**. Stop a foreground server with Ctrl+C in the terminal that started it. The launcher checks the port first: if Splash is already running, it prints the existing URL and exits without starting another server or loading another model. No account or API key is required. Creating projects, editing, imports, history, and merges work before a model is installed. Enrichment and branches become available after a successful model load; there is no automatic mock fallback.

If port 43827 is busy, choose another port. To keep this checkout's data in a specific directory:

```sh
python3 scripts/start.py --port 43828 --data-dir .local-data
```

On Windows, replace `python3` with `py -3`. These launchers always bind to loopback. Splash is a single-user local application, not an authenticated public service.

To install a double-clickable Mac app instead of running from the repository, see [macOS application](#macos-application).

### Stop Splash

The default port is now **43827**. An instance already running on the previous default stays on port 8000 until restarted. Stop that instance with `python3 scripts/stop.py --port 8000`, then run `python3 scripts/start.py` and open `http://127.0.0.1:43827/`. Projects remain in the same application-data directory.

Wait for the draft's saved indicator before stopping. Press **Ctrl+C** in the server terminal, or run the shutdown script from a second terminal:

```sh
python3 scripts/stop.py
```

On Windows, use `py -3 scripts/stop.py`. For a different port or a slower shutdown, use `python3 scripts/stop.py --port 43828 --timeout 60`.

The script targets only your user account's Splash server from this checkout on the selected port. It requests graceful shutdown and waits for the server, inference worker, and tool processes to exit. Saved projects, drafts, and model downloads are preserved. If no matching server is found, it exits without stopping anything; it never kills unrelated Python processes or force-kills a slow shutdown. A timeout or permission failure returns a nonzero exit code and explains what remains pending. Closing a browser tab does not stop the server.

Linux and macOS support the normal launcher and direct Uvicorn commands, including `--reload`. On Windows, start with the updated `scripts/start.py`; console-break delivery requires a shared console session. For servers in a separate Windows console, older launches, or direct Uvicorn launches, use Ctrl+C in the original terminal. Linux shutdown is tested; Windows console handling and macOS remain unverified on those platforms. The script uses the existing `psutil` dependency and falls back to the checkout's `.venv` if needed; it does not load model weights.

### Hardware profiles and dependency checks

Setup checks **the project's `.venv`**, even when another Python environment is active in your terminal. It reports Python and Node versions, pinned app/model packages, model-library import failures, GPU architecture and driver, RAM, free VRAM, and free disk space. A short-lived process runs a small matrix/softmax operation on the selected device family; `--quantize` also tests an NF4 round trip. Checks do not download model weights or change saved projects.

```sh
# Diagnose an existing install without installing or changing anything.
python3 scripts/setup.py --check

# Install the editor and the runtime selected for this machine.
python3 scripts/setup.py

# Fedora / GTX 1050: install the legacy NVIDIA runtime and check Compact capacity.
python3 scripts/setup.py --profile nvidia-legacy --level compact

# RTX 5090: include the NF4 dependency and check Extended capacity.
python3 scripts/setup.py --profile nvidia --level extended --quantize

# Explicitly opt out of PyTorch and model adapters (editor only).
python3 scripts/setup.py --profile editor

# Machine-readable diagnostics, or editor-only validation.
python3 scripts/setup.py --check --json
python3 scripts/setup.py --check --profile editor
```

| Setup profile | Intended hardware | Pinned PyTorch build |
|---|---|---|
| `auto` (default) | Selects the NVIDIA GPU with the most VRAM, otherwise Apple Silicon or CPU | Resolves to a profile below |
| `nvidia-legacy` | Maxwell, Pascal (including GTX 1050), Volta; also usable on supported Turing/Ampere/Ada/Hopper cards | `2.14.1+cu126` from the official CUDA 12.6 index |
| `nvidia` | Turing and newer, including RTX 2070 and RTX 5090 | `2.14.1+cu130` from the official CUDA 13.0 index |
| `cpu` | CPU execution | `2.14.1+cpu` on Linux/Windows; macOS Apple Silicon wheel from PyPI |
| `metal` | macOS on Apple Silicon | `2.14.1` from PyPI; Metal execution must pass |
| `editor` | Projects, editing, history, and research without model inference | No model runtime required |

The bundled NVIDIA installation profiles target Linux/Windows x86-64. Intel macOS has no wheel for the pinned model runtime; the editor remains available there. [PyTorch's legacy GPU guidance](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) explains why Pascal needs the CUDA 12.6 build rather than CUDA 13. Auto-detection cannot identify an NVIDIA card when its driver or `nvidia-smi` is unavailable; the report makes this visible. No system drivers are installed or changed.

`--level compact`, `balanced`, and `extended` select the resource estimates for Qwen3 diffusion, Dream, and LLaDA. They do not select a model in the app or download weights. Add `--quantize` for the pinned bitsandbytes dependency on an NVIDIA profile. Memory/disk estimates are advisory; a passing runtime check is not a guarantee of model fit, speed, or quality. GTX 1050, Windows, and Metal model inference remain unverified on actual hardware.

Normal setup installs the selected runtime into the same `.venv` used by `scripts/start.py`. Missing PyTorch, failed imports, incompatible builds, or a failed final device check make setup exit with code **1**, rather than report a complete installation. `--check` remains read-only and returns **0** only when its checks pass. Use `--profile editor` to install or check only editor dependencies; this is also the supported setup path on platforms without a bundled model runtime. The older `--install-models` flag remains accepted for compatibility but is no longer necessary. A working PyTorch build is kept when it matches the selected profile; CPU checks can also use an existing CUDA build. Installing a different GPU profile can replace the PyTorch wheel inside `.venv`, so stop Splash first. Checkpoints and saved projects are preserved.

### Manual installation

Linux/macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements-dev.txt
npm --prefix frontend ci
npm --prefix frontend run build
.venv/bin/python -m backend.app
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\python.exe -m backend.app
```

The runtime dependencies, including the official `mcp==2.3.0` Python SDK, are pinned in [backend/requirements.txt](backend/requirements.txt). Node dependencies use the committed lockfile. Linux is the tested platform; Windows and macOS commands are provided, but their complete application and accelerator workflows have not been tested on those systems.

## macOS application

Build this on the Mac you want to ship to. An Apple Silicon Mac produces an Apple Silicon app. An Intel Mac produces an Intel app. The result is `dist/Splash.app`, which does not need a separate Python or Node install to run.

Requirements on the build Mac: macOS 13 or later, Python 3.11 or later, the Xcode Command Line Tools (`xcode-select --install`), and network access. Node.js 22.12 or later is needed only when `frontend/dist` is not already built. PyTorch's macOS 2.14.1 wheel requires Apple Silicon and macOS 14.

```sh
python3 scripts/package_macos.py
```

`make package-macos` runs that same script.

Add `--dmg` to also write `dist/Splash-macos.dmg`. Add `--no-torch` for a smaller app that can edit and store projects without the model runtime. Add `--skip-frontend-build` to reuse an existing `frontend/dist`.

Open `dist/Splash.app`, or copy it to `/Applications`. The app starts at loopback port 43827 (or the next free port), opens the studio in the browser, and stays in the Dock. Click the Dock icon to bring the studio back. Quit from the **Splash** menu (Command-Q). A second launch reopens the running studio.

If Gatekeeper blocks an unsigned app the first time, right-click `Splash.app` and choose **Open**. Projects and settings stay in `~/Library/Application Support/Splash`. The log is `~/Library/Logs/Splash/splash.log`.

Apple Silicon builds include PyTorch 2.14.1 and the model adapters. Intel builds do not: PyTorch 2.14.1 has no Intel macOS wheel. Download a checkpoint from **Model & hardware** after launch; checkpoints are not stored inside the app. **Metal remains unverified.** A failed Metal check is an error and does not silently switch to CPU. Choose CPU for the measured fallback.

This packager was added from a Linux checkout and has not been launched on macOS here. Build and open it once on the target Mac before relying on it.

## Install and select a model

The simplest path is `python3 scripts/setup.py`, which includes PyTorch automatically (see the profiles above). For manual installation, choose a PyTorch build appropriate to your operating system and GPU using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the model adapters:

```sh
.venv/bin/python -m pip install -r backend/requirements-models.txt
```

On Windows use `.\.venv\Scripts\python.exe` in place of `.venv/bin/python` in all model commands. The tested Linux environment uses Python 3.14, PyTorch `2.14.1+cu130`, Transformers `4.57.6`, Accelerate `1.15.0`, and optional bitsandbytes `0.50.2`. To reproduce its modern NVIDIA runtime, where matching wheels are available:

```sh
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r backend/requirements-models.txt
.venv/bin/python -m pip install bitsandbytes==0.50.2
```

Use `--profile nvidia-legacy` on the GTX 1050; its CUDA 12.6 build differs from the CUDA 13 build above. For CPU-only use, select the CPU profile and leave NF4 and CPU offload unchecked. For Apple Silicon, use the Metal profile; **Metal remains unverified in this release**, so CPU is an explicit fallback. A failed accelerator check is shown as an error and never changes the selected engine silently.

Open **Model & hardware**:

1. Refresh hardware and dependencies. Splash reports OS, CPU threads, RAM, disk space, package versions, and free/total VRAM. **Free VRAM is current available capacity**; the loaded model and runtime allocations consume part of the card's total memory. Device indices use PCI ordering.
2. Choose an adapter family and checkpoint source: **Recommended**, **Hugging Face**, or **Local directory**. The adapter must match the checkpoint architecture. Allow space for the download, extraction/cache metadata, and runtime memory.
3. Set the context budget and diffusion steps. Enable **NF4** for a compatible CUDA device with bitsandbytes installed. CPU offload is a separate, experimental option, with a default 6 GiB GPU / 24 GiB CPU budget.
4. Click **Load & run check**. The worker loads actual weights and runs a generation. The dialog reports the result, latency, and peak allocated GPU memory. The selected configuration is restored at the next application start.

All projects use this one model; agents supply different instructions and tool permissions. Changing the selected model replaces the worker's loaded model. Hugging Face downloads use pinned commit revisions and reuse partial downloads when retried. Downloaded models live under the application's `models/` directory. Checkpoints from your own directory are used in place.

### Hugging Face and local repositories

For a Hugging Face checkpoint, select **Hugging Face**, enter `owner/model` or `https://huggingface.co/owner/model`, and optionally supply a branch, tag, or commit in **Revision**. A blank revision means `main`. **Check checkpoint source** reads metadata/configuration, shows repository file size where available, validates the architecture, and resolves the requested revision to an immutable commit. **Download from Hugging Face** downloads that commit. Then select your device and click **Load & run check**. Downloading never loads a model automatically.

Custom checkpoints currently need one of the bundled adapter types: `a2d-qwen3` (Compact), `Dream` (Balanced), or `llada2_moe` (Extended). A Hugging Face account or repository existing does not establish compatibility. Arbitrary autoregressive models, GGUF, and other diffusion architectures need additional adapters. Compatible configuration is only a preliminary check; actual loading/generation still has to succeed.

For a local model repository, select **Local directory** and enter its full path, such as `~/models/my-checkpoint` or `C:\models\my-checkpoint`. The folder needs `config.json`, tokenizer files, and actual weights (including every indexed shard). A code-only Git checkout or Git LFS pointer is insufficient; the checker explains which files are missing. Local paths are used in place and no model is uploaded.

The **Downloaded checkpoints & transfers** list preserves completed downloads across restarts and lets you reuse them with **Use this checkpoint**. Custom downloads are stored by repository and resolved commit under `models/hub/`, independently of recommended checkpoints. Accepted edits and synthesized variants record the actual Hugging Face repository/commit; local checkpoints are labeled local rather than attributed to a catalog revision. Failed model loads do not replace the last successfully saved model configuration.

Private/gated models use the Hugging Face credentials already configured on the backend machine (local Hub login or `HF_TOKEN`); Splash does not collect or store tokens in its project database. Obtain access through Hugging Face first. Repository IDs and root repository URLs are accepted; put `/tree/...` revisions in the separate Revision field. Checkpoint-source inspection contacts Hugging Face and caches the small configuration file, but does not execute model code. See [Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/download).

The candidate checkpoints contain custom model code, loaded locally with `trust_remote_code=True`. Bundled downloads pin that code to these revisions; only select an existing checkpoint directory you trust.

| Profile | Checkpoint | Pinned revision | Approximate download | App context ceiling |
|---|---|---|---:|---:|
| Compact | [Qwen3-0.6B diffusion MDLM](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) | `c8d24a3f4adaeef46881b450e1bf7d1005203bd7` | 1.6 GB | 1,024 tokens |
| Balanced | [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | `05334cb9faaf763692dcf9d8737c642be2b2a6ae` | 15 GB | 2,048 tokens |
| Extended | [LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini) | `20e64e2ad21644d0e5248586ed9c942cdd45de0f` | 34 GB | 32,768 tokens |

These are application ceilings, including instructions, input, context, and output reservation; they are not guarantees that the maximum fits every device. NF4 reduces runtime weight memory, not the source checkpoint download. The LLaDA adapter uses its native confidence-based block/editing schedule. **Generation mode → Quality** is the default: LLaDA uses confidence threshold 0.7, editing threshold 0.5, and 16 post-editing steps, following the pinned checkpoint guidance. **Speed** restores thresholds 0.5/0.0. For Dream and Compact, Quality treats the configured diffusion steps as a minimum and reserves at least one step per two output tokens; Speed uses the configured schedule. Actual schedule and token counts are recorded in generation metrics. Quality takes longer, and larger context/output budgets can increase peak memory. Existing configurations without this field use Quality after restart/reload; prior measurements below used the older schedule. Local compatibility changes are documented in [backend/vendor/NOTICE](backend/vendor/NOTICE).

## Choose an agent and create a project

Click **New prompt project**, choose an agent, and enter the initial prompt. A project name is optional. The first main revision preserves the original text exactly; the draft starts as a copy.

| Agent | Use it for | Default output |
|---|---|---|
| **splash** | Clear, precise, readable graduate-level wording for general instructions | Polished prose |
| **splash-code** | Coding objectives, constraints, phases, tests, and acceptance criteria | Structured Markdown |
| **splash-search** | Scholarly research questions, methods, evidence standards, and related directions | Structured research brief |

The agent is fixed for the life of a project. Change the output style under **Context & sources** if you prefer prose. Use sidebar search and **Rename** to organize projects.

Open **System prompts** on a project to add instructions that belong to that project. They travel with the system message when the local model reviews that project's draft, replies to comments, explores its research branches, synthesizes its directions, or proposes a source-informed edit. Another project's model run does not receive them. Each prompt has a name, an instruction of up to 2,000 characters, and a switch for whether it is used. Disabled prompts stay saved on the project. Application rules and the required response format still apply, and these prompts do not promote a main revision. Changing enabled instructions cancels obsolete work, clears pending proposals, and marks old model comments as outdated so the next review uses the new instructions. Your draft, main history, and prior wording decisions stay saved. A project holds up to 12. **Import as new** and a complete project archive each take their own copy.

If a project has no enabled system prompt, the model uses that mode's built-in default: one for **splash**, one for **splash-code**, and one for **splash-search**. An enabled project prompt replaces the default. **Prompt library**, in the sidebar or at the bottom of the system-prompt window, keeps those three defaults plus prompts you save from a project. Pull copies a library prompt onto the open project; later edits in the library do not change projects that already pulled it. **Download** writes one prompt as `splash-system-prompt` JSON, and **Import JSON** adds that file to the library on this computer. The library holds 40 saved prompts besides the built-in defaults.

Input can be multilingual. The model is instructed to enrich in English while retaining code, identifiers, quotations, references, numbers, and constraints. Unsupported details should become questions, not asserted requirements. Enrichment conservatively withholds edits that change recognized explicit English requirements such as “Preserve…”, “Do not…”, and “Use only…”. Branch synthesis retains omitted explicit requirements in a labeled constraints section. Literal/code/number checks and a limited multilingual testing-requirement check reject some other lossy outputs. They do **not** prove complete semantic equivalence; examine negations, scope, and implied obligations in the review.

### Imports and exports

- **Import as new** copies the current main into a new project, with an independently chosen agent.
- **Version history → Import into new project** can begin from any stored revision, including an archived main.
- Context pages and source records are copied, with origin metadata. The new project has its own history and remains usable after the source is deleted.
- **Export → Text/Markdown** exports the approved main, excluding unmerged draft changes.
- **Export → Complete project JSON** includes main history, saved drafts, variants, extracted context, sources, query history, and lineage. Merge-review tokens are excluded.
- **Import project archive** reads a Splash JSON export and starts a new project from its current main with copied context and evidence. It deliberately starts a fresh history; it is not an in-place database restore. Existing local projects are not overwritten. The archive's old history remains available in the exported JSON.

For a full restore of the exact application state, back up the SQLite database as described below.

## Drafts, suggestions, and explicit merges

The **Working Draft** is your editable input space. The **Main Prompt** pane displays the live enriched output as a complete prompt. A **Document comments** rail sits beside the document; on narrower screens, the panes and rail stack vertically. Typing autosaves after a short pause. The main pane reflects your input while generation is pending and updates when the model finishes; its status distinguishes waiting, running, ready, and failed enrichment. Updates depend on your hardware's generation speed, rather than arriving on every keystroke. Disable live enrichment for manual-only refinement.

**Automatic sampling waits for content.** Autosaving a keystroke does not send it to the model. Enrichment, comment review, and research all use the same boundary check:

- A clause or sentence containing at least two words, ended with sentence punctuation, a comma/semicolon, or a newline, becomes eligible after **two seconds without editing**. For example, `Use Python.` or `The budget is $500;` qualifies.
- Without punctuation, finish **at least six words followed by a space** and pause for **3.5 seconds**. The next sample needs six additional/changed words, not another space after each individual word. Common unfinished English endings such as “and,” “to,” and “the” keep waiting.
- Partial words, one-word fragments, ongoing IME composition, unsaved edits, hidden tabs, and an open merge review hold automatic requests. The position being edited matters even inside an already long document. Unicode word segmentation and multilingual punctuation are supported. Pasting a paragraph of six or more words can count as a completed phrase without a trailing space.
- Slower hardware can extend the enrichment pause to the most recent measured generation time. Comments and research run no more than once per ten seconds each, when eligible, and wait for foreground enrichment. Unchanged content is skipped.

These are conservative boundary heuristics, not a grammatical parser or a guarantee that a statement is complete. **Refine entire prompt**, **Review document**, and manual research explicitly bypass the automatic boundary wait. The full saved draft remains the model's context; the app does not send isolated keystrokes or remove unfinished text from your saved draft.

The live pane is a preview until you confirm a merge. Generation never replaces your input draft. **Copy live prompt** copies the enriched output; **Show saved main** displays the last confirmed main; **Discard enrichment** returns the preview to your draft. **Refine entire prompt** requests a fresh rewrite in the same pane. Stale results cannot apply to a newer draft, including input containing emoji or other multilingual text.

**Processing and decisions are saved with the project.** Switching projects, refreshing the browser, reconnecting, or restarting Splash does not forget recently completed work. A SQLite ledger retains the most recent 128 attempts for each of enrichment, comment review, and research, identified by their input text and relevant model/context settings. Automatic requests reuse completed attempts; failed or interrupted attempts wait for new input or an explicit manual retry. A new model configuration or uploaded context can permit a fresh relevant pass. Generated previews, progress updates, and source timestamps do not trigger new cycles.

- **Dismiss** or **Resolve** on a wording suggestion keeps your current wording, removes its edit from the live preview, and remembers that passage. **Discard enrichment** remembers all rejected edits in that preview. Dismissed research additions are remembered by their project source references.
- **Apply to draft**, accepting research edits, a confirmed merge, or selecting a generated branch variant saves a checkpoint and waits for your next input. Splash does not automatically rewrite its own newly accepted output. Approved wording is protected against reversal during later enrichment.
- Editing elsewhere preserves these choices as the passages move. Editing the chosen passage releases its protection. **Reopen** revisits a dismissed/resolved wording discussion; applied edits remain in history and can be revised in the Working Draft.
- Manual **Refine entire prompt**, **Review document**, and research requests can rerun a completed input, while still respecting remembered wording choices. Recent resolved discussions and author replies also inform the reviewer. Matching and protection are based on anchored passages, not a guarantee of semantic equivalence across substantially rewritten text.

This memory is project-local, included in JSON exports, and removed when its project is deleted. Importing into a new project starts independent processing and decision history. Existing valid dismissed/resolved suggestions are migrated into decision memory; earlier accepted wording is recovered where its replacement can be uniquely located. Existing prompt text and main history are preserved.

For coding and research projects, evidence-based additions remain under **Research suggestions**, with rationale, supporting passages, source links, and evidence levels. **Accept edit** or **Accept all research** explicitly adds those suggestions to your input draft; **Dismiss** rejects them. They are never silently included in the live language rewrite.

Merge is always two stages:

1. **Review live merge** displays the exact additions, removals, and attached source references between the saved main and the live enriched prompt. Without a pending rewrite, **Review draft merge** reviews your input draft. Reviewing changes neither your input nor the saved main.
2. **Confirm merge & archive prior main** saves that exact reviewed prompt, retains the previous main and input draft in history, and initializes the next working draft from the promoted text, in one SQLite transaction.

**Keep editing**, Escape, or closing the review leaves the main and input unchanged. Any intervening draft, main revision, or reviewed enrichment change invalidates confirmation. Retrying the same successful confirmation returns the existing result rather than creating another main revision. Tab is ordinary keyboard navigation; it never promotes a prompt.

Example: begin with “Explain caching.” Add “Include two concrete examples” to your working draft. Watch the enriched wording appear in the main pane, then review and confirm to create main version 2. If that version drifts from your intent, open version 1 in **Version history** and choose **Restore as draft**. Edit or promote it through the same review flow. Version 2 remains available.

### A local collaborative document: you and the AI

Splash's collaboration is between **you and the local AI reviewer**. Projects, discussion threads, and model inference remain on this computer. This release does not provide shared accounts, remote collaborators, network synchronization, or simultaneous multi-user editing.

- **Several comments at once:** the agent can suggest wording, ask clarification questions, or identify competing instructions. Live enrichment proposes focused edits or clarification questions, each with an exact passage and a specific explanation. Cited context appears under **Supporting context**, with verified quotations and page/source references. AI review can add several questions or conflict comments in one pass.
- **Anchored discussions:** highlighted passages in the main preview open their comments. Click a quoted passage in a comment to select it in the Working Draft. A changed phrase may no longer appear verbatim in the live preview; its comment still links to the input it describes.
- **Your comments and replies:** select text in the Working Draft, write a note under **Add your comment**, and submit it. Reply within any open thread. **Ask AI to respond** requests a model response to that discussion; replying by itself saves your message without starting inference.
- **Explicit edits:** **Apply to draft** applies only that suggestion and preserves the preceding draft for recovery. It never changes the saved main. Conflicts and questions have no automatic Apply action: the author decides which facts or requirements are intended.
- **Resolve, dismiss, and reopen:** resolved/dismissed discussions remain in **All discussions** and can be reopened. Applied suggestions stay in discussion history. Background reviews preserve replies and do not reopen an unchanged resolved issue.
- **Revision safety:** unchanged passages keep their comments when surrounding text moves. Editing an anchored passage marks the old thread **Passage changed** and disables its edit; the discussion remains readable. Stale edits and comments belonging to another project cannot modify the current draft.

**Automatic review** samples eligible clauses and phrases in the saved active draft at most every ten seconds and skips unchanged completed reviews. **Review document** requests a fresh pass immediately. Review shares the existing diffusion worker at lower priority than live enrichment, so typing and foreground generation take precedence. Each AI pass covers up to four model-sized sections, adds at most twelve model review comments, and shows excluded characters. Enrichment edits and questions together are limited to eight per pass. A newer completed enrichment pass retires superseded open suggestions while preserving their discussion history. These comment limits are separate from the five-card research-branch limit.

Local English consistency checks can flag explicit positive/negative commands, conflicting values for the same stated subject, and incompatible word-count ranges. They are labeled **Consistency check**, not presented as model reasoning. For example, “Use Python. Do not use Python. The budget is $500. The budget is $900.” creates separate conflict discussions quoting both sides. A broader model pass looks for semantic conflicts and ambiguity; it must return validated, exact passage references. Invalid model output gets one repair attempt, then a visible unavailable-review notice. No checker guarantees that every contradiction is detected, and different contexts can make apparently competing statements compatible.

Open, current conflict comments pause automatic rewriting so the agent does not silently choose one of the directions. Clarify the working draft and resolve the discussion when satisfied. A merge review shows an additional reminder if conflicts remain; you retain control over promoting a deliberately unresolved prompt.

Comments are stored with each project in SQLite and included in a complete JSON export. Importing a prompt as a new independent project starts a fresh discussion history; it does not bring over the source project's comment threads.

The [local Dream review check](artifacts/comment-workspace-check.json) produced an anchored offline-versus-cloud conflict in addition to the two local consistency findings, while preserving the input draft. It also illustrates remaining model quality limitations: generated comments can have awkward wording or trailing artifacts. AI review can miss contradictions or incorrectly flag compatible goals; dismiss an incorrect comment to resume enrichment. The reviewer does not establish which user-provided fact is true.

### Recovery and concurrent editing

Main revisions are immutable. Autosaves also retain prior working drafts separately; **Include drafts & variants** reveals these entries. Every saved revision can be compared against the current main or restored as a new working draft.

The browser keeps a recovery copy of an unsaved draft. After reload or a connection interruption, Splash offers recovery when it differs from the saved version. If another tab changes the project, version checks prevent silent overwriting: resolve the conflict using the displayed recovery options. WebSocket connections retry automatically; job polling provides additional recovery. Keep the server running until the UI says **All changes saved** before closing it.

## Research branches and synthesis

For a **splash-search** project, open **Research branches**. Generate three cards by default; the selector permits one to five, and the server enforces the five-card ceiling.

- Select a single card and choose **Use branch as draft** to make a draft variant.
- Select two to five, or choose **Select all**, then **Synthesize selected** to combine the directions with the working prompt.
- The model is instructed to deduplicate overlap and put contradictory requirements under open questions. Review this result: conflict detection is model-assisted, not a formal guarantee.
- The prior working draft is preserved. The new variant records its originating main/draft, complete branch cards, condensed branch briefs, and source IDs.
- Branch generation uses the first 1,000 characters of the working draft and a bounded evidence brief; put the research objective early. Synthesis processes the full draft by section, carrying all selected branch briefs to each section.
- Selected cards are retained during background activity. Clear the selection before replacing them with newly generated cards.

Branch creation, selection, and synthesis never merge a main automatically. The synthesis is another draft to inspect, edit, and promote explicitly.

## Context and research tools

**Context & sources** accepts pasted notes, UTF-8 text, Markdown, code, and PDFs with extractable text. Limits are 8 MB per upload, 200 PDF pages, 500,000 extracted characters per document, and 30 documents per project. Prompt/draft text is limited to 80,000 characters; JSON imports are limited to 12 MB.

PDF page numbers, document names, and extracted passages are retained. **The stored attachment is extracted text with page references, not the original PDF binary or its layout.** Keep the original file separately. OCR, image interpretation, encrypted/image-only PDFs, and automatic repository inspection are outside this release.

Retrieval is local lexical passage ranking across the current draft. Enrichment retrieves up to **16 project passages**, then uses the loaded model's actual tokenizer and chat template to pack references within the configured context limit. Reference material can occupy up to **4,096 tokens**, with space reserved for instructions, the draft, output, and one repair attempt. The planner includes explicit draft requirements, prior author decisions, current objective excerpts, uploaded passages, accepted evidence, and historical main context as space allows. When evidence exists, at least two thirds of the reference allowance is available to it. Historical main text cannot override the current draft. Small profiles may fit very little context; the planner never silently raises the selected model's context ceiling.

**Adaptive output budgets** replace the fixed 256-token enrichment limit. The requested budget is 512 tokens for drafts up to 128 input tokens, 1,024 up to 512, 2,048 up to 1,536, and 4,096 for longer input. The actual budget is reduced to fit the selected context and instructions. A truncated response gets one retry with a larger reserved budget when space permits, up to 4,096; a formatting repair uses the original budget. End-of-text, assistant-turn boundaries from the bundled chat templates, and remaining diffusion masks are checked, so an incomplete response is withheld and a generated following turn is not treated as part of the suggestion. These budgets are response ceilings, not promises of output length, memory fit, or latency. Branch generation, synthesis, research planning, and comment replies retain their separate budgets.

Long drafts are split at exact character boundaries using token counts, with all input preserved. Each section receives the selected references. This is a bounded context view and does not establish global semantic consistency. The persisted **Enrichment review** below the editor reports the context limit, input/output allowances, used/excluded uploaded material, draft coverage, and reasons suggestions were withheld; it survives reloads.

The model is given a worked format example and asked to consider objectives, audience, deliverables, constraints, and missing information silently, then return up to two short anchored EDIT/QUESTION records per section (strict JSON is also accepted). Replacements are requested at up to 80 words and rationales at one sentence; these are generation instructions, not a semantic quality guarantee. Edits need a specific rationale; uncertain requirements become questions without changing the live prompt. No-change output is valid. The host verifies unique exact anchors, nonoverlapping edits, evidence IDs/quotations, known protected literals, and explicit English requirements. Unsupported numeric targets, URLs, code literals, and some new obligations are withheld. Existing author decisions still take precedence. A malformed or truncated pass gets at most one repair per section; if it remains invalid, the previous valid preview is preserved with an explanation. No deterministic check proves full semantic faithfulness or factual support—review the reasoning before applying an edit.

### MCP architecture and permissions

FastAPI hosts one persistent bundled MCP server over stdio through the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk). All projects share this process and its public caches. Project-private documents remain scoped by a project ID supplied by the host, never chosen by a generated request.

| Read-only MCP tool | Purpose | Agent access |
|---|---|---|
| `search_context` | Rank passages from this project's uploads | All agents |
| `read_context` | Read a referenced passage from this project | All agents |
| `search_scholarly` | Search Semantic Scholar, Crossref, or optional Scholar | Coding/research, when enabled |
| `search_official_docs` | Search the supported documentation domains | Coding/research, when enabled |
| `fetch_source` | Fetch a public page or text PDF with provenance | Coding/research, when enabled |

Inference, database mutations, and merge confirmation stay in the application. Generated search calls are validated against schemas and agent permissions. A malformed plan gets one repair attempt, then deterministic retrieval. MCP failures appear as unavailable research and do not prevent local editing or model-only refinement; local context retrieval has an in-process fallback.

The host validates public destinations and every redirect, rejects private/local IPs and nonstandard ports, and pins requests to a validated address. Retrieved material is treated as evidence, never as authorization. There are no arbitrary third-party MCP servers or autonomous code execution tools.

### Research settings and provider behavior

Research is opt-in per coding/research project. Once enabled, the active visible project samples changed prompts about every ten seconds. Unchanged input is skipped, and only one research job per project runs at a time. **Research now** can request another cycle explicitly.

Each cycle has at most **two searches and three source fetches**, with provider timeouts/backoff and a two-minute cycle deadline. Editing cancels obsolete project jobs. Active enrichment has priority over research generation. Public results are cached in the tool process for 24 hours, up to 500 entries; the cache is shared but uploaded context is not placed in it.

Providers:

- **Semantic Scholar:** scholarly metadata and abstracts when available. Its unauthenticated API can return 429; Splash reports that limit and continues through the second provider.
- **Crossref:** publication metadata and available abstracts; metadata is not treated as full text.
- **Official documentation:** Bing's RSS search restricted to Python, MDN, TypeScript, React, Node.js, Rust, Go, Microsoft Learn, and GitHub Docs domains. Search availability and relevance can vary.
- **Supplied public URLs:** up to the remaining fetch budget, with redirect and destination validation. Private/intranet URLs are intentionally unavailable.
- **Google Scholar:** separately opt-in and experimental. Results are cached, requests are spaced at least one minute apart, and 403/429/CAPTCHA responses stop the scraper for that tool-server session. There is no CAPTCHA bypass. Scholar restricts automated access; see its [official guidance](https://scholar.google.com/intl/en/scholar/help.html). Use the other providers when unavailable.

Only focused query keywords are sent to search providers; uploaded documents are not sent wholesale. Keywords are derived from the working prompt, so avoid enabling external research on a prompt whose search terms must remain private. Source fetching contacts the requested public sites. Query history is visible in the project (latest 200 entries). Evidence is labeled as **snippet**, **metadata**, **abstract**, **passage**, or **full-text excerpt**; none of these labels certifies a claim. Fetched text is bounded, not necessarily a whole publication.

With a ready model, relevant excerpts can produce source-informed edit proposals. Without one, retrieval can still produce clearly labeled “evidence to consult” cards. Neither kind edits a draft without acceptance.

## Measured hardware and quality

Checks were run on October 6, 2026 UTC on Linux, an Intel Core Ultra 9 285K with approximately 64 GB RAM, NVIDIA driver 615.71.09, and the runtime above. Figures below are **individual small-prompt samples**, not percentiles, sustained throughput, startup/download time, or full UI latency. GPU memory is PyTorch peak allocated memory, not total device reservation.

| Device | Model / mode | Generation samples | Peak GPU allocation | Result |
|---|---|---:|---:|---|
| RTX 2070, 8 GB | Compact, FP16 | 2.23 s first / 1.08 s second | ~1,232 MiB | Runs; weak enrichment quality |
| RTX 5090, 32 GB | Compact, BF16 | 1.67 s first / 0.68 s second | ~1,232 MiB | Runs; weak enrichment quality |
| RTX 2070, 8 GB | Dream, NF4 | 11.30 s / 10.62 s | ~5,841 MiB | Runs; slower updates |
| RTX 5090, 32 GB | Dream, BF16 | 9.88 s first / 1.36 s second | ~14,842 MiB | Runs; preferable starting point on this machine |
| RTX 5090, 32 GB | LLaDA 2.1 mini, NF4 | 7.93 s / 14.11 s | ~9,993 MiB | Runs; slower, quality needs further evaluation |
| CPU, 4 inference threads | Compact, FP32 | 4.59 s / 4.68 s | Not applicable | CPU execution verified |
| GTX 1050 | Compact candidate | Not measured | Not measured | **Unverified on actual hardware** |
| Windows / Apple Metal | Candidate profiles | Not measured | Not measured | **Unverified** |

GPU samples use a 128-token output budget and 64 configured steps; LLaDA uses its own schedule and can stop early. CPU samples use only 64 output tokens and 16 steps, so their timings are not directly comparable. The larger model downloads do not imply they fit unquantized on an 8 GB GPU. Older Pascal devices need a compatible older PyTorch/CUDA environment, or CPU fallback; no specific GTX 1050 GPU installation is certified here.

Raw outputs and metrics:

- [Compact on both GPUs](artifacts/model-qualification.json)
- [Dream on RTX 5090](artifacts/dream-qualification.json) and [Dream NF4 on RTX 2070](artifacts/dream-2070-qualification.json)
- [LLaDA NF4](artifacts/llada-qualification.json) and [CPU fallback](artifacts/cpu-qualification.json)
- [Final live LLaDA enrichment check](artifacts/live-enrichment-check.json), with its explicit requirement preserved and both main/draft unchanged until acceptance
- [Five-branch LLaDA synthesis audit](artifacts/branch-synthesis-evaluation.json), including the consent omission that prompted the constraint-retention safeguard
- [Eight-case enrichment audit](artifacts/enrichment-evaluation.json), including English, Spanish, French, German, Chinese, and Arabic
- [Live research-provider checks](artifacts/provider-check.json)
- [Grounded LLaDA review check](artifacts/grounded-llada-live-check.json), using the expanded-context pipeline and Quality mode on October 10, 2026

The newer grounded-review check on RTX 5090 produced **zero accepted suggestions**. LLaDA used the uploaded website brief, but failed the requested edit format on both attempts and introduced an unsupported “under 100 KB” image target. The format validator withheld the response, leaving the main and working draft unchanged. The two generations took 39.2 and 66.6 seconds (105.8 seconds combined), with peak allocated VRAM of approximately 11,987 MiB. This is one case, not a benchmark; it demonstrates that more context and larger budgets do not by themselves resolve this checkpoint's instruction-following problems. Inspect withheld output in **Enrichment review**. Reliable grounded suggestions still require model and harness qualification.

The multilingual audit found useful English rewrites but also a dropped French testing requirement, awkward repetition, and limited research expansion. The new constraint guard catches that particular testing omission; general semantic fidelity and graduate-level quality remain review responsibilities. Compact often answers the task instead of improving the instruction. Longer outputs can end at the token limit. No model is represented as fully quality-qualified.

The UI and network remain separate from the worker. One generation runs at a time, the latest pending request wins, and obsolete generations stop between diffusion steps. Browser checks exercise typing during generation and autosave/reload; a formal cross-platform typing-latency percentile benchmark and process-wide peak RAM study remain future work. Research latency is recorded independently in provider checks.

## Storage, backups, and deletion

The default directory comes from `platformdirs`:

- Linux: typically `~/.local/share/Splash`
- Windows: typically `%LOCALAPPDATA%\Splash`
- macOS: `~/Library/Application Support/Splash`

The exact path appears in **Model & hardware**. `SPLASH_DATA_DIR` or the launcher's `--data-dir` overrides it. Configuration is read from the environment; a `.env` file is not loaded automatically.

`splash.sqlite3` contains projects, immutable main revisions, recoverable drafts, variants, extracted attachments, evidence, import lineage, settings, and the model configuration. SQLite uses WAL and transactional version checks. Model weights occupy `models/`, and CPU offload can occupy `offload/`. A configured external checkpoint remains in its original location. The browser also uses local storage for unsaved recovery data.

For a complete backup, stop Splash cleanly and copy its data directory (or use SQLite's online backup facility while running). Do not copy only the live `.sqlite3` file while omitting its WAL. Model weights can be downloaded again; the database cannot reconstruct deleted user prompts without a backup.

**Delete** permanently removes that project's stored history, drafts, context, and evidence from the app; export or back it up first. Independently imported projects survive. Deletion does not delete shared weights or external exports/backups, and is not a forensic secure erase of SQLite/WAL or storage media. Remove shared weights manually when no longer needed. Public caches disappear when the tool process stops. Local-only storage is not encryption; normal OS account and disk protections apply.

## Develop and verify

In separate terminals:

```sh
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 43827 --reload --ws-max-size 100000
npm --prefix frontend run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api`, `/ws`, and `/healthz` to the backend. Production builds serve from FastAPI. After changing frontend code outside Vite development mode, rebuild and reload the page.

```sh
.venv/bin/python -m unittest discover -s tests -v
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend audit
.venv/bin/python -m pip check
.venv/bin/python scripts/check_mcp.py
.venv/bin/python scripts/check_providers.py
```

Backend tests cover isolation, cross-agent imports, copied-context independence, Unicode, autosave conflicts, immutable history, exact diffs, stale/canceled/duplicate merges, transaction rollback, restoration, five-card enforcement, synthesis lineage, protected constraints, MCP permissions, malformed planning, provider failures, Scholar blocks, private-network rejection, WebSocket identity, and scheduler cancellation. Frontend tests cover serialized autosave, recovery, conflict handling, safe source links, and connection behavior. MCP/provider scripts contact real processes/public services; their outputs can differ with network availability.

To repeat actual model qualification, allowing checkpoint downloads only when requested:

```sh
.venv/bin/python scripts/qualify_models.py --models compact --devices cuda:0 cuda:1 --download
.venv/bin/python scripts/qualify_models.py --models dream --devices cuda:0 --quantize --output artifacts/dream-2070-qualification.json
.venv/bin/python scripts/qualify_models.py --models llada --devices cuda:1 --quantize --output artifacts/llada-qualification.json
```

Inspect your device indices first. The qualification script uses repository-local `.models/`; the app's download button uses its app-data directory. Point the model dialog at `.models/<model>` to reuse script downloads. `scripts/evaluate_enrichment.py` now runs the repeatable cases in `tests/fixtures/enrichment_cases.json` through the grounded enrichment path, in a temporary store. It never changes user projects or downloads weights. Choose an existing checkpoint and a free device explicitly:

```sh
.venv/bin/python scripts/evaluate_enrichment.py --model llada --path .models/llada --device cuda:1 --context 32768 --quantize --output artifacts/llada-grounded-review.json
```

Use `--model dream --path .models/dream --context 2048` for Dream; choose the appropriate device. `--limit 3` runs a quick subset. `--quality-mode speed` records the faster schedule. `--baseline` also records the former 256-token rewrite request for human comparison, using the selected generation schedule. Results retain raw responses, exact proposals, evidence, token/latency/memory metrics, and unchanged-main/draft checks. Human usefulness and semantic-fidelity ratings are intentionally left blank; literal checks are not quality scores. Use separate output files for each configuration and compare the same cases. Exit status 1 flags a runtime or unrepaired format failure; status 0 does not certify useful suggestions. The earlier eight-case multilingual artifact remains a historical audit.

### Application interfaces

Interactive HTTP documentation is available at **http://127.0.0.1:43827/docs**. Principal endpoints:

| Route under `/api` | Action |
|---|---|
| `/projects` | List/search and create; creation can name a source project/revision |
| `/projects/{id}` | Read, rename/configure, or delete |
| `/projects/{id}/draft` | Save with `expected_version` |
| `/projects/{id}/merge-review` | Capture the exact main/draft comparison |
| `/projects/{id}/merge/{review_id}` | Confirm idempotently, or cancel with DELETE |
| `/projects/{id}/comments`, `/comments/{comment_id}/reply`, `/comments/{comment_id}` | Create an anchored discussion, reply, resolve/dismiss/reopen |
| `/projects/{id}/comments/review`, `/comments/{comment_id}/ask`, `/comments/{comment_id}/apply` | Review the document, request an AI reply, or explicitly apply a suggested edit |
| `/projects/{id}/restore` | Restore a revision as a draft |
| `/projects/{id}/compare/{revision_id}` | Compare a saved revision to the current main |
| `/projects/{id}/documents` | Upload context |
| `/projects/{id}/refine` | Start proposal generation |
| `/projects/{id}/branches/generate`, `/select`, `/synthesize` | Manage branch cards and variants |
| `/projects/{id}/research`, `/jobs` | Research and job status |
| `/projects/{id}/export`, `/projects/import-file` | Export an archive/prompt or import an archive |
| `/hardware`, `/models`, `/models/inspect`, `/models/download`, `/models/load` | Hardware/package status, checkpoint inspection, pinned Hub downloads, and shared inference |
| `/projects/{id}/system-prompts` | Read or replace that project's system prompts |
| `/prompt-repo` | List the built-in defaults and saved library, or save one prompt |
| `/prompt-repo/import` | Add a `splash-system-prompt` version 1 JSON file to the library |
| `/prompt-repo/{id}` | Remove a saved library prompt; built-in defaults stay |

`/ws/projects/{id}` emits protocol-2 events scoped to that project: `project_changed`, `progress`, `job_complete`, `job_error`, `job_canceled`, and `project_deleted`. Project-change events include agent, main ID, and draft version. Job results carry proposals, context accounting, research results, or variants. Mutations remain validated HTTP operations.

The earlier deterministic mock remains isolated in `backend/engine.py` and legacy `/ws` for regression coverage. Its responses explicitly say `engine: "mock"`; the project studio never uses it. The old `tests/test_latency.py` measures that mock protocol only and must not be cited as diffusion-model performance.

### Code map

- `backend/storage.py`: transactional project store and revision invariants.
- `backend/api.py`: project/model HTTP APIs and scoped WebSocket events.
- `backend/models.py`: hardware inspection, adapters, and persistent inference process.
- `backend/checkpoints.py`: compatible local checkpoint checks, Hugging Face revision resolution, and resumable downloads.
- `backend/comments.py`: anchored discussions, conservative consistency checks, model review validation, and revision-safe suggestion application.
- `backend/state.py`: persistent processing checkpoints, remembered edit decisions, and protection against repeated or reversed suggestions.
- `scripts/setup.py`, `scripts/dependencies.py`: hardware-specific dependency installation and isolated runtime checks.
- `backend/agents.py`: policies, proposals, branches, synthesis, and research scheduling.
- `backend/mcp_host.py`, `mcp_server.py`, `research.py`: shared MCP lifecycle, permissions, local retrieval, and public providers.
- `frontend/src/app.jsx`, `components/StudioParts.jsx`, `client.js`: project UI, merge review, and serialized draft recovery.
- `frontend/src/assets/splash-logo.jpg`: original Splash logo, shared by the app and this README. `frontend/src/index.css` defines the charcoal, teal, and violet theme with reusable color variables.
- `tests/`, `frontend/tests/`, `scripts/`, `artifacts/`: regression tests and reproducible qualification records.
- `mobyMinutes/`: historical design/build records, not the current runtime specification.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Model controls say no model is ready | Install PyTorch/model requirements, download or select a checkpoint, then load it. Editing/history remain available. |
| `address already in use` on port 43827 | Splash may already be running. Open `http://127.0.0.1:43827/`; the updated `scripts/start.py` detects an existing instance before loading weights. To restart this checkout's server, run `python3 scripts/stop.py` first. Do not launch several instances against the same GPU. |
| `ModuleNotFoundError: No module named 'torch'` | Run `python3 scripts/setup.py` from the project directory; PyTorch is now included by default. On GTX 1050 use `--profile nvidia-legacy --level compact`. Resolve any failed setup checks, then restart using `scripts/start.py` so Splash uses the same `.venv` that was checked. |
| GPU selector shows less memory than the card's advertised capacity | The first value is **free** VRAM and the second is **total** VRAM. Loaded models, runtime caches, and other GPU programs reduce free memory. Use `nvidia-smi` to inspect current processes. |
| Hub download fails or says access denied | Check the repository ID/revision, connection, and local Hub login or `HF_TOKEN`. Gated repositories require prior access. Retry the download to reuse partial files. |
| Checkpoint check reports wrong architecture or LFS pointers | Choose the matching adapter and a complete weights directory. Pull Git LFS weights or download through Hugging Face. A Git source repository alone is not a model checkpoint. |
| CUDA unavailable or GPU scan fails | Run `nvidia-smi` in a normal local terminal. A sandbox permission failure does not prove the GPU or driver is absent. Match the runtime to the architecture. |
| GPU out of memory | Reduce context, choose Compact, or try the measured Dream NF4 profile with bitsandbytes. Check other GPU processes. Offload is experimental and slower. |
| Model import or custom-code error | Use the pinned requirements and matching catalog checkpoint. LLaDA requires the bundled eager-mask/RoPE compatibility path. Do not mix a random Transformers major version into the tested environment. |
| A rewrite is rejected for omitted content | The draft is intact. Try another model or diffusion schedule, or refine a smaller task. Review semantic constraints even when the guard passes. |
| Research returns 429, CAPTCHA, or no results | Other providers may continue. Wait for backoff, try a focused query, or supply a public source URL. Scholar blocking is not bypassed. |
| MCP tools are unavailable | Local editing still works. The host retries server startup automatically; inspect backend stderr and verify the pinned MCP dependency in the same virtual environment. |
| Unsaved changes or a stale-merge error | Resolve the displayed recovery/conflict choice, save the draft, and open a new merge review. Never force an old review onto a new revision. |
| The main pane does not show enrichment | Turn on **Live enrichment**, load a model successfully, and type in **Working Draft**. If viewing the saved version, select **Show live enrichment**. Wait for generation to finish or inspect its error. **Refine entire prompt** requests a manual update. Saving a new main version still requires review and confirmation. |
| PDF yields no text | Use a text-based PDF or paste a transcription; OCR is outside scope. |
| UI shows old content or is not built | Run `npm --prefix frontend run build`, then reload. Restart the backend after backend changes. |
| Tests hang in a restricted sandbox | Run in a normal local terminal with loopback socket access. In-process WebSocket tests still need event-loop socket facilities. |
| Port already in use | Start with `python3 scripts/start.py --port 43828` or stop your previous Splash instance. The macOS app picks the next free loopback port. |
| macOS app does not open | Build it on a Mac with `python3 scripts/package_macos.py`. Right-click the app and choose Open the first time. Quit from the Splash menu. Read `~/Library/Logs/Splash/splash.log`. |

Custom training, stronger multilingual/constraint evaluation, certified GTX 1050/Windows/Metal profiles, richer retrieval, cloud synchronization, Windows application packaging, arbitrary external MCP servers, and autonomous code execution are future work. A macOS app can be built with `scripts/package_macos.py`. See [MODEL_ROADMAP.md](MODEL_ROADMAP.md) for the remaining model-specific work.
