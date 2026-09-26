<p align="center">
  <img src="assets/banner.jpg" alt="Gemini Image MCP" width="100%" />
</p>

<p align="center">
  <img src="assets/logo.png" alt="Gemini Image MCP logo" width="96" />
</p>

# gemini-image-mcp

An MCP server that generates and edits images by driving the consumer **Gemini web app**
(`gemini.google.com`) with your own browser session cookies, via the
[`gemini-webapi`](https://pypi.org/project/gemini-webapi/) library — instead of the paid,
metered Gemini API.

**Why this exists:** the official Gemini API bills per image. If you already have a
Gemini web account where image generation works, this lets you use that account instead
of a metered API key. The trade-off is described below — read it before using this.

This project is **not affiliated with, endorsed by, or supported by Google**. "Gemini" is
a Google product; this is an independent, unofficial client built on top of the
third-party `gemini-webapi` library.

## Disclaimer

Driving the consumer web app with your own session cookies, instead of an official API
key, is outside Google's Terms of Service for that product. Any consequence — rate
limiting, a challenge, or account action — lands on your Google account, not on this
code. This tool doesn't take a position on whether that trade-off is worth it beyond
making sure you know it up front. See the [License](#license) section for the full
no-warranty disclaimer.

## Features

- **`generate_image`** and **`edit_image`** MCP tools backed by your own Gemini web
  session — no API key, no per-image billing.
- **`list_models`** to discover which models your account can use.
- Every generated or edited image accumulates into a single, self-contained, interactive
  HTML gallery (`output/gallery.html`) — the primary way to browse your output. Click any
  image for a detail view with its full prompt, metadata, and copy buttons.
- **Optional GitHub publishing:** point it at a repository you own and every new image is
  also uploaded there, giving you a stable public URL to use in other projects. Off unless
  you configure it — see [Publishing images to GitHub](#publishing-images-to-github).
- `gemini_image_mcp.core` has zero `mcp` imports, so it can be imported directly by other
  Python code (e.g. a Telegram bot) without going through the MCP protocol at all.
- Automatic image-format correction: Gemini doesn't always return the format you'd expect,
  so saved files are renamed to match their actual content (e.g. `.jpg` instead of a
  mislabeled `.png`).

## Requirements

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/)
- A Gemini web account where image generation **already works manually** in your browser.
  Image generation is 18+ and region-limited, and requires **Gemini Apps Activity** to be
  turned on for your account. If the web UI itself won't generate an image, this tool
  can't either — fix that first.

## Install / Quickstart

```bash
uv sync
```

```bash
copy .env.example .env
```

Edit `.env` and paste in `GEMINI_1PSID` / `GEMINI_1PSIDTS` (see
[Getting your cookies](#getting-your-cookies) below), then run the smoke test:

```bash
uv run python scripts/smoke_test.py "a watercolor fox"
```

The smoke test prints the saved image path(s) and the gallery path on success, or a clean
error message (e.g. about missing/expired cookies) on failure.

## Getting your cookies

1. In Chrome, sign in and go to `https://gemini.google.com`. **Manually generate an image
   there first** (e.g. ask it to "generate an image of a fox") and confirm it actually
   works. If the web UI itself won't generate an image, this tool cannot either — fix
   that first.
2. Press `F12` to open Chrome DevTools.
3. Go to the **Application** tab -> **Storage** -> **Cookies** -> `https://gemini.google.com`.
4. Find the cookie named `__Secure-1PSID`. Click it and copy its **Value** column (a long
   string). This is your `GEMINI_1PSID`.
5. Find the cookie named `__Secure-1PSIDTS`. Copy its **Value** the same way. This is your
   `GEMINI_1PSIDTS`.
6. Copy `.env.example` to `.env` and paste the two values into `GEMINI_1PSID` and
   `GEMINI_1PSIDTS`.

### Automatic cookies from your browser (recommended)

Instead of pasting cookies, sign in to https://gemini.google.com in a supported browser
and leave `GEMINI_1PSID` / `GEMINI_1PSIDTS` empty. The server reads both cookies straight
from the browser's cookie store (via `browser-cookie3`) at startup, and again whenever
Gemini reports the session as signed out - so a fresh login is picked up without touching
`.env`. The browser doesn't need to be running. Stay signed in there (no private window,
don't clear its cookies).

In `auto` mode the server tries Firefox, LibreWolf, Edge, Chrome, Brave, Chromium, Vivaldi
and Opera in that order, and uses the first one whose Gemini login works. On macOS only
Firefox and LibreWolf are tried automatically, because reading Chromium-based browsers
there triggers a Keychain password prompt. Name the browser explicitly to use it anyway.

| Browser | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Firefox, LibreWolf | Yes | Yes | Yes |
| Edge, Brave, Vivaldi, Opera, Chromium | Yes | Yes (Keychain prompt) | Yes |
| Chrome | **No** - see below | Yes (Keychain prompt) | Yes |
| Safari | - | Yes (with Full Disk Access) | - |

- `GEMINI_COOKIE_SOURCE` - `auto` (default: `.env` values if set, else the browsers
  above, also used as a fallback when the `.env` values are stale), `env` (only `.env`),
  or one browser: `firefox`, `librewolf`, `edge`, `chrome`, `brave`, `chromium`,
  `vivaldi`, `opera`, `safari`.
- `GEMINI_BROWSER_COOKIE_FILE` - path to a specific profile's cookie database, if you have
  several profiles (for `auto`, it applies to Firefox). `GEMINI_FIREFOX_COOKIE_FILE` still
  works as the older name.

Run `uv run gemini-image-mcp doctor --offline` to see which browser was found, or why
none could be read.

### Cookie lifetime

`__Secure-1PSIDTS` rotates frequently. `gemini-webapi` refreshes it in the background
while the server is running and persists the refreshed value to the path in
`GEMINI_COOKIE_PATH` (default: a private per-user folder: `%LOCALAPPDATA%\gemini-image-mcp\cookies` on Windows,
`~/Library/Application Support/gemini-image-mcp/cookies` on macOS,
`~/.local/state/gemini-image-mcp/cookies` on Linux), so in the happy path you only paste
cookies once. The cache is kept outside `output/` on purpose, so sharing or publishing
your images never exposes your session. If you paste cookies by hand
and find yourself re-pasting often, keeping Firefox or Edge signed in to Gemini and
letting the server read them is a more durable workaround.

**Chrome on Windows can't be read automatically.** Since Chrome 127, Chrome on Windows
encrypts its cookies with app-bound encryption, which can only be decrypted with
administrator rights, and an MCP server should never run as administrator. If Chrome is
your main browser, sign in to Gemini once in Firefox or Edge just for this server, or copy
the two values by hand as above.

### These cookies are your account

`__Secure-1PSID` and `__Secure-1PSIDTS` are equivalent to being logged in as you. Never
commit them, paste them into a chat, or share them with anyone. `.env` is gitignored by
default — keep it that way.

## Registering with Claude Code

```bash
claude mcp add gemini-image-mcp -- uv run --directory "/path/to/gemini-image-generator-mcp" gemini-image-mcp
```

(On Windows this might look like
`uv run --directory "C:\Users\you\projects\gemini-image-generator-mcp" gemini-image-mcp`.)

This registers the server to run via the `gemini-image-mcp` console script that `uv sync`
installs into the project's virtual environment (defined in `pyproject.toml`'s
`[project.scripts]`), always executed from the correct project directory regardless of
your current working directory.

Equivalent JSON (e.g. for `claude_desktop_config.json` or `.mcp.json`):

```json
{
  "mcpServers": {
    "gemini-image-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/gemini-image-generator-mcp",
        "gemini-image-mcp"
      ]
    }
  }
}
```

## Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `generate_image` | `prompt: str`, `model: str \| None = None`, `publish: bool \| None = None` | Absolute path(s) of the newly saved image(s), the absolute path to `output/gallery.html`, the image count, the model actually used, and each image's public URL if it was published. |
| `edit_image` | `prompt: str`, `image_paths: list[str]`, `model: str \| None = None`, `publish: bool \| None = None` | Same return shape as `generate_image`, applied to edits of the given source image(s). |
| `publish_image` | `image_paths: list[str]` | Uploads existing images (absolute paths, `images/<file>` paths, or gallery ids) to your configured GitHub repo and returns their public URLs. |
| `list_models` | _(none)_ | The models (image/chat) available to the signed-in account, as display name + internal model name/id, so you can pick a non-default `model` for the other two tools. |
| `doctor` | _(none)_ | A setup health check, one ok/warn/fail line per item (see [Troubleshooting](#troubleshooting)). |

None of the tools return base64 image data — the gallery is the intended viewing surface,
and every tool result includes its path so you can always click through to it.

## Publishing images to GitHub

Optionally, every generated or edited image can also be committed to a GitHub repository
of your choice (for example a public `my-images` repo), so you get a direct link you can
drop into websites, READMEs, or other projects:

```
https://raw.githubusercontent.com/<owner>/<repo>/<branch>/images/65157777b2c94e0f8d1a3c5e7f902468.jpg
```

Files are named by their gallery id, and commit messages carry only a short id, so your
prompts stay private even in a public repo (see `GEMINI_GITHUB_INCLUDE_PROMPT` below to
opt in to prompt-based names). Upload goes through the GitHub REST API, one commit per
image, so no local clone is needed. To turn it on, add to `.env`:

```dotenv
GEMINI_GITHUB_REPO=yourname/my-images
```

- **Token** - `GEMINI_GITHUB_TOKEN`, else `GITHUB_TOKEN`, else the GitHub CLI's login
  (`gh auth token`). **Recommended:** create a
  [fine-grained token](https://github.com/settings/personal-access-tokens/new) limited to
  that one repository with only **Contents: read and write**, and set it as
  `GEMINI_GITHUB_TOKEN`. The `gh` fallback is convenient but its token usually has full
  `repo` access to *all* your repositories, far more than this server needs.
- `GEMINI_GITHUB_BRANCH` - branch to commit to (default: the repo's default branch).
- `GEMINI_GITHUB_PATH` - folder inside the repo (default: `images`).
- `GEMINI_GITHUB_AUTO_PUBLISH` - `false` (default) publishes only when a call passes
  `publish=true` or you use the `publish_image` tool; `true` publishes every new image.
- `GEMINI_GITHUB_INCLUDE_PROMPT` - `false` (default). Set to `true` to name files after
  the prompt (`images/a-watercolor-fox-65157777.jpg`) and put the prompt in the commit
  message. Remember that anything in a public repo, including its history, is public.
- Only images inside `output/images/` can be published; any other path is refused.

You can also override per call ("generate ... and publish it"), and publish older
images later with `publish_image`. Published images show a **Published** badge in the
gallery with **Copy link** / **Markdown** buttons.

A publishing failure (bad token, network) never loses the image: it's still saved locally
and added to the gallery, and the tool result explains what went wrong.

Raw links only work without authentication if the target repository is **public**. Keep
in mind that anything you publish there is visible to everyone.

## Using the core library directly

`gemini_image_mcp.core` is deliberately MCP-free, so it can be imported directly — for
example from a Telegram bot:

```python
from gemini_image_mcp.config import get_settings
from gemini_image_mcp import core

async def handle_image_request(prompt: str) -> None:
    settings = get_settings()
    records = await core.generate_images(prompt, settings=settings)
    for record in records:
        absolute_path = settings.output_dir / record.image_path
        # e.g. send `absolute_path` back to the Telegram chat
```

## Troubleshooting

Start with the built-in health check:

```bash
uv run gemini-image-mcp doctor
```

It checks, in order:

| Check | What it looks at |
| --- | --- |
| Output directory | `GEMINI_OUTPUT_DIR` exists and is writable. |
| Cookie cache | The cache is outside the output folder (so sharing images never shares your session). |
| Cookies | `GEMINI_1PSID` / `GEMINI_1PSIDTS` are set, or a supported browser has a Gemini login (and if not, why each browser couldn't be read). |
| Gemini sign-in | The cookies actually work, and how many models the account can use. |
| Gallery | How many images the manifest holds, and whether a corrupt one was set aside. |
| GitHub publishing | Where the token comes from, that it can push to the repo, and whether the repo is public. |

Each line is `[ OK ]`, `[WARN]`, `[FAIL]` or `[SKIP]` with a hint on how to fix it, and the
command exits with status 1 if anything failed. Add `--offline` to skip the calls to Gemini
and GitHub. Cookie and token values are never printed. You can also just ask Claude to "run
the doctor", which calls the `doctor` tool.

- **Auth / cookie expiry** — `generate_image`, `edit_image`, and `list_models` all raise a
  clean error naming the exact problem when `GEMINI_1PSID`/`GEMINI_1PSIDTS` are missing or
  have expired. Re-copy both values from Chrome DevTools (see above) and update `.env` (or
  the running process's environment).
- **"No images returned" for a prompt that seems reasonable** — this is almost always
  Gemini refusing the request (safety filter, region restriction, age-gating) rather than
  a bug. The error message includes the model's own text response, which usually explains
  the refusal directly.
- **Usage limits / temporarily blocked** — Gemini's own rate limiting on the web app.
  Wait before retrying; there is no bypass.
- **GitHub publish failed** - the tool result includes GitHub's message. 401/403 means
  the token is missing, expired, or lacks Contents write access to the repo; 404 means
  `GEMINI_GITHUB_REPO` is wrong or the token can't see that repository.
- **Where things live on disk** — generated/edited images: `output/images/`; the gallery:
  `output/gallery.html`; the manifest backing the gallery: `output/manifest.json`; the
  persisted cookie cache: wherever `GEMINI_COOKIE_PATH` points (default: a per-user folder
  outside the project, see [Cookie lifetime](#cookie-lifetime); an old `output/.cookies`
  cache is moved there automatically). All of `output/` is gitignored.

## Known limitations

- **Unofficial and fragile by nature.** This drives a consumer web app that Google can
  change at any time without notice; a Google-side change can break this project until
  it's updated.
- **Region/age gated.** Image generation availability depends on your Google account's
  region and age, not on this project.
- **Output format is whatever Gemini returns.** Currently that's typically JPEG,
  regardless of the extension you might expect — this project sniffs the real format and
  corrects the file extension accordingly rather than trusting the request.
- **The manifest grows unbounded.** `output/manifest.json` accumulates every record ever
  generated; there's no pruning or rotation.
- **Not a daemon.** The MCP server's lifetime is tied to the Claude session that launched
  it — it's not designed to run as a standalone, long-lived background service.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md) for the credential-handling model and how to report a
vulnerability.

## License

[MIT](LICENSE) — see the disclaimer above for the ToS and warranty caveats specific to
this project; the MIT license text itself covers the standard no-warranty terms.
