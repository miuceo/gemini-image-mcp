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
  HTML gallery (`output/gallery.html`) — the primary way to browse your output.
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

### Cookie lifetime

`__Secure-1PSIDTS` rotates frequently. `gemini-webapi` refreshes it in the background
while the server is running and persists the refreshed value to the path in
`GEMINI_COOKIE_PATH` (default: `output/.cookies`), so in the happy path you only paste
cookies once. Chrome's session credentials tend to be shorter-lived than Firefox's; if you
find yourself re-pasting often, a dedicated Firefox profile kept signed in to Gemini is a
more durable workaround.

**Chrome cookies cannot be imported automatically.** Since Chrome 127, Windows encrypts
its cookie store with app-bound encryption that tools like `browser-cookie3` cannot
decrypt. There is no shortcut here — copy the two values by hand as above.

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
| `generate_image` | `prompt: str`, `model: str \| None = None` | Absolute path(s) of the newly saved image(s), the absolute path to `output/gallery.html`, the image count, and the model actually used. |
| `edit_image` | `prompt: str`, `image_paths: list[str]`, `model: str \| None = None` | Same return shape as `generate_image`, applied to edits of the given source image(s). |
| `list_models` | _(none)_ | The models (image/chat) available to the signed-in account, as display name + internal model name/id, so you can pick a non-default `model` for the other two tools. |

None of the tools return base64 image data — the gallery is the intended viewing surface,
and every tool result includes its path so you can always click through to it.

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
- **Where things live on disk** — generated/edited images: `output/images/`; the gallery:
  `output/gallery.html`; the manifest backing the gallery: `output/manifest.json`; the
  persisted cookie cache: wherever `GEMINI_COOKIE_PATH` points (default:
  `output/.cookies`). All of `output/` is gitignored.

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
