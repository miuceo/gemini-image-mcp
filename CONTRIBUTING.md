# Contributing

Thanks for considering a contribution to `gemini-image-mcp`.

## Dev setup

```bash
uv sync
```

This creates `.venv` and installs all runtime and dev dependencies from `uv.lock`.

Run the test suite with:

```bash
uv run pytest
```

If you're adding a feature, add or update tests alongside it. Tests must not depend on a
live Gemini account or make real network calls — mock `gemini_webapi` at the boundary
(`gemini_image_mcp.client`) instead.

## Reporting bugs

Open a GitHub issue with:

- What you ran (tool call, script, or command) and what you expected.
- The actual error message or behavior. `GeminiImageError` subclasses are designed to be
  safe to paste verbatim — but double-check your paste doesn't include anything else from
  your terminal (paths, environment dumps, etc.) that you don't want public.
- Your OS and Python version.

## Pull requests

- Keep PRs focused; unrelated cleanup makes review harder.
- **Never commit real Gemini session cookies, `.env`, generated images, or `output/`
  contents.** Double-check `git diff`/`git status` before pushing — a leaked `.env` means
  your Google session is exposed and should be rotated immediately (sign out of Google
  everywhere to invalidate it).
- Don't add telemetry, analytics, crash reporting, or any code that phones home to a
  third-party service. This project only talks to Google's own Gemini endpoints on the
  user's behalf, and contributions should preserve that property.
- `core.py` must stay free of `mcp` imports — it's meant to be importable directly by
  other consumers (e.g. a Telegram bot).
- Describe what you changed and why in the PR description.

## Code style

There's no enforced formatter/linter configured yet. Match the existing style (type hints,
docstrings on public functions, exceptions translated at the `client.py` boundary rather
than leaking third-party exception types).
