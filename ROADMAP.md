# Roadmap

Planned work for gemini-image-mcp, roughly in priority order. Everything here stays free
and open source.

## Status (2026-09-27)

Done and released on `main`:

- Security hardening: `publish_image` / `edit_image` only accept real images from safe
  locations; the cookie cache lives outside `output/`; publishing and prompt disclosure are
  opt-in; the gallery sanitizes URLs and sets a Content-Security-Policy.
- CI: read-only permissions, SHA-pinned actions, `uv sync --locked`, Dependabot.
- Robustness: GitHub 422 retry fix, no stray copies after a failed edit, input limits.
- `doctor` diagnostics (MCP tool and `gemini-image-mcp doctor [--offline]`).
- Cookies from any supported browser (Firefox, LibreWolf, Edge, Brave, Chromium, Vivaldi,
  Opera, Safari; Chrome except on Windows).

- v0.2.0 (2026-10-14): background jobs + idempotency keys, batches with backoff/retry/resume,
  exact sizes and aspect ratios, projects/tags, previews in tool results, rendered-text check,
  timeouts on every network step, `ping`, `doctor test_generation`, `gh` token fallback now
  opt-in.

## Next up

1. **Publish to PyPI** so users can run `uvx gemini-image-mcp` without cloning. Add a
   release workflow (trusted publishing, no stored token), then list the server in the MCP
   registry.
2. **Prompt helpers**: style presets, number of variations.

## Later

- Gallery: delete, favorites, export, "reuse prompt" button.
- Write the prompt into image metadata (PNG text chunk / EXIF).
- Tooling: `ruff`, `mypy`, `pre-commit`, `pip-audit`, CodeQL in CI.
- Ready-made config snippets for Gemini CLI, Cursor and VS Code in the README.
- Telegram bot on top of `gemini_image_mcp.core` (it has no MCP dependency by design).
- Refactor `Settings.load()` so it doesn't mutate `os.environ` or create directories as a
  side effect (mainly for testability).

## Maintainer notes

- The `gh auth token` fallback is now opt-in (GEMINI_GITHUB_ALLOW_GH_CLI); set a
  fine-grained token scoped to the images repo as GEMINI_GITHUB_TOKEN.
