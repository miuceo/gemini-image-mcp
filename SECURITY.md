# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public GitHub issue. Use
GitHub's [private vulnerability reporting](../../security/advisories/new) for this
repository if it's enabled, or open a regular issue asking a maintainer to contact you for
a private channel. Please don't include real Gemini session cookies, `.env` contents, or
other secrets in any report.

Include:

- A description of the issue and its impact.
- Steps to reproduce, if applicable.
- The version/commit you tested against.

## Credential handling model

This project authenticates as you to the consumer Gemini web app using two session cookie
values (`__Secure-1PSID`, `__Secure-1PSIDTS`) that you supply yourself. Understanding where
those values go is the core of this project's security model:

- They are read from your local `.env` file (or your process environment) and, once
  refreshed, persisted to a local cookie cache file (`GEMINI_COOKIE_PATH`, default
  `output/.cookies`). Both are gitignored by default and never leave your machine as part
  of this project's normal operation.
- They are sent only to Google's own `gemini.google.com` endpoints, via the third-party
  [`gemini-webapi`](https://pypi.org/project/gemini-webapi/) library that this project
  depends on — the same place they'd go if you used the Gemini web app in your browser.
- This project does not transmit your cookies, prompts, or generated images to any server
  other than Google's. It has no telemetry, analytics, or third-party logging. If a
  contribution ever adds any of that, treat it as a security bug (see `CONTRIBUTING.md`).
- If you optionally set `GEMINI_PROXY`, requests to Google go through that proxy instead of
  directly — only set it to a proxy you trust, since it would see the same traffic Google
  otherwise would.

Because these cookies are equivalent to being logged in as you, treat them like a
password: don't paste them into chat, commit them, or share them with anyone.

## If you leak your cookies

If `__Secure-1PSID` / `__Secure-1PSIDTS` end up somewhere they shouldn't (a public commit,
a shared terminal log, a paste to an untrusted party):

1. Sign out of your Google account everywhere (Google Account → Security → "Manage all
   devices" / "Sign out of all sessions"). This immediately invalidates the leaked session,
   including the stolen cookies.
2. Sign back in, generate an image manually in the Gemini web app to confirm access still
   works, then re-copy fresh cookie values into your local `.env`.
3. If you accidentally committed the cookies to a git repository, rotate them as above
   *and* scrub the commit from history (e.g. `git filter-repo` or BFG) before treating the
   repo as clean — removing a later commit alone does not remove the value from history.

## Dependency on a third-party library

This project relies on `gemini-webapi`, an unofficial, reverse-engineered client for the
Gemini web app that this project does not control. Security issues in that library itself
should be reported to its own repository.
