# Security Policy

## Supported Versions

WhisperSync is in active development; security fixes target the latest `main`.

## Reporting a Vulnerability

Please **do not** open a public issue for security problems.

Instead, report privately via GitHub's
[private vulnerability reporting](../../security/advisories/new) (Security →
Advisories → "Report a vulnerability"). Include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- affected version/commit and your environment.

We aim to acknowledge reports within a few days and will keep you informed about
the fix and disclosure timeline.

## Scope notes

WhisperSync runs entirely locally. The only network access is a one-time download
of the Whisper model weights from Hugging Face; there is no telemetry. Be mindful
that generated `.fcpxml` files and exported transcripts contain `file://` paths
and the transcribed text of your recordings — treat them as you would the source
media.
