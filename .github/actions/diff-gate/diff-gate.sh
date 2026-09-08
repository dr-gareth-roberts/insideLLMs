#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '::error::This legacy action is retired. Use the repository-root action and the split comment workflow documented in wiki/tutorials/CI-Integration.md.' >&2
exit 1
