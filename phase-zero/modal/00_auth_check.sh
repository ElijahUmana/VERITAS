#!/usr/bin/env bash
# Phase-Zero #0 — Modal auth gate-check. Fast, read-only, proves we are authed
# and which workspace/credits we are on BEFORE spending GPU seconds.
#
# Run:  bash phase-zero/modal/00_auth_check.sh
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODAL="${MODAL_BIN:-$R/.venv/bin/modal}"
[ -x "$MODAL" ] || MODAL="$HOME/.local/bin/modal"
[ -x "$MODAL" ] || MODAL="modal"

echo "=== Modal auth gate-check @ $(date -u +%FT%TZ) ==="
echo "modal bin: $MODAL"
"$MODAL" --version

if [ ! -f "$HOME/.modal.toml" ]; then
  echo
  echo "BLOCKED: ~/.modal.toml is missing — NOT AUTHED YET."
  echo "  Fix (either):"
  echo "    1) $MODAL setup            # browser OAuth"
  echo "    2) $MODAL token set --token-id <id> --token-secret <secret>"
  echo "  Then redeem \$250 credit:    open https://modal.com/credits   code SFZ-ZLT-F8E"
  exit 1
fi

echo
echo "--- modal profile current ---";  "$MODAL" profile current 2>&1 || true
echo "--- modal profile list ---";     "$MODAL" profile list 2>&1 || true
echo "--- modal app list (proves a real authenticated API call) ---"
if "$MODAL" app list 2>&1; then
  echo
  echo "PASS  Modal auth verified — workspace reachable. Proceed to live GPU tests."
else
  echo
  echo "FAIL  ~/.modal.toml exists but 'modal app list' failed — token invalid/expired."
  exit 1
fi
