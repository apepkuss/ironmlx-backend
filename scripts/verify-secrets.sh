#!/usr/bin/env bash
# Scan the tracked source tree for credentials before it enters CI artifacts.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly GITLEAKS_BIN="${GITLEAKS_BIN:-gitleaks}"
readonly EXPECTED_GITLEAKS_VERSION="8.30.1"

command -v "$GITLEAKS_BIN" >/dev/null 2>&1 || {
  echo "error: gitleaks $EXPECTED_GITLEAKS_VERSION is required" >&2
  exit 1
}

version="$($GITLEAKS_BIN version)"
[ "$version" = "$EXPECTED_GITLEAKS_VERSION" ] || {
  echo "error: expected gitleaks $EXPECTED_GITLEAKS_VERSION, found $version" >&2
  exit 1
}

scan_root="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-secret-scan.XXXXXX")"
cleanup() {
  rm -rf "$scan_root"
}
trap cleanup EXIT

# Scan exactly the tracked tree, excluding ignored build products and local caches.
git -C "$REPO_ROOT" checkout-index --all --prefix="$scan_root/"
"$GITLEAKS_BIN" dir "$scan_root" \
  --config "$REPO_ROOT/.gitleaks.toml" \
  --no-banner \
  --redact \
  --max-target-megabytes 20

echo "Secret scan passed: tracked source tree contains no detected credentials"
