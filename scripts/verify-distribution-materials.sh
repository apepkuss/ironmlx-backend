#!/usr/bin/env bash
# Block public binary distribution until deferred P0-8B approval and SBOM exist.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"

fail() {
  echo "error: $*" >&2
  exit 1
}

for required in \
  "$REPO_ROOT/LICENSE" \
  "$REPO_ROOT/NOTICE" \
  "$REPO_ROOT/THIRD_PARTY_NOTICES.md" \
  "$REPO_ROOT/third-party-inventory.json" \
  "$REPO_ROOT/docs/model-license-boundary.md" \
  "$REPO_ROOT/SBOM.cdx.json"; do
  [ -s "$required" ] || fail "required distribution material is missing or empty: $required"
done

"$SCRIPT_DIR/verify-sbom.sh"

licenses_dir="$REPO_ROOT/THIRD_PARTY_LICENSES"
[ -d "$licenses_dir" ] || fail "required third-party license directory is missing: $licenses_dir"
find "$licenses_dir" -type f -size +0c -print -quit | grep -q . || \
  fail "third-party license directory contains no non-empty license text"

echo "IronMLX distribution materials verification passed"
