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

[ "$IRONMLX_PUBLIC_DISTRIBUTION_READY" = "true" ] || fail \
  "public distribution is disabled; complete P0-8B and explicitly enable it in scripts/release-config.sh"

"$SCRIPT_DIR/verify-distribution-materials.sh"

echo "IronMLX public-distribution legal gate passed"
