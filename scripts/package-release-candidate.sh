#!/usr/bin/env bash
# Package an existing RC tag as an explicitly ad-hoc, non-notarized prerelease.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
rc_tag="${1:?usage: package-release-candidate.sh vX.Y.Z-rc.N}"
mode="${2:-validate}"
case "$mode" in validate|publish) ;; *) echo "error: expected validate or publish" >&2; exit 1 ;; esac
python3 "$SCRIPT_DIR/verify-release-identity.py" --candidate "$rc_tag" "$REPO_ROOT/dist/IronMLX.app"
exec "$SCRIPT_DIR/package-development-preview.sh" \
  "$rc_tag" "$(git -C "$REPO_ROOT" rev-parse HEAD)" release-candidate "$mode"
