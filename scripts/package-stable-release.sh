#!/usr/bin/env bash
# Package a signed and notarized IronMLX stable release.
# Developer ID signing and notarization are performed before this gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"

readonly SOURCE_APP="${1:-$REPO_ROOT/dist/IronMLX.app}"
readonly OUTPUT_DIR="${2:-$REPO_ROOT/.build/stable-release}"
readonly PRODUCT_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/VERSION")"

fail() {
  echo "error: $*" >&2
  exit 1
}

python3 "$SCRIPT_DIR/verify-release-identity.py" "${3:-v$PRODUCT_VERSION}" "$SOURCE_APP"
"$SCRIPT_DIR/release-legal-gate.sh"
"$SCRIPT_DIR/verify-version-consistency.sh"
[ -d "$SOURCE_APP/Contents" ] || fail "signed App Bundle is missing: $SOURCE_APP"

for tool in codesign ditto hdiutil plutil shasum spctl xcrun; do
  command -v "$tool" >/dev/null || fail "required packaging tool is missing: $tool"
done

developer_id_status="$(plutil -extract IronMLXDeveloperIDSigned raw "$SOURCE_APP/Contents/Info.plist")"
[ "$developer_id_status" = "developer_id" ] || \
  fail "stable release requires IronMLXDeveloperIDSigned=developer_id"
notarization_status="$(plutil -extract IronMLXNotarizationStatus raw "$SOURCE_APP/Contents/Info.plist")"
[ "$notarization_status" = "stapled" ] || \
  fail "stable release requires IronMLXNotarizationStatus=stapled"
distribution_channel="$(plutil -extract IronMLXDistributionChannel raw "$SOURCE_APP/Contents/Info.plist")"
[ "$distribution_channel" = "stable" ] || \
  fail "stable release requires IronMLXDistributionChannel=stable"

"$SCRIPT_DIR/verify-app-bundle.sh" "$SOURCE_APP"
codesign --verify --deep --strict --verbose=2 "$SOURCE_APP"
xcrun stapler validate "$SOURCE_APP"
spctl --assess --type execute --verbose=4 "$SOURCE_APP"

python3 "$SCRIPT_DIR/release-archives.py" assemble "$SOURCE_APP" "$OUTPUT_DIR"

echo "Stable release assets: $OUTPUT_DIR"
