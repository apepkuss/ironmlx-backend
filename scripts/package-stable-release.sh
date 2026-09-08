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

for tool in codesign ditto hdiutil plutil shasum spctl; do
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
spctl --assess --type execute --verbose=4 "$SOURCE_APP"

package_name="IronMLX-$PRODUCT_VERSION"
package_root="$OUTPUT_DIR/$package_name"
release_app="$package_root/IronMLX.app"
rm -rf "$OUTPUT_DIR"
mkdir -p "$package_root"
ditto "$SOURCE_APP" "$release_app"

cp "$REPO_ROOT/LICENSE" "$package_root/LICENSE"
cp "$REPO_ROOT/NOTICE" "$package_root/NOTICE"
cp "$REPO_ROOT/SBOM.cdx.json" "$package_root/SBOM.cdx.json"
cp "$REPO_ROOT/THIRD_PARTY_NOTICES.md" "$package_root/THIRD_PARTY_NOTICES.md"
cp "$REPO_ROOT/third-party-inventory.json" "$package_root/third-party-inventory.json"
cp "$REPO_ROOT/docs/model-license-boundary.md" "$package_root/model-license-boundary.md"
cp -R "$REPO_ROOT/THIRD_PARTY_LICENSES" "$package_root/THIRD_PARTY_LICENSES"

zip_path="$OUTPUT_DIR/$package_name.zip"
dmg_path="$OUTPUT_DIR/$package_name.dmg"
ditto -c -k --sequesterRsrc --keepParent "$package_root" "$zip_path"
hdiutil create \
  -volname "IronMLX $PRODUCT_VERSION" \
  -srcfolder "$package_root" \
  -format UDZO \
  -ov \
  "$dmg_path" >/dev/null

"$SCRIPT_DIR/verify-model-distribution-boundary.sh" "$zip_path"
"$SCRIPT_DIR/verify-model-distribution-boundary.sh" "$dmg_path"

(
  cd "$OUTPUT_DIR"
  shasum -a 256 \
    "$(basename "$dmg_path")" \
    "$(basename "$zip_path")" \
    LICENSE NOTICE SBOM.cdx.json THIRD_PARTY_NOTICES.md \
    third-party-inventory.json model-license-boundary.md > SHA256SUMS
  find THIRD_PARTY_LICENSES -type f -print | LC_ALL=C sort | while IFS= read -r license_file; do
    shasum -a 256 "$license_file"
  done >> SHA256SUMS
  shasum -a 256 -c SHA256SUMS >/dev/null
)

echo "Stable release assets: $OUTPUT_DIR"
