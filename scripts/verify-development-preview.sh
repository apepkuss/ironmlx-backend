#!/usr/bin/env bash
# Verify preview labels, ad-hoc signature, archives, checksums, and Bundle gates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly PRODUCT_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/VERSION")"
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"

asset_dir="${1:-}"
preview_tag="${2:-}"
source_commit="${3:-}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ -d "$asset_dir" ] || fail "asset directory not found: $asset_dir"
channel="${4:-development-preview}"
app_name="IronMLX Development Preview"
if [ "$channel" = "release-candidate" ]; then
  python3 "$SCRIPT_DIR/verify-release-identity.py" --candidate "$preview_tag"
  [ "$source_commit" = "$(git -C "$REPO_ROOT" rev-parse HEAD)" ] || fail "RC source commit mismatch"
  app_name="IronMLX Release Candidate"
elif [ "$channel" = "development-preview" ]; then
  [[ "$preview_tag" =~ ^preview-[0-9]{8}-[0-9a-f]{7}$ ]] || fail "invalid preview tag"
else
  fail "unsupported prerelease channel: $channel"
fi
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source commit"

package_name="IronMLX-$PRODUCT_VERSION-$preview_tag-ADHOC-NOT-NOTARIZED"
zip_path="$asset_dir/$package_name.zip"
dmg_path="$asset_dir/$package_name.dmg"
for required in \
  "$zip_path" \
  "$dmg_path" \
  "$asset_dir/DEVELOPMENT-PREVIEW-NOTICE.txt" \
  "$asset_dir/PREVIEW-BUILD-METADATA.json" \
  "$asset_dir/LICENSE" \
  "$asset_dir/NOTICE" \
  "$asset_dir/SBOM.cdx.json" \
  "$asset_dir/THIRD_PARTY_NOTICES.md" \
  "$asset_dir/third-party-inventory.json" \
  "$asset_dir/model-license-boundary.md" \
  "$asset_dir/RELEASE-NOTES.md" \
  "$asset_dir/SHA256SUMS"; do
  [ -f "$required" ] || fail "required preview asset is missing: $required"
done

grep -Fq "$IRONMLX_PREVIEW_WARNING_ZH" "$asset_dir/DEVELOPMENT-PREVIEW-NOTICE.txt" || \
  fail "preview notice does not contain the required warning"
grep -Fq "$IRONMLX_PREVIEW_WARNING_ZH" "$asset_dir/RELEASE-NOTES.md" || \
  fail "release notes do not contain the required warning"

[ "$(plutil -extract distribution_channel raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "$channel" ] || fail "metadata channel mismatch"

[ "$(plutil -extract preview_tag raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "$preview_tag" ] || \
  fail "metadata preview tag mismatch"
[ "$(plutil -extract ironmlx_commit raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "$source_commit" ] || \
  fail "metadata source commit mismatch"
[ "$(plutil -extract product_version raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = \
  "$PRODUCT_VERSION" ] || fail "metadata product version mismatch"
grep -Fq "Product version: \`$PRODUCT_VERSION\`" "$asset_dir/RELEASE-NOTES.md" || \
  fail "release notes product version mismatch"
[ "$(plutil -extract developer_id_signed raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "false" ] || \
  fail "metadata must declare Developer ID signing disabled"
[ "$(plutil -extract apple_notarized raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = "false" ] || \
  fail "metadata must declare Apple notarization disabled"
[ "$(plutil -extract mlx_repository raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = \
  "$IRONMLX_MLX_REPOSITORY" ] || fail "metadata MLX fork repository mismatch"
[ "$(plutil -extract mlx_upstream_revision raw "$asset_dir/PREVIEW-BUILD-METADATA.json")" = \
  "$IRONMLX_MLX_UPSTREAM_REVISION" ] || fail "metadata MLX upstream revision mismatch"
diff -q "$REPO_ROOT/THIRD_PARTY_NOTICES.md" "$asset_dir/THIRD_PARTY_NOTICES.md" >/dev/null || \
  fail "preview notice asset differs from the verified source material"
diff -q "$REPO_ROOT/LICENSE" "$asset_dir/LICENSE" >/dev/null || \
  fail "preview LICENSE asset differs from the verified source material"
diff -q "$REPO_ROOT/NOTICE" "$asset_dir/NOTICE" >/dev/null || \
  fail "preview NOTICE asset differs from the verified source material"
diff -q "$REPO_ROOT/SBOM.cdx.json" "$asset_dir/SBOM.cdx.json" >/dev/null || \
  fail "preview SBOM asset differs from the verified source material"
diff -q "$REPO_ROOT/third-party-inventory.json" "$asset_dir/third-party-inventory.json" >/dev/null || \
  fail "preview inventory asset differs from the verified source material"
diff -qr "$REPO_ROOT/THIRD_PARTY_LICENSES" "$asset_dir/THIRD_PARTY_LICENSES" >/dev/null || \
  fail "preview license assets differ from the verified source material"
diff -q "$REPO_ROOT/docs/model-license-boundary.md" "$asset_dir/model-license-boundary.md" >/dev/null || \
  fail "preview model license boundary differs from the verified source material"

(
  cd "$asset_dir"
  shasum -a 256 -c SHA256SUMS
)

if zipinfo -1 "$zip_path" | grep -E '(^/|(^|/)\.\.(/|$)|/Users/)' >/dev/null; then
  fail "ZIP contains an unsafe or developer-specific path"
fi

"$SCRIPT_DIR/verify-model-distribution-boundary.sh" "$zip_path"
"$SCRIPT_DIR/verify-model-distribution-boundary.sh" "$dmg_path"

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/ironmlx-preview-verify.XXXXXX")"
zip_extract="$temp_root/zip"
dmg_mount="$temp_root/dmg"
mounted=0

cleanup() {
  if [ "$mounted" -eq 1 ]; then
    hdiutil detach "$dmg_mount" -quiet || true
  fi
  rm -rf "$temp_root"
}
trap cleanup EXIT

verify_preview_app() {
  local package_root="$1"
  local preview_app="$package_root/$app_name.app"
  local signature_details

  diff -q "$REPO_ROOT/THIRD_PARTY_NOTICES.md" "$package_root/THIRD_PARTY_NOTICES.md" >/dev/null || \
    fail "archive root third-party notices differ from the verified source material"
  diff -q "$REPO_ROOT/LICENSE" "$package_root/LICENSE" >/dev/null || \
    fail "archive root LICENSE differs from the verified source material"
  diff -q "$REPO_ROOT/NOTICE" "$package_root/NOTICE" >/dev/null || \
    fail "archive root NOTICE differs from the verified source material"
  diff -q "$REPO_ROOT/SBOM.cdx.json" "$package_root/SBOM.cdx.json" >/dev/null || \
    fail "archive root SBOM differs from the verified source material"
  diff -q "$REPO_ROOT/third-party-inventory.json" "$package_root/third-party-inventory.json" >/dev/null || \
    fail "archive root third-party inventory differs from the verified source material"
  diff -qr "$REPO_ROOT/THIRD_PARTY_LICENSES" "$package_root/THIRD_PARTY_LICENSES" >/dev/null || \
    fail "archive root third-party licenses differ from the verified source material"
  diff -q "$REPO_ROOT/docs/model-license-boundary.md" "$package_root/model-license-boundary.md" >/dev/null || \
    fail "archive root model license boundary differs from the verified source material"
  [ -d "$preview_app/Contents" ] || fail "preview App is missing: $preview_app"
  if [ "$channel" = "release-candidate" ]; then
    python3 "$SCRIPT_DIR/verify-release-identity.py" --candidate "$preview_tag" "$preview_app"
    [ "$(plutil -extract IronMLXPreviewTag raw "$preview_app/Contents/Info.plist")" = "$preview_tag" ] || fail "RC Bundle tag mismatch"
    diff -q "$asset_dir/PREVIEW-BUILD-METADATA.json" "$preview_app/Contents/Resources/PREVIEW-BUILD-METADATA.json" || fail "RC Bundle metadata mismatch"
    diff -q "$asset_dir/PREVIEW-BUILD-METADATA.json" "$package_root/PREVIEW-BUILD-METADATA.json" || fail "RC archive metadata mismatch"
  fi
  "$SCRIPT_DIR/verify-app-bundle.sh" "$preview_app"
  "$SCRIPT_DIR/verify-model-distribution-boundary.sh" "$preview_app"
  [ "$(plutil -extract CFBundleDisplayName raw "$preview_app/Contents/Info.plist")" = \
    "$app_name" ] || fail "preview display name is not explicit"
  [ "$(plutil -extract IronMLXDistributionChannel raw "$preview_app/Contents/Info.plist")" = \
    "$channel" ] || fail "preview distribution channel is missing"
  [ "$(plutil -extract IronMLXDeveloperIDSigned raw "$preview_app/Contents/Info.plist")" = "unsigned" ] || \
    fail "preview App must declare Developer ID signing disabled"
  [ "$(plutil -extract IronMLXNotarizationStatus raw "$preview_app/Contents/Info.plist")" = \
    "not_notarized" ] || fail "preview App must declare notarization not performed"
  [ "$(plutil -extract IronMLXAppleNotarized raw "$preview_app/Contents/Info.plist")" = "false" ] || \
    fail "preview App must declare Apple notarization disabled"
  grep -Fq "$IRONMLX_PREVIEW_WARNING_ZH" \
    "$preview_app/Contents/Resources/DEVELOPMENT-PREVIEW-NOTICE.txt" || \
    fail "preview App does not contain the required warning"

  signature_details="$(codesign -dvvv "$preview_app" 2>&1)"
  grep -Fq "Signature=adhoc" <<<"$signature_details" || fail "preview App is not ad-hoc signed"
  grep -Fq "TeamIdentifier=not set" <<<"$signature_details" || \
    fail "preview App unexpectedly has a signing TeamIdentifier"
}

mkdir -p "$zip_extract" "$dmg_mount"
ditto -x -k "$zip_path" "$zip_extract"
verify_preview_app "$zip_extract/$package_name"

hdiutil attach "$dmg_path" -readonly -nobrowse -mountpoint "$dmg_mount" -quiet
mounted=1
verify_preview_app "$dmg_mount"
hdiutil detach "$dmg_mount" -quiet
mounted=0

echo "IronMLX development preview verification passed"
