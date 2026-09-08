#!/usr/bin/env bash
# Package an explicitly labeled ad-hoc, non-notarized development preview.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT
# shellcheck source=release-config.sh
source "$SCRIPT_DIR/release-config.sh"

case "${4:-publish}" in
  validate)
    [ "${3:-}" = release-candidate ] || { echo "error: validation-only packaging requires RC channel" >&2; exit 1; }
    "$SCRIPT_DIR/verify-distribution-materials.sh"
    ;;
  publish) "$SCRIPT_DIR/release-legal-gate.sh" ;;
  *) echo "error: invalid packaging mode" >&2; exit 1 ;;
esac

readonly SOURCE_APP="$REPO_ROOT/dist/IronMLX.app"
readonly BUILD_ROOT="$REPO_ROOT/.build/development-preview-release"
readonly ASSET_DIR="$BUILD_ROOT/assets"
readonly PRODUCT_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/VERSION")"

preview_tag="${1:-}"
source_commit="${2:-}"

fail() {
  echo "error: $*" >&2
  exit 1
}

channel="${3:-development-preview}"
app_name="IronMLX Development Preview"
if [ "$channel" = "release-candidate" ]; then
  python3 "$SCRIPT_DIR/verify-release-identity.py" --candidate "$preview_tag" "$SOURCE_APP"
  [ "$source_commit" = "$(git -C "$REPO_ROOT" rev-parse HEAD)" ] || fail "RC source commit mismatch"
  app_name="IronMLX Release Candidate"
elif [ "$channel" = "development-preview" ]; then
  [[ "$preview_tag" =~ ^preview-[0-9]{8}-[0-9a-f]{7}$ ]] || fail "invalid preview tag"
  [[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source commit"
  [ "${preview_tag##*-}" = "${source_commit:0:7}" ] || fail "preview tag SHA mismatch"
else
  fail "unsupported prerelease channel: $channel"
fi
[ -d "$SOURCE_APP/Contents" ] || fail "build the App Bundle first: $SOURCE_APP"

for tool in codesign ditto hdiutil plutil shasum; do
  command -v "$tool" >/dev/null || fail "required packaging tool is missing: $tool"
done

package_name="IronMLX-$PRODUCT_VERSION-$preview_tag-ADHOC-NOT-NOTARIZED"
package_root="$BUILD_ROOT/$package_name"
preview_app="$package_root/$app_name.app"
notice_file="$package_root/DEVELOPMENT-PREVIEW-NOTICE.txt"
metadata_file="$package_root/PREVIEW-BUILD-METADATA.json"
third_party_notices="$package_root/THIRD_PARTY_NOTICES.md"
third_party_inventory="$package_root/third-party-inventory.json"
third_party_licenses="$package_root/THIRD_PARTY_LICENSES"
model_license_boundary="$package_root/model-license-boundary.md"
project_license="$package_root/LICENSE"
project_notice="$package_root/NOTICE"
sbom="$package_root/SBOM.cdx.json"

rm -rf "$BUILD_ROOT"
mkdir -p "$package_root" "$ASSET_DIR"
ditto "$SOURCE_APP" "$preview_app"
cp "$REPO_ROOT/THIRD_PARTY_NOTICES.md" "$third_party_notices"
cp "$REPO_ROOT/third-party-inventory.json" "$third_party_inventory"
cp -R "$REPO_ROOT/THIRD_PARTY_LICENSES" "$third_party_licenses"
cp "$REPO_ROOT/docs/model-license-boundary.md" "$model_license_boundary"
cp "$REPO_ROOT/LICENSE" "$project_license"
cp "$REPO_ROOT/NOTICE" "$project_notice"
cp "$REPO_ROOT/SBOM.cdx.json" "$sbom"

cat > "$notice_file" <<EOF
$app_name / $channel

警告：${IRONMLX_PREVIEW_WARNING_ZH}。
WARNING: ${IRONMLX_PREVIEW_WARNING_EN}.

This artifact is not a stable release. macOS Gatekeeper is expected to block
normal installation because the App uses only an ad-hoc signature and carries
no Apple notarization ticket.

Preview tag: $preview_tag
IronMLX source commit: $source_commit
MLX source commit: $IRONMLX_MLX_COMMIT
EOF

created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$metadata_file" <<EOF
{
  "apple_notarized": false,
  "created_at_utc": "$created_at",
  "developer_id_signed": false,
  "distribution_channel": "$channel",
  "ironmlx_commit": "$source_commit",
  "mlx_commit": "$IRONMLX_MLX_COMMIT",
  "mlx_repository": "$IRONMLX_MLX_REPOSITORY",
  "mlx_upstream_repository": "$IRONMLX_MLX_UPSTREAM_REPOSITORY",
  "mlx_upstream_revision": "$IRONMLX_MLX_UPSTREAM_REVISION",
  "product_version": "$PRODUCT_VERSION",
  "preview_tag": "$preview_tag",
  "signature_type": "ad-hoc",
  "warning": "${IRONMLX_PREVIEW_WARNING_ZH}"
}
EOF

cp "$notice_file" "$preview_app/Contents/Resources/DEVELOPMENT-PREVIEW-NOTICE.txt"
cp "$metadata_file" "$preview_app/Contents/Resources/PREVIEW-BUILD-METADATA.json"
plutil -replace CFBundleDisplayName -string "$app_name" \
  "$preview_app/Contents/Info.plist"
plutil -replace IronMLXDistributionChannel -string "$channel" \
  "$preview_app/Contents/Info.plist"
plutil -replace IronMLXDeveloperIDSigned -string unsigned "$preview_app/Contents/Info.plist"
plutil -replace IronMLXNotarizationStatus -string not_notarized "$preview_app/Contents/Info.plist"
plutil -insert IronMLXAppleNotarized -bool NO "$preview_app/Contents/Info.plist"
plutil -insert IronMLXPreviewTag -string "$preview_tag" "$preview_app/Contents/Info.plist"
if [ "$channel" = "development-preview" ]; then
  plutil -replace IronMLXSourceCommit -string "$source_commit" "$preview_app/Contents/Info.plist"
fi
source_tree_state="$(plutil -extract IronMLXSourceTreeState raw "$preview_app/Contents/Info.plist")"
[[ "$source_tree_state" =~ ^(clean|dirty)$ ]] || fail "source App has invalid IronMLXSourceTreeState"
plutil -replace IronMLXMLXCommit -string "$IRONMLX_MLX_COMMIT" "$preview_app/Contents/Info.plist"

codesign --force --deep --sign - "$preview_app"
"$SCRIPT_DIR/verify-app-bundle.sh" "$preview_app"

signature_details="$(codesign -dvvv "$preview_app" 2>&1)"
grep -Fq "Signature=adhoc" <<<"$signature_details" || fail "preview App is not ad-hoc signed"
grep -Fq "TeamIdentifier=not set" <<<"$signature_details" || \
  fail "preview App unexpectedly has a signing TeamIdentifier"

zip_path="$ASSET_DIR/$package_name.zip"
dmg_path="$ASSET_DIR/$package_name.dmg"
ditto -c -k --sequesterRsrc --keepParent "$package_root" "$zip_path"
hdiutil create \
  -volname "$app_name" \
  -srcfolder "$package_root" \
  -format UDZO \
  -ov \
  "$dmg_path" >/dev/null

"$SCRIPT_DIR/verify-model-distribution-boundary.sh" "$zip_path"
"$SCRIPT_DIR/verify-model-distribution-boundary.sh" "$dmg_path"

cp "$notice_file" "$ASSET_DIR/DEVELOPMENT-PREVIEW-NOTICE.txt"
cp "$metadata_file" "$ASSET_DIR/PREVIEW-BUILD-METADATA.json"
cp "$third_party_notices" "$ASSET_DIR/THIRD_PARTY_NOTICES.md"
cp "$third_party_inventory" "$ASSET_DIR/third-party-inventory.json"
cp -R "$third_party_licenses" "$ASSET_DIR/THIRD_PARTY_LICENSES"
cp "$model_license_boundary" "$ASSET_DIR/model-license-boundary.md"
cp "$project_license" "$ASSET_DIR/LICENSE"
cp "$project_notice" "$ASSET_DIR/NOTICE"
cp "$sbom" "$ASSET_DIR/SBOM.cdx.json"

cat > "$ASSET_DIR/RELEASE-NOTES.md" <<EOF
# ⚠️ $app_name

> **${IRONMLX_PREVIEW_WARNING_ZH}。**

This prerelease is **${IRONMLX_PREVIEW_WARNING_EN}**.

- Channel: $channel
- Product version: \`$PRODUCT_VERSION\`
- Preview tag: \`$preview_tag\`
- IronMLX immutable commit: \`$source_commit\`
- MLX immutable commit: \`$IRONMLX_MLX_COMMIT\`
- MLX source: IronMLX fork \`$IRONMLX_MLX_REPOSITORY\`
- MLX upstream base: \`$IRONMLX_MLX_UPSTREAM_REVISION\`
- Platform: Apple Silicon arm64, macOS 26.2+
- Signature: ad-hoc only; no Developer ID identity or Team ID
- Apple notarization/stapling: not performed

Gatekeeper is expected to block normal installation. This build must not be
described or redistributed as a stable release. Developer ID signing,
notarization, stapling, formal target-machine inference acceptance, and stable
release publication remain outside this preview stage.
EOF

(
  cd "$ASSET_DIR"
  shasum -a 256 \
    "$(basename "$dmg_path")" \
    "$(basename "$zip_path")" \
    DEVELOPMENT-PREVIEW-NOTICE.txt \
    PREVIEW-BUILD-METADATA.json \
    LICENSE \
    NOTICE \
    SBOM.cdx.json \
    THIRD_PARTY_NOTICES.md \
    third-party-inventory.json \
    model-license-boundary.md \
    RELEASE-NOTES.md > SHA256SUMS
  find THIRD_PARTY_LICENSES -type f -print | LC_ALL=C sort | while IFS= read -r license_file; do
    shasum -a 256 "$license_file"
  done >> SHA256SUMS
)

"$SCRIPT_DIR/verify-development-preview.sh" "$ASSET_DIR" "$preview_tag" "$source_commit" "$channel"
echo "Development preview assets: $ASSET_DIR"
