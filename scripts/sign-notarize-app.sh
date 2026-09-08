#!/usr/bin/env bash
# Developer ID signing and Apple notarization; secrets are supplied via env.
set -euo pipefail
set +x
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly APP="$(realpath "${1:?App or DMG path required}")"
case "$APP" in
  *.app) kind=app ;;
  *.dmg) kind=dmg ;;
  *) echo 'error: expected App or DMG' >&2; exit 1 ;;
esac
readonly WORK="${RUNNER_TEMP:?RUNNER_TEMP required}/ironmlx-signing"
readonly KEYCHAIN="$WORK/signing.keychain-db"
for name in IRONMLX_DEVELOPER_ID_P12_BASE64 IRONMLX_DEVELOPER_ID_P12_PASSWORD IRONMLX_SIGNING_IDENTITY IRONMLX_APPLE_TEAM_ID IRONMLX_NOTARY_KEY_ID IRONMLX_NOTARY_ISSUER_ID IRONMLX_NOTARY_PRIVATE_KEY; do
  [ -n "${!name:-}" ] || { echo "error: missing $name" >&2; exit 1; }
done
[[ "$IRONMLX_SIGNING_IDENTITY" = "Developer ID Application: "*" ($IRONMLX_APPLE_TEAM_ID)" ]] || {
  echo 'error: expected Developer ID Application identity for configured team' >&2; exit 1;
}
[ ! -e "$WORK" ] || { echo 'error: signing workspace already exists' >&2; exit 1; }
umask 077
mkdir -p "$WORK"
cleanup() {
  security delete-keychain "$KEYCHAIN" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s' "$IRONMLX_DEVELOPER_ID_P12_BASE64" | base64 --decode > "$WORK/certificate.p12"
printf '%s\n' "$IRONMLX_NOTARY_PRIVATE_KEY" > "$WORK/notary.p8"
keychain_password="$(openssl rand -hex 32)"
security create-keychain -p "$keychain_password" "$KEYCHAIN"
security set-keychain-settings -lut 21600 "$KEYCHAIN"
security unlock-keychain -p "$keychain_password" "$KEYCHAIN"
security import "$WORK/certificate.p12" -k "$KEYCHAIN" -P "$IRONMLX_DEVELOPER_ID_P12_PASSWORD" -T /usr/bin/codesign >/dev/null
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$keychain_password" "$KEYCHAIN" >/dev/null
xcrun notarytool store-credentials ironmlx-release --key "$WORK/notary.p8" \
  --key-id "$IRONMLX_NOTARY_KEY_ID" --issuer "$IRONMLX_NOTARY_ISSUER_ID" --keychain "$KEYCHAIN" >/dev/null

sign() {
  codesign --force --sign "$IRONMLX_SIGNING_IDENTITY" --keychain "$KEYCHAIN" \
    --options runtime --timestamp "$@"
}
if [ "$kind" = app ]; then
  # All sealed metadata must be final BEFORE signing. The claimed stapled status
  # is accepted only after the actual ticket and Gatekeeper checks below succeed.
  plutil -replace IronMLXDeveloperIDSigned -string developer_id "$APP/Contents/Info.plist"
  plutil -replace IronMLXNotarizationStatus -string stapled "$APP/Contents/Info.plist"
  plutil -replace IronMLXDistributionChannel -string stable "$APP/Contents/Info.plist"
  sparkle="$APP/Contents/Frameworks/Sparkle.framework/Versions/B"
  sign "$sparkle/XPCServices/Installer.xpc"
  sign --entitlements "$SCRIPT_DIR/../ironmlx-app/Packaging/SparkleDownloader.entitlements" "$sparkle/XPCServices/Downloader.xpc"
  sign "$sparkle/Autoupdate"
  sign "$sparkle/Updater.app"
  sign "$APP/Contents/Frameworks/Sparkle.framework"
  sign "$APP/Contents/Helpers/ironmlx"
  sign "$APP/Contents/Helpers/iron-bench"
  sign "$APP"
else
  codesign --force --sign "$IRONMLX_SIGNING_IDENTITY" --keychain "$KEYCHAIN" --timestamp "$APP"
fi
codesign --verify --deep --strict "$APP"
codesign -dv "$APP" 2>&1 | grep -Fx "TeamIdentifier=$IRONMLX_APPLE_TEAM_ID"

payload="$APP"
if [ "$kind" = app ]; then
  ditto -c -k --sequesterRsrc --keepParent "$APP" "$WORK/notary.zip"
  payload="$WORK/notary.zip"
fi
mkdir -p "$SCRIPT_DIR/../.build/notarization"
result="$SCRIPT_DIR/../.build/notarization/$kind.json"
xcrun notarytool submit "$payload" --keychain-profile ironmlx-release \
  --keychain "$KEYCHAIN" --wait --timeout 60m --output-format json > "$result"
python3 - "$result" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
if result.get('status') != 'Accepted':
    raise SystemExit('error: notarization not Accepted; inspect .build/notarization receipt')
PY
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"
codesign --verify --deep --strict "$APP"
if [ "$kind" = app ]; then
  spctl --assess --type execute --verbose=4 "$APP"
else
  spctl --assess --type open --context context:primary-signature --verbose=4 "$APP"
fi
