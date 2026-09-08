# Automatic updates

IronMLX uses the pinned Sparkle 2.9.6 runtime. `stable` and `release-candidate`
builds use separate signed HTTPS feeds; `development` remains restricted to
loopback HTTPS. Ordinary local builds default to updates disabled.

## Version and channel contract

- `CFBundleShortVersionString` stays equal to `VERSION` (`X.Y.Z`).
- `CFBundleVersion` is a positive integer and must increase for every published
  update, including successive RCs of the same product version. Use
  `scripts/bump-version.sh X.Y.Z N` with an explicit higher N for another RC.
- RC tags use `vX.Y.Z-rc.N`; stable tags use `vX.Y.Z`. The display version in the
  feed includes the RC suffix, but Sparkle compares the numeric build number.
- The two feeds are `https://raw.githubusercontent.com/<owner>/<repo>/updates/stable.xml`
  and `https://raw.githubusercontent.com/<owner>/<repo>/updates/release-candidate.xml`.
  The publisher maintains the dedicated `updates` branch; it does not change
  source branches or release tags. RC items also carry Sparkle's
  `release-candidate` channel marker. Stable clients do not opt into that marker.
- Switching an installed RC to stable is a deliberate install/channel change,
  not an automatic promotion through the RC feed.

## Persistent signing configuration

The update Ed25519 key is separate from Apple Developer ID. Keep one persistent
32-byte seed for published updates. The existing key utility can generate that
format; its development name does not make generated keys temporary:

```bash
swift scripts/generate-development-update-key.swift /absolute/private/path/update-key
```

The file must not already exist. The utility writes it with mode 0600 and prints
only its public key. Store that public key as GitHub repository variable
`IRONMLX_UPDATE_PUBLIC_ED_KEY`, and the seed file's contents as secret
`IRONMLX_UPDATE_PRIVATE_ED_KEY`. Retain the private seed outside the checkout;
changing the key after distribution requires a separate migration plan.

The publisher refuses to sign if the supplied seed does not match the public
key embedded in the App. No persistent key or GitHub setting is created by
running the source tests.

For an eventual signed stable Bundle, provide before building/signing:

```bash
IRONMLX_UPDATE_CHANNEL=stable \
IRONMLX_UPDATE_FEED_URL=https://raw.githubusercontent.com/OWNER/REPO/updates/stable.xml \
IRONMLX_UPDATE_PUBLIC_ED_KEY=PUBLIC_KEY \
scripts/build-app-bundle.sh
```

RC CI sets the corresponding RC URL and channel. Validation-only runs without
a configured public key use an ephemeral public key and do not upload their
installer artifacts, even if distribution is otherwise authorized. Publishing
requires the persistent public key and private-key secret.

## Publication ordering

1. Run the existing identity, archive and distribution gates.
2. Create an App-only update ZIP from the exact packaged/signed App. Do not use
   the user-facing installer ZIP as an update archive.
3. Sign the ZIP and the XML feed with Sparkle, verify both signatures and bind
   the outputs in `update.json`. No delta updates are generated.
4. Upload the update ZIP, signed XML and metadata alongside the release assets.
5. Download the published ZIP and compare its SHA-256; verify the tag's source
   commit, release visibility and prerelease status.
6. Publish the signed XML to the feed branch only after those checks pass.
   Reject non-increasing builds and use a non-force reference update; concurrent
   changes fail rather than overwrite another channel's feed.

The RC workflow has this sequence in its explicit publication job. The stable
workflow builds, signs and notarizes the Bundle before packaging updates; see
[Stable release pipeline](stable-release-pipeline.md). This does not remove
any Developer ID, notarization or distribution authorization gate.

If Release creation succeeded but feed publication failed, reuse the exact
published update ZIP/XML/`update.json` and rerun `publish-update-feed.py`; do not
regenerate or overwrite the release. An identical already-published manifest is
an idempotent retry. The feed starts returning updates only after its first
successful publication; a missing/offline feed must leave the App installed.

## Verification boundaries

`test_app_updates.py` validates channel/version policy, monotonic build rules,
real Sparkle archive/feed signatures, key mismatch and tamper rejection.
`validate-update-installation.py` compiles the production update manager into an
isolated App, tests a localhost HTTPS upgrade and relaunch, and leaves IronMLX
user data untouched. It temporarily trusts a localhost TLS certificate in the
login keychain and removes it afterward. This harness does not qualify production
model runtime recovery, M1 Pro installation, or a notarized upgrade.

The full App's existing `applicationShouldTerminate` cancels downloads and awaits
`backend.stopForAppQuit()` before replying to termination. Final App acceptance
must still verify this with the production helper, including active requests and
real configuration/model preservation on both target machines.
