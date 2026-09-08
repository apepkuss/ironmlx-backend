# Stable release pipeline

The `Stable Release` workflow builds an existing `vX.Y.Z` tag from a clean,
full checkout. The tag must match `VERSION`, the source App version and build
number. The publish job checks out the validated commit and rechecks the tag
and Bundle identity after downloading the candidate ZIP. MLX is built at the
commit pinned in `scripts/release-config.sh`; Rust 1.94.0 and cargo-about 0.9.1
are installed explicitly.

## Validation without Apple credentials

Tag pushes and manual dispatch with `publish=false` only build and validate.
The workflow configures the stable HTTPS feed with the repository variable
`IRONMLX_UPDATE_PUBLIC_ED_KEY`, builds the self-contained Release App and
assembles, extracts and mounts the candidate archives for content checks.
This mode does not access signing secrets, create a GitHub Release or publish
a feed. It does not qualify Developer ID, notarization or Gatekeeper behavior.
The stable public key must already be configured, including for validation.

## Production configuration

Manual dispatch with `publish=true` transfers the validated App as a ZIP to
preserve permissions, framework symlinks and signatures. Transfer requires the
existing `release-legal-gate.sh` to pass. This workflow does not enable the
public distribution flag or approve distribution materials.

The `publish` job uses the `stable-release` GitHub environment. Configure its
reviewers and allowed deployment refs before production use. Store these values
in that environment (the existing Sparkle repository secret is also supported):

| Kind | Name | Value |
|---|---|---|
| Secret | `IRONMLX_DEVELOPER_ID_P12_BASE64` | Base64 PKCS#12 containing Developer ID Application certificate and private key |
| Secret | `IRONMLX_DEVELOPER_ID_P12_PASSWORD` | Nonempty PKCS#12 export password |
| Variable | `IRONMLX_SIGNING_IDENTITY` | Full `Developer ID Application: Name (TEAMID)` identity |
| Variable | `IRONMLX_APPLE_TEAM_ID` | Certificate team ID |
| Secret | `IRONMLX_NOTARY_KEY_ID` | App Store Connect team API key ID |
| Secret | `IRONMLX_NOTARY_ISSUER_ID` | Team API issuer UUID |
| Secret | `IRONMLX_NOTARY_PRIVATE_KEY` | API key `.p8` contents |
| Secret | `IRONMLX_UPDATE_PRIVATE_ED_KEY` | Existing Sparkle Ed25519 seed, matching the build's public key |

The public key variable must remain at repository scope because the build job
does not use the production environment. No Apple credentials are generated or
configured by implementing this workflow.

## Signing and publication order

1. Import the certificate and notarization credentials into a temporary keychain.
2. Finalize Bundle metadata before sealing it. Sign Sparkle components from the
   inside out, preserve Downloader sandbox/network permissions with the explicit
   entitlement file, then sign Rust helpers and the App with hardened runtime
   and a secure timestamp. No JIT or library-validation exceptions are added.
3. Submit the App ZIP with `notarytool`, require `Accepted`, staple and validate
   the App ticket, then run signature and Gatekeeper checks. The Info.plist
   `stapled` field is a sealed declaration, never proof of notarization by itself.
4. Assemble the installer ZIP/DMG from the stapled App. Sign and notarize the
   final DMG, staple its ticket, refresh `SHA256SUMS`, then recheck both archives
   against the reference App and distribution materials.
5. Sign the app-only update ZIP and XML with Sparkle. Create an additional
   `RELEASE-SHA256SUMS` covering every uploaded payload and material file.
6. Create a draft, upload assets, check the exact asset set and download every
   file to verify hashes. Recheck the remote tag before making the draft public.
7. Verify public downloads again, then publish the update feed using the existing
   non-force, monotonic-build feed publisher.

## Failure and retry

Build/signature/notarization failures stop before Release creation. Temporary
keychain, certificate and private-key files are removed on normal failure or
termination, with an additional workflow `always()` cleanup. Apple receipts are
written under `.build/notarization`; they are not public release assets.

Upload or draft verification failures leave a draft for inspection and never
promote it. A normal rerun refuses an existing Release; it never replaces assets
or deletes the draft automatically. Inspect the failed draft and resolve it
explicitly before retrying. If a public Release exists but feed publication
failed, reuse its exact update ZIP, XML and `update.json` with
`publish-update-feed.py`; do not rebuild or overwrite that Release. An unchanged
older feed remains usable until the new feed is successfully published.

Local failure-path tests simulate Apple/GitHub operations; they verify ordering,
rejection and cleanup, not remote acceptance. Actual notarization, Gatekeeper,
public download and full App upgrade acceptance remain required after credentials
and distribution authorization are ready.

References: [Apple notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow),
[Sparkle manual signing](https://sparkle-project.org/documentation/sandboxing/).
