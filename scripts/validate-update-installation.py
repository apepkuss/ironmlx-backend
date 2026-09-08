#!/usr/bin/env python3
"""Exercise the production update manager in an isolated Sparkle test App.

Uses a temporary localhost TLS certificate in the login keychain, removed on
exit. Does not launch IronMLX's model runtime or read its user configuration.
"""
import functools
import http.server
import os
from pathlib import Path
import plistlib
import shutil
import ssl
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK = ROOT / "ironmlx-app/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
BIN = ROOT / "ironmlx-app/.build/artifacts/sparkle/Sparkle/bin"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def wait_for(predicate, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    raise TimeoutError("Sparkle installation did not complete")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    with tempfile.TemporaryDirectory(prefix="ironmlx-update-install-") as temporary:
        root = Path(temporary)
        feed = root / "feed"
        feed.mkdir()
        certificate, tls_key = root / "localhost.pem", root / "localhost.key"
        private = root / "update.key"
        public = subprocess.check_output(["swift", str(ROOT / "scripts/generate-development-update-key.swift"), str(private)], text=True).strip()
        run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes", "-days", "1",
            "-keyout", tls_key, "-out", certificate, "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fingerprint = subprocess.check_output(["openssl", "x509", "-in", str(certificate), "-noout", "-fingerprint", "-sha1"], text=True).strip().split("=")[1].replace(":", "")
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(feed)))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        app = root / "install/UpdateValidation.app"
        (app / "Contents/MacOS").mkdir(parents=True)
        (app / "Contents/Frameworks").mkdir()
        run("ditto", FRAMEWORK, app / "Contents/Frameworks/Sparkle.framework")
        marker = root / "restarted"
        # Compile the actual manager source, with only its logger supplied by this harness.
        source = root / "main.swift"
        source.write_text('''import AppKit
import Foundation
public enum IronMLXAppLogger {
    public static func info(_ message: String) { print(message) }
    public static func error(_ message: String) { print(message) }
}
@MainActor final class Delegate: NSObject, NSApplicationDelegate {
    var updater: (any AppUpdateManaging)?
    func applicationDidFinishLaunching(_ notification: Notification) {
        if Bundle.main.infoDictionary?["CFBundleVersion"] as? String == "2" && !CommandLine.arguments.contains("--offline") {
            try! Data("restarted".utf8).write(to: URL(fileURLWithPath: MARKER))
            NSApp.terminate(nil)
            return
        }
        updater = SparkleAppUpdateManager.make()
    }
}
let app = NSApplication.shared
let delegate = Delegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
'''.replace("MARKER", '"' + str(marker) + '"'))
        run("swiftc", "-swift-version", "6", "-O", "-F", FRAMEWORK.parent, "-framework", "Sparkle", "-framework", "AppKit",
            "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks",
            ROOT / "ironmlx-app/Sources/IronMLXAppCore/AppUpdateManager.swift", source,
            "-o", app / "Contents/MacOS/UpdateValidation")
        info = dict(CFBundleIdentifier="com.ironmlx.update-validation", CFBundleName="UpdateValidation",
                    CFBundleExecutable="UpdateValidation", CFBundlePackageType="APPL", CFBundleVersion="1",
                    CFBundleShortVersionString="0.1.0", LSUIElement=True, LSMinimumSystemVersion="26.2",
                    IronMLXUpdateChannel="development", SUFeedURL=f"https://127.0.0.1:{server.server_port}/appcast.xml",
                    SUPublicEDKey=public, SUEnableAutomaticChecks=True, SUAutomaticallyUpdate=True,
                    SUEnableSystemProfiling=False, SURequireSignedFeed=True, SUVerifyUpdateBeforeExtraction=True)
        plist = app / "Contents/Info.plist"
        plist.write_bytes(plistlib.dumps(info))
        run("codesign", "--force", "--deep", "--sign", "-", app, stderr=subprocess.DEVNULL)
        target = root / "target/UpdateValidation.app"
        run("ditto", app, target)
        info["CFBundleVersion"] = "2"
        (target / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
        run("codesign", "--force", "--sign", "-", target, stderr=subprocess.DEVNULL)
        run("ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", target, feed / "update.zip")
        run(BIN / "generate_appcast", "--ed-key-file", private, "--maximum-deltas", "0",
            "--download-url-prefix", f"https://127.0.0.1:{server.server_port}/", feed,
            stdout=subprocess.DEVNULL)
        user_data = root / "user-data"
        user_data.mkdir()
        (user_data / "config.json").write_text('{"sentinel":true}')
        (user_data / "model.safetensors").write_bytes(b"model sentinel")
        process = None
        keychain = Path.home() / "Library/Keychains/login.keychain-db"
        try:
            run("security", "add-trusted-cert", "-r", "trustRoot", "-p", "ssl", "-s", "127.0.0.1",
                "-k", keychain, certificate, timeout=60)
            with (root / "app.log").open("w") as log:
                process = subprocess.Popen([str(app / "Contents/MacOS/UpdateValidation"),
                                            "--ironmlx-development-update-test-marker", str(root / "install-ready")],
                                           stdout=log, stderr=log)
                wait_for(marker.exists)
                run("codesign", "--verify", "--deep", "--strict", app)
                assert plistlib.loads(plist.read_bytes())["CFBundleVersion"] == "2"
                assert (user_data / "config.json").read_text() == '{"sentinel":true}'
                assert (user_data / "model.safetensors").read_bytes() == b"model sentinel"
            # Offline checks must not replace or downgrade the newly installed App.
            installed_info = plistlib.loads(plist.read_bytes())
            installed_info["SUFeedURL"] = "https://127.0.0.1:1/appcast.xml"
            plist.write_bytes(plistlib.dumps(installed_info))
            run("codesign", "--force", "--sign", "-", app, stderr=subprocess.DEVNULL)
            offline = root / "offline"
            with (root / "offline.log").open("w") as log:
                process = subprocess.Popen([str(app / "Contents/MacOS/UpdateValidation"), "--offline",
                                            "--ironmlx-development-update-test-marker", str(offline)],
                                           stdout=log, stderr=log)
                wait_for(lambda: offline.with_suffix(".error").exists(), seconds=60)
                assert plistlib.loads(plist.read_bytes())["CFBundleVersion"] == "2"
            print("Production update manager/Sparkle integration passed: signed HTTPS feed, install 1 -> 2, relaunch, offline failure, external data preserved")
        finally:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            subprocess.run(["security", "remove-trusted-cert", str(certificate)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["security", "delete-certificate", "-Z", fingerprint, str(keychain)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
