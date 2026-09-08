#!/usr/bin/env swift
import CryptoKit
import Foundation

guard CommandLine.arguments.count == 2,
      let raw = Data(base64Encoded: try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines)), raw.count == 32 else {
    FileHandle.standardError.write(Data("error: expected a Sparkle 32-byte Ed25519 seed file\n".utf8))
    exit(1)
}
let key = try Curve25519.Signing.PrivateKey(rawRepresentation: raw)
print(key.publicKey.rawRepresentation.base64EncodedString())
