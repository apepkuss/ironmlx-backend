import Foundation
import Sparkle

@MainActor
public protocol AppUpdateManaging: AnyObject {
    var canCheckForUpdates: Bool { get }

    func checkForUpdates(_ sender: Any?)
}

@MainActor
public final class DisabledAppUpdateManager: AppUpdateManaging {
    public let canCheckForUpdates = false

    public init() {}

    public func checkForUpdates(_ sender: Any?) {}
}

enum AppUpdateConfigurationError: LocalizedError, Equatable {
    case unsupportedChannel(String)
    case missingValue(String)
    case invalidDevelopmentFeed(String)
    case invalidPublicFeed(String)
    case invalidPublicEdKey
    case automaticUpdatesNotEnabled
    case signatureVerificationNotRequired
    case systemProfilingEnabled

    var errorDescription: String? {
        switch self {
        case let .unsupportedChannel(channel):
            "Unsupported update channel: \(channel)"
        case let .missingValue(key):
            "Missing required update configuration: \(key)"
        case let .invalidDevelopmentFeed(value):
            "Development update feed must use HTTPS on a loopback host: \(value)"
        case let .invalidPublicFeed(value):
            "Public update feed must use HTTPS without credentials on a non-loopback host: \(value)"
        case .invalidPublicEdKey:
            "Update public EdDSA key must decode to 32 bytes"
        case .automaticUpdatesNotEnabled:
            "Update-enabled builds must enable automatic update checks and downloads"
        case .signatureVerificationNotRequired:
            "Updates must require signed feeds and pre-extraction verification"
        case .systemProfilingEnabled:
            "Updates must not send system profile information"
        }
    }
}

struct AppUpdateConfiguration: Equatable {
    static let channelKey = "IronMLXUpdateChannel"
    static let developmentChannel = "development"

    let channel: String
    let feedURL: URL
    let publicEdKey: String

    init(infoDictionary: [String: Any]) throws {
        let channel = Self.nonEmptyString(infoDictionary[Self.channelKey])
        guard let channel else {
            throw AppUpdateConfigurationError.missingValue(Self.channelKey)
        }
        guard [Self.developmentChannel, "stable", "release-candidate"].contains(channel) else {
            throw AppUpdateConfigurationError.unsupportedChannel(channel)
        }

        guard let feed = Self.nonEmptyString(infoDictionary["SUFeedURL"]) else {
            throw AppUpdateConfigurationError.missingValue("SUFeedURL")
        }
        let parsedURL = URL(string: feed)
        let validTransport = parsedURL?.scheme?.lowercased() == "https"
            && parsedURL?.host != nil && parsedURL?.user == nil
            && parsedURL?.password == nil && parsedURL?.fragment == nil
        if channel == Self.developmentChannel {
            guard validTransport, let host = parsedURL?.host, Self.isLoopbackHost(host) else {
                throw AppUpdateConfigurationError.invalidDevelopmentFeed(feed)
            }
        } else {
            guard validTransport, let host = parsedURL?.host, !Self.isLoopbackHost(host) else {
                throw AppUpdateConfigurationError.invalidPublicFeed(feed)
            }
        }
        guard let feedURL = parsedURL else {
            throw AppUpdateConfigurationError.invalidPublicFeed(feed)
        }
        guard let publicEdKey = Self.nonEmptyString(infoDictionary["SUPublicEDKey"]) else {
            throw AppUpdateConfigurationError.missingValue("SUPublicEDKey")
        }
        guard Data(base64Encoded: publicEdKey)?.count == 32 else {
            throw AppUpdateConfigurationError.invalidPublicEdKey
        }
        guard infoDictionary["SUEnableAutomaticChecks"] as? Bool == true,
              infoDictionary["SUAutomaticallyUpdate"] as? Bool == true
        else {
            throw AppUpdateConfigurationError.automaticUpdatesNotEnabled
        }
        guard infoDictionary["SURequireSignedFeed"] as? Bool == true,
              infoDictionary["SUVerifyUpdateBeforeExtraction"] as? Bool == true
        else {
            throw AppUpdateConfigurationError.signatureVerificationNotRequired
        }
        guard infoDictionary["SUEnableSystemProfiling"] as? Bool == false else {
            throw AppUpdateConfigurationError.systemProfilingEnabled
        }

        self.channel = channel
        self.feedURL = feedURL
        self.publicEdKey = publicEdKey
    }

    private static func nonEmptyString(_ value: Any?) -> String? {
        guard let value = value as? String else {
            return nil
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func isLoopbackHost(_ host: String) -> Bool {
        let normalized = host.lowercased()
        return normalized == "localhost" || normalized == "127.0.0.1" || normalized == "::1"
    }
}

@MainActor
public final class SparkleAppUpdateManager: NSObject, AppUpdateManaging, SPUUpdaterDelegate {
    private var controller: SPUStandardUpdaterController!
    private let developmentTestMarkerURL: URL?
    private let channel: String

    public var canCheckForUpdates: Bool {
        controller.updater.canCheckForUpdates
    }

    private init(developmentTestMarkerURL: URL?, channel: String) {
        self.channel = channel
        self.developmentTestMarkerURL = developmentTestMarkerURL
        super.init()
    }

    public static func make(bundle: Bundle = .main) -> any AppUpdateManaging {
        guard let infoDictionary = bundle.infoDictionary,
              infoDictionary[AppUpdateConfiguration.channelKey] != nil
        else {
            IronMLXAppLogger.info("Automatic updates are disabled for this build")
            return DisabledAppUpdateManager()
        }

        let configuration: AppUpdateConfiguration
        do {
            configuration = try AppUpdateConfiguration(infoDictionary: infoDictionary)
        } catch {
            IronMLXAppLogger.error("Automatic update configuration rejected: \(error.localizedDescription)")
            return DisabledAppUpdateManager()
        }

        let markerURL = configuration.channel == AppUpdateConfiguration.developmentChannel
            ? parseDevelopmentUpdateTestMarkerURL(arguments: ProcessInfo.processInfo.arguments) : nil
        let manager = SparkleAppUpdateManager(developmentTestMarkerURL: markerURL, channel: configuration.channel)
        manager.controller = SPUStandardUpdaterController(
            startingUpdater: false,
            updaterDelegate: manager,
            userDriverDelegate: nil
        )
        do {
            try manager.controller.updater.start()
        } catch {
            manager.recordDevelopmentUpdateError(error)
            IronMLXAppLogger.error(
                "Failed to start automatic updater: \(error.localizedDescription)"
            )
            return DisabledAppUpdateManager()
        }
        IronMLXAppLogger.info(
            "Automatic updater started: checks=\(manager.controller.updater.automaticallyChecksForUpdates) "
                + "downloads=\(manager.controller.updater.automaticallyDownloadsUpdates) "
                + "allowed=\(manager.controller.updater.allowsAutomaticUpdates)"
        )
        if markerURL != nil {
            IronMLXAppLogger.info("Starting development automatic update check")
            manager.controller.updater.checkForUpdatesInBackground()
        }
        return manager
    }

    public func checkForUpdates(_ sender: Any?) {
        controller.checkForUpdates(sender)
    }

    public func allowedChannels(for updater: SPUUpdater) -> Set<String> {
        channel == "release-candidate" ? ["release-candidate"] : []
    }


    public func updater(
        _ updater: SPUUpdater,
        willInstallUpdateOnQuit item: SUAppcastItem,
        immediateInstallationBlock immediateInstallHandler: @escaping () -> Void
    ) -> Bool {
        guard let developmentTestMarkerURL else {
            return false
        }
        do {
            try Data("ready\n".utf8).write(to: developmentTestMarkerURL, options: .atomic)
        } catch {
            IronMLXAppLogger.error(
                "Failed to write development update marker: \(error.localizedDescription)"
            )
            return false
        }
        DispatchQueue.main.async {
            immediateInstallHandler()
        }
        return true
    }

    public func updater(_ updater: SPUUpdater, didAbortWithError error: Error) {
        recordDevelopmentUpdateError(error)
    }

    private func recordDevelopmentUpdateError(_ error: Error) {
        guard let developmentTestMarkerURL else { return }
        let errorURL = developmentTestMarkerURL.appendingPathExtension("error")
        do {
            try Data("\(error.localizedDescription)\n".utf8).write(to: errorURL, options: .atomic)
        } catch {
            IronMLXAppLogger.error(
                "Failed to write development update error marker: \(error.localizedDescription)"
            )
        }
    }
}

private func parseDevelopmentUpdateTestMarkerURL(arguments: [String]) -> URL? {
    guard let flagIndex = arguments.firstIndex(of: "--ironmlx-development-update-test-marker"),
          arguments.indices.contains(flagIndex + 1)
    else {
        return nil
    }
    let path = arguments[flagIndex + 1].trimmingCharacters(in: .whitespacesAndNewlines)
    guard path.hasPrefix("/"), !path.isEmpty else {
        return nil
    }
    return URL(fileURLWithPath: path)
}
