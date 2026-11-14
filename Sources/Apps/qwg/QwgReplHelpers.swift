// QwgReplHelpers.swift — Interactive slash-command handlers for qwg repl.

import ArgumentParser
import Foundation
import QWCli
import QWCore
import QWEngine
import QWHead
import QWKV
import QWQuant
import QWServer
import QWToken

extension Repl {
    enum SlashAction { case continueLoop, exit }

    static func handleSlashCommand(
        cmd: String,
        arg: String,
        cliColor: QWCli.Color,
        engine: QWEngine,
        runtimeAssets: QWModelRuntimeAssets,
        cliCfg: QWResolvedCliEngineConfig,
        showThinking: Bool,
        hideThinking: Bool,
        conversation: QWConversation?,
        visionController: QWCLIVisionController?,
        history: inout [QWJson],
        activeImageHistory: inout [String],
        recalledVaultPaths: inout Set<String>,
        pendingTurnImage: inout String?,
        activeMemoryPack: inout QWResolvedCliMemoryPack?,
        mountedMemoryTokens: inout [Int],
        currentSystem: inout String?,
        isThinking: inout Bool,
        displayThinking: inout Bool,
        currentReasoningEffort: inout String,
        currentMaxTokens: inout Int?,
        showStats: inout Bool
    ) -> SlashAction {
        switch cmd {
        case "/exit", "/quit", "/q":
            print("Goodbye!")
            return .exit
        case "/clear", "/reset":
            conversation?.reset()
            history.removeAll()
            activeImageHistory.removeAll()
            recalledVaultPaths.removeAll()
            visionController?.unload()
            if let s = currentSystem, !s.isEmpty {
                history.append(.object([("role", .string("system")), ("content", .string(s))]))
            }
            print("\(cliColor.systemNotice)[Conversation history cleared.]\(cliColor.reset)\n")
            return .continueLoop
        case "/image":
            pendingTurnImage = arg.isEmpty ? nil : arg
            print("\(cliColor.systemNotice)[Next turn image: \(pendingTurnImage ?? "none")]\(cliColor.reset)\n")
            return .continueLoop
        case "/help", "/h", "/?":
            print("""
            \(cliColor.bold)Commands:\(cliColor.reset)
              /help, /h               Show this help message
              /clear, /reset          Clear conversation history
              /image [path]           Attach an image to the next turn
              /system [prompt]        View or update system prompt (empty to clear)
              /thinking [on|off]      Toggle or set thinking mode
              /effort [low|med|xhigh] Set reasoning effort
              /tokens [N]             Set max generated tokens per turn
              /stats                  Toggle performance statistics display
              /cache                  Show prefix cache memory and SSD usage
              /memory [mount|unmount] Inspect, mount, or unmount LSA codebase memory pack
              /history                Show conversation turn count
              /exit, /quit, /q        Exit REPL

            """)
            return .continueLoop
        case "/memory", "/repo":
            if arg.lowercased() == "unmount" {
                engine.unmountSharedBasePrefixes()
                activeMemoryPack = nil
                mountedMemoryTokens = []
                recalledVaultPaths.removeAll()
                print("\(cliColor.systemNotice)[Memory pack unmounted.]\(cliColor.reset)\n")
            } else if arg.lowercased().hasPrefix("mount ") {
                let targetPath = String(arg.dropFirst("mount ".count)).trimmingCharacters(in: .whitespaces)
                if let resolvedMem = QWResolvedCliMemoryPack.resolve(
                    explicitRepoPath: targetPath, disableMemoryPack: false,
                    modelName: runtimeAssets.descriptor.family.rawValue,
                    geometryDigest: runtimeAssets.descriptor.geometryDigestV3Hex,
                    preferredKVMode: String(describing: cliCfg.resolvedKVMode)
                ), resolvedMem.artifact.kvMode.lowercased() == String(describing: cliCfg.resolvedKVMode).lowercased(),
                   let mounted = try? engine.mountStandalonePrefixCache(from: resolvedMem.cacheFileURL) {
                    activeMemoryPack = resolvedMem
                    mountedMemoryTokens = mounted.tokens
                    recalledVaultPaths.removeAll()
                    print("\(cliColor.systemNotice)[Mounted '\(resolvedMem.manifest.packName)' (\(mounted.tokens.count) tokens, kv=\(resolvedMem.artifact.kvMode.uppercased())).]\(cliColor.reset)\n")
                } else {
                    print("\(cliColor.systemNotice)[Could not mount memory pack at '\(targetPath)' (check .qwg/manifest.json and matching KV mode '\(cliCfg.resolvedKVMode)').]\(cliColor.reset)\n")
                }
            } else if let mem = activeMemoryPack {
                let delta = QWMemoryPack.computeDelta(for: mem.manifest)
                let cortexCount = mem.manifest.files.filter(\.inCortex).count
                let vaultCount = mem.manifest.files.count - cortexCount
                print("\(cliColor.bold)Active LSA Memory Pack:\(cliColor.reset) \(mem.manifest.packName) (\(mem.manifest.rootPath))")
                print("  Files / Tokens : \(mem.manifest.files.count) files (\(cortexCount) cortex + \(vaultCount) vault), \(mountedMemoryTokens.count) cortex tokens (enforced KV: \(mem.artifact.kvMode.uppercased()))")
                print("  Vault Recalled : \(recalledVaultPaths.count) file(s) paged into session")
                print("  Live Delta     : \(delta.modifiedFiles.count) modified, \(delta.deletedFiles.count) deleted\n")
            } else {
                print("\(cliColor.dim)No LSA memory pack mounted. Use '/memory mount <path>' or run 'qwg index'.\(cliColor.reset)\n")
            }
            return .continueLoop
        case "/cache":
            let cacheState = engine.prefixCache
            if cacheState.isConfigured {
                print("\(cliColor.bold)Prefix Cache Status:\(cliColor.reset)")
                print("  Warm RAM : \(QWStatFormat.mebibytes(cacheState.warmBytes)) (\(cacheState.warmEntries.map(String.init) ?? "unavailable") entries)")
                print("  Cold SSD : \(QWStatFormat.mebibytes(cacheState.ssdBytes)) (\(cacheState.ssdEntries.map(String.init) ?? "unavailable") entries)")
            } else {
                print("\(cliColor.dim)Prefix cache is not enabled.\(cliColor.reset)")
            }
            print("")
            return .continueLoop
        case "/system":
            conversation?.reset()
            if arg.isEmpty {
                currentSystem = nil
                if let first = history.first, first.get("role").isString("system") { history.removeFirst() }
                print("\(cliColor.systemNotice)[System prompt cleared.]\(cliColor.reset)\n")
            } else {
                currentSystem = arg
                if let first = history.first, first.get("role").isString("system") {
                    history[0] = .object([("role", .string("system")), ("content", .string(arg))])
                } else {
                    history.insert(.object([("role", .string("system")), ("content", .string(arg))]), at: 0)
                }
                print("\(cliColor.systemNotice)[System prompt updated.]\(cliColor.reset)\n")
            }
            return .continueLoop
        case "/thinking":
            conversation?.reset()
            if arg.lowercased() == "on" || arg.lowercased() == "true" { isThinking = true }
            else if arg.lowercased() == "off" || arg.lowercased() == "false" { isThinking = false }
            else { isThinking.toggle() }
            displayThinking = isThinking && showThinking && !hideThinking
            print("\(cliColor.systemNotice)[Thinking \(isThinking ? "enabled" : "disabled").]\(cliColor.reset)\n")
            return .continueLoop
        case "/effort":
            conversation?.reset()
            let validEfforts = ["low", "medium", "xhigh"]
            if validEfforts.contains(arg.lowercased()) {
                currentReasoningEffort = arg.lowercased()
                print("\(cliColor.systemNotice)[Reasoning effort set to '\(currentReasoningEffort)'.]\(cliColor.reset)\n")
            } else {
                print("\(cliColor.systemNotice)[Usage: /effort low|medium|xhigh (current: \(currentReasoningEffort))]\(cliColor.reset)\n")
            }
            return .continueLoop
        case "/tokens":
            if let n = Int(arg), n > 0 {
                currentMaxTokens = n
                print("\(cliColor.systemNotice)[Max generated tokens set to \(n).]\(cliColor.reset)\n")
            } else {
                let cur = currentMaxTokens.map(String.init) ?? (isThinking ? "512 (default)" : "128 (default)")
                print("\(cliColor.systemNotice)[Usage: /tokens <positive integer> (current: \(cur))]\(cliColor.reset)\n")
            }
            return .continueLoop
        case "/stats":
            showStats.toggle()
            print("\(cliColor.systemNotice)[Performance statistics \(showStats ? "enabled" : "disabled").]\(cliColor.reset)\n")
            return .continueLoop
        case "/history":
            let turns = history.filter { $0.get("role").isString("user") }.count
            print("\(cliColor.systemNotice)[History contains \(turns) user turn(s), total \(history.count) message(s).]\(cliColor.reset)\n")
            return .continueLoop
        default:
            print("\(cliColor.systemNotice)Unknown command '\(cmd)'. Type '/help' for available commands.\(cliColor.reset)\n")
            return .continueLoop
        }
    }
}
