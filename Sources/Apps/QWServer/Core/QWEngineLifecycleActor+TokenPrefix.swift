// QWEngineLifecycleActor+TokenPrefix.swift — Logprobs, stop suffixes, and incremental token prefix encoding.

import Foundation
import QWCli
import QWCore
import QWEngine
import QWHAL
import QWHead
import QWKV
import QWMM
import QWQuant
import QWToken
import QWVision

extension QWEngineLifecycleActor {
    nonisolated static func makeLogprobEntry(_ lp: QWTokenLogprob, tokenizer: QWTokenizer) -> ChatLogprobTokenEntry {
        let tokStr = (try? tokenizer.decode([lp.tokenId], skipSpecialTokens: false)) ?? ""
        let tokBytes = Array(tokStr.utf8).map { Int($0) }
        let tops: [ChatLogprobTokenEntry.TopLogprob] = lp.topCandidates.map { cand in
            let cStr = (try? tokenizer.decode([cand.tokenId], skipSpecialTokens: false)) ?? ""
            let cBytes = Array(cStr.utf8).map { Int($0) }
            return ChatLogprobTokenEntry.TopLogprob(token: cStr, logprob: cand.logprob, bytes: cBytes)
        }
        return ChatLogprobTokenEntry(token: tokStr, logprob: lp.logprob, bytes: tokBytes, topLogprobs: tops)
    }

    nonisolated static func heldStopSuffixUTF8(_ text: String, stops: [String]) -> Int {
        var best = 0
        for s in stops {
            let maxChars = min(s.count - 1, text.count)
            guard maxChars > 0 else { continue }
            for len in stride(from: maxChars, through: 1, by: -1) {
                let prefix = String(s.prefix(len))
                if text.hasSuffix(prefix) {
                    best = max(best, prefix.utf8.count)
                    break
                }
            }
        }
        return best
    }

    nonisolated static func firstCompleteJsonObjectEnd(in text: String) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false
        var started = false
        for idx in text.indices {
            let ch = text[idx]
            if inString {
                if escaped {
                    escaped = false
                } else if ch == "\\" {
                    escaped = true
                } else if ch == "\"" {
                    inString = false
                }
                continue
            }
            if ch == "\"" {
                inString = true
            } else if ch == "{" {
                depth += 1
                started = true
            } else if ch == "}" {
                depth -= 1
                if started && depth == 0 {
                    return text.index(after: idx)
                }
            }
        }
        return nil
    }

    /// Incrementally encodes a rendered chat wire string by caching tokenized prefixes up to
    /// `<|im_start|>` turn boundaries across concurrent sessions.
    func encodeWireIncrementallyWithSystemPrefix(
        wire: String,
        tokenizer tok: QWTokenizer,
        packPath: String
    ) throws -> (tokens: [Int], systemPrefixTokenCount: Int?) {
        let marker = "<|im_start|>"
        guard tok.tokenId(for: marker) != nil,
              let lastRange = wire.range(of: marker, options: .backwards),
              lastRange.lowerBound > wire.startIndex else {
            return (try tok.encode(wire), nil)
        }
        let maxEntries = max(12, config.maxConcurrentSessions * 2 + 4)
        func storePrefixEntry(_ entry: CachedTokenPrefixEntry) {
            cachedTokenPrefixes.removeAll { $0.packPath == entry.packPath && $0.prefixWire == entry.prefixWire }
            cachedTokenPrefixes.append(entry)
            if cachedTokenPrefixes.count > maxEntries {
                cachedTokenPrefixes.removeFirst(cachedTokenPrefixes.count - maxEntries)
            }
        }

        var systemWire: String? = nil
        if wire.hasPrefix("<|im_start|>system") || wire.hasPrefix("<|im_start|>developer") {
            let afterFirst = wire.index(wire.startIndex, offsetBy: marker.count)
            if let secondRange = wire.range(of: marker, range: afterFirst..<wire.endIndex),
               secondRange.lowerBound <= lastRange.lowerBound {
                systemWire = String(wire[..<secondRange.lowerBound])
            }
        }

        let newPrefixWire = String(wire[..<lastRange.lowerBound])
        let tailWire = String(wire[lastRange.lowerBound...])
        let newUTF8Count = newPrefixWire.utf8.count
        let markerUTF8 = Array(marker.utf8)

        var bestIdx: Int? = nil
        var bestLen = 0
        for (idx, entry) in cachedTokenPrefixes.enumerated() where entry.packPath == packPath && !entry.prefixWire.isEmpty {
            let count = entry.prefixWire.utf8.count
            guard count > bestLen, count <= newUTF8Count,
                  newPrefixWire.utf8.starts(with: entry.prefixWire.utf8) else { continue }
            if count == newUTF8Count {
                bestIdx = idx; bestLen = count
            } else {
                let deltaStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: count)
                if newPrefixWire.utf8[deltaStart...].starts(with: markerUTF8) {
                    bestIdx = idx; bestLen = count
                }
            }
        }

        if let idx = bestIdx {
            let matched = cachedTokenPrefixes.remove(at: idx)
            cachedTokenPrefixes.append(matched)
            let deltaStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: matched.prefixWire.utf8.count)
            let deltaWire = String(newPrefixWire[deltaStart...])
            if deltaWire.isEmpty {
                let tailTokens = try tok.encode(tailWire)
                return (matched.prefixTokens + tailTokens, matched.systemTokenCount)
            } else if deltaWire.utf8.starts(with: markerUTF8) {
                let deltaTokens = try tok.encode(deltaWire)
                let updatedPrefixTokens = matched.prefixTokens + deltaTokens
                let sysCount = matched.systemTokenCount
                    ?? (systemWire != nil && matched.prefixWire == systemWire ? matched.prefixTokens.count : nil)
                storePrefixEntry(CachedTokenPrefixEntry(
                    packPath: packPath, prefixWire: newPrefixWire,
                    prefixTokens: updatedPrefixTokens, systemTokenCount: sysCount))
                let tailTokens = try tok.encode(tailWire)
                return (updatedPrefixTokens + tailTokens, sysCount)
            }
        }

        var resolvedSystemTokenCount: Int? = nil
        let prefixTokens: [Int]
        if let sysW = systemWire, sysW.utf8.count < newUTF8Count, newPrefixWire.utf8.starts(with: sysW.utf8) {
            let sysTokens = try tok.encode(sysW)
            resolvedSystemTokenCount = sysTokens.count
            storePrefixEntry(CachedTokenPrefixEntry(
                packPath: packPath, prefixWire: sysW, prefixTokens: sysTokens, systemTokenCount: sysTokens.count))
            let remStart = newPrefixWire.utf8.index(newPrefixWire.utf8.startIndex, offsetBy: sysW.utf8.count)
            let remTokens = try tok.encode(String(newPrefixWire[remStart...]))
            prefixTokens = sysTokens + remTokens
        } else {
            prefixTokens = try tok.encode(newPrefixWire)
            if let sysW = systemWire, sysW.utf8.count == newUTF8Count {
                resolvedSystemTokenCount = prefixTokens.count
            }
        }
        storePrefixEntry(CachedTokenPrefixEntry(
            packPath: packPath, prefixWire: newPrefixWire,
            prefixTokens: prefixTokens, systemTokenCount: resolvedSystemTokenCount))
        let tailTokens = try tok.encode(tailWire)
        return (prefixTokens + tailTokens, resolvedSystemTokenCount)
    }

    nonisolated public func validateRequestCapabilities(_ request: ChatCompletionRequest) throws {
        guard !request.extractImageUrls().isEmpty else { return }
        let cfg = statusBox.getConfig()
        let eager = cfg.loadVision || ProcessInfo.processInfo.environment["QW_VISION_EAGER"] == "1"
        guard eager else {
            throw QWMultimodalError.badArgument(
                "Vision tower is disabled (--no-load-vision). Enable --load-vision to process image inputs."
            )
        }
        let state = statusBox.engineState()
        if state.isLoaded {
            let visDepth = statusBox.getLoadedParamsInfo()?.modelDescriptor?.visDepth ?? 0
            if !state.capabilities.contains(.vision) || visDepth <= 0
                || !QWVisionOnDemandBackend.hasVisionWeights(modelPath: cfg.packPath) {
                throw QWMultimodalError.badArgument("Loaded model does not support vision inputs.")
            }
        } else if !QWVisionOnDemandBackend.hasVisionWeights(modelPath: cfg.packPath) {
            throw QWMultimodalError.badArgument("Model pack does not contain vision weights.")
        }
    }

    nonisolated static func prepareTokenHeal(
        prepared: QWPreparedChatPrompt,
        request: ChatCompletionRequest,
        promptTokens: [Int],
        systemPrefixTokenCount: Int?,
        stops: Set<Int>,
        tokenizer: QWTokenizer,
        vocabulary: Int,
        supportsTokenConstraints: Bool,
        hasImages: Bool
    ) -> (promptTokens: [Int], constraint: (any QWTokenConstraint)?, stripUTF8Count: Int, didPeel: Bool) {
        guard prepared.isAssistantPrefill,
              prepared.activeConstraint == nil,
              supportsTokenConstraints,
              !hasImages,
              promptTokens.count >= 2 else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        let prefillText = request.messages.last?.extractTextContent() ?? ""
        let prefillUTF8 = Array(prefillText.utf8)
        let lastTok = promptTokens[promptTokens.count - 1]
        guard !prefillUTF8.isEmpty,
              !stops.contains(lastTok),
              (promptTokens.count - 1) >= (systemPrefixTokenCount ?? 0) else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        let table = QWStructuredOutput.tokenTable(for: tokenizer, vocabulary: vocabulary)
        guard lastTok >= 0 && lastTok < table.vocabulary,
              let tokBytes = table.bytes[lastTok], !tokBytes.isEmpty,
              tokBytes.count <= prefillUTF8.count,
              prefillUTF8.suffix(tokBytes.count).elementsEqual(tokBytes) else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        var probe = QWStreamDecoder(tokenizer)
        guard let healedStr = try? probe.decode(lastTok, skipSpecialTokens: true),
              !healedStr.isEmpty, probe.flush().isEmpty else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        let constraint = QWPrefixHealingConstraint(prefixBytes: tokBytes, table: table)
        guard constraint.hasSupport else {
            return (promptTokens, prepared.activeConstraint, 0, false)
        }
        return (Array(promptTokens.dropLast()), constraint, healedStr.utf8.count, true)
    }
}
