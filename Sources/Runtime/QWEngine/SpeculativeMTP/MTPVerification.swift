// MTPVerification.swift — QWEngineDriver MTPVerification family (split from QWEngineDriver.swift; same-module extension).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    func verifyTokensLocked(_ session: QWEngineSession, tokens: [Int],
                                   completed: ((any MTLCommandBuffer) throws -> Void)? = nil) throws -> QWVerifyWindow {
        guard verifiedSession == nil, let verifyWindow, let gdnReplayCheckpoint,
              (2...verifyWindow.rows).contains(tokens.count), !session.sessionFinished, !session.stateFailed,
              !session.verificationPending, session.trunkHiddenValid,
              session.position > 0, session.position <= ropeRows - tokens.count else {
            throw QWEngineDriverError.state("verify requires a seeded live session, B=2/3, and available positions")
        }
        try kvCache.validate(session.kv)
        guard kvCache.written(session.kv) == session.position else {
            throw QWEngineDriverError.state("verify trunk cursor mismatch")
        }
        for token in tokens where token < 0 || token >= geometry.vocab { throw QWEngineDriverError.badToken(token) }
        try claimMLAIndexerState(session)
        let end = session.position + tokens.count
        if end > session.kv.tokenCapacity {
            try kvCache.grow(session.kv, to: end)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        try kvCache.prepareWrites(session.kv, positions: session.position..<end)
        for (row, token) in tokens.enumerated() {
            try copyEmbedRow(token, to: verifyWindow.hiddenRow(row))
        }
        guard let cb = hal.queue.makeCommandBuffer() else { throw QWEngineDriverError.state("verify command buffer") }
        let stageProfiler = mtpVerificationProfiling
            ? try QWGPUStageProfiler<QWMTPVerificationStage>(device: hal.device,
                                                            segmentCapacity: geometry.numVirtualLayers * 7 + 3)
            : nil
        let tVerify0 = Self.decodeTraceEnabled ? Date() : nil
        self.activeCB = cb
        self.stepCBs = stageProfiler == nil ? [] : nil
        decodeTel.coldStageUs = 0
        decodeTel.coldWaitUs = 0
        decodeTel.coldIOUs = 0
        decodeTel.coldStaged = 0
        defer {
            self.activeCB = nil
            self.activeEnc = nil
            self.stepCBs = nil
        }
        verificationGpuMsByStage = nil
        try verifyWindow.encode(cb, hal: hal, head: head,
                                checkpoint: gdnReplayCheckpoint,
                                rec: session.recSlab, ring: session.ringSlab,
                                headStorage: headStorage,
                                q4Head: mtpVerifyHeadQ4 ? mtpDraftHead : nil,
                                activeRows: tokens.count,
                                unrotateForF16Head: true,
                                multimaxParams: multimaxParams,
                                windowEncoder: {
                                    try self.encodeLayerMajorVerification(cb, session: session, window: verifyWindow,
                                                                          checkpoint: gdnReplayCheckpoint,
                                                                          activeRows: tokens.count,
                                                                          stageProfiler: stageProfiler)
                                },
                                currentCommandBuffer: { self.activeCB ?? cb },
                                stageProfiler: stageProfiler) { _ in }
        let finalCB = self.activeCB ?? cb
        finalCB.commit()
        finalCB.waitUntilCompleted()
        if let pipe = coldEvents {
            let r = pipe.takeStepResult()
            pipe.tuneBridge()
            decodeTel.coldWaitUs += r.waitUs
            decodeTel.coldIOUs += r.ioUs
            decodeTel.coldStaged += r.staged
            if let f = r.failure { throw QWEngineDriverError.state("cold event staging: \(f)") }
        }
        flushPendingColdUnloads()
        if let tVerify0 {
            let allCBs = (self.stepCBs ?? []) + [finalCB]
            let gpuUs = Self.gpuBusyMicros(allCBs)
            let wallUs = Date().timeIntervalSince(tVerify0) * 1e6
            FileHandle.standardError.write(Data(String(
                format: "[verify] pos=%d rows=%d wall=%.2fms gpu_busy=%.2fms cbs=%d cold_stage=%.2fms wait=%.2fms io=%.2fms staged=%d\n",
                session.position, tokens.count, wallUs / 1000, gpuUs / 1000, allCBs.count,
                decodeTel.coldStageUs / 1000, decodeTel.coldWaitUs / 1000, decodeTel.coldIOUs / 1000, decodeTel.coldStaged).utf8))
        }
        guard finalCB.status == .completed else {
            commandBufferErrors += 1
            session.stateFailed = true
            throw QWEngineDriverError.state("verify GPU failure: \(String(describing: finalCB.error))")
        }
        do {
            if let splitCBs = self.stepCBs, !splitCBs.isEmpty {
                for splitCB in splitCBs {
                    try completed?(splitCB)
                }
            }
            try completed?(finalCB)
            if let stageProfiler {
                let result = try stageProfiler.resolve(
                    totalGpuMicros: QWMTPStageTiming.gpuDurationMs(finalCB) * 1000)
                verificationGpuMsByStage = Dictionary(uniqueKeysWithValues:
                    result.gpuMicrosByStage.map { ($0.key.rawValue, $0.value / 1000) })
            }
        }
        catch {
            session.stateFailed = true
            throw error
        }
        do { _ = try verifyWindow.completedTargets(activeRows: tokens.count) }
        catch {
            session.stateFailed = true
            throw error
        }
        kvCache.markWritten(session.kv, upTo: end)
        session.verificationPending = true
        verifiedSession = session
        verifiedInputs = tokens
        return verifyWindow
    }

    /// Reorders independent projection rows, not each row's arithmetic.
    /// GDN executes rows in causal order. Shared attention keeps per-query
    /// causal limits after materializing the complete verification window.
    func encodeLayerMajorVerification(_ cb: any MTLCommandBuffer, session: QWEngineSession,
                                              window: QWVerifyWindow,
                                              checkpoint: QWGDNReplayCheckpoint,
                                              activeRows: Int,
                                              stageProfiler: QWGPUStageProfiler<QWMTPVerificationStage>? = nil) throws {
        var currentCB = cb
        func makeEncoder(_ stage: QWMTPVerificationStage) throws -> any MTLComputeCommandEncoder {
            let activeBuf = self.activeCB ?? currentCB
            currentCB = activeBuf
            if let stageProfiler {
                return try stageProfiler.makeEncoder(commandBuffer: activeBuf, stage: stage)
            }
            guard let encoder = activeBuf.makeComputeCommandEncoder() else {
                throw QWEngineDriverError.state("layer-major verification encoder")
            }
            return encoder
        }
        var enc = try makeEncoder(.norm)
        var encoderOpen = true
        self.activeCB = currentCB
        self.activeEnc = enc
        defer {
            if encoderOpen { enc.endEncoding() }
            self.activeEnc = nil
        }
        func beginStage(_ stage: QWMTPVerificationStage) throws {
            guard stageProfiler != nil else { return }
            enc.endEncoding()
            encoderOpen = false
            enc = try makeEncoder(stage)
            encoderOpen = true
            self.activeEnc = enc
        }
        guard (2...window.rows).contains(activeRows) else {
            throw QWEngineDriverError.state("layer-major active row count is invalid")
        }
        let h = geometry.hidden
        guard let normalized = window.normalized,
              let mlpRows = window.mlpRows,
              let mtpBatchedTrunk,
              let mtpBatchedMLP else {
            throw QWEngineDriverError.state("batched trunk verification buffers are absent")
        }
        let partials = mtpSharedAttentionPartials
        let plan = geometry.executionPlan
        try encodeRMSNormFast(enc, x: window.hidden, gamma: layersW[0].ln1, out: normalized, rows: activeRows)
        for (layer, p) in plan.enumerated() {
            let weights = layersW[p]
            let nextP = (layer + 1 < plan.count) ? plan[layer + 1] : nil
            let gamma = nextP.map { layersW[$0].ln1 } ?? finalNorm
            if let mhc = weights.mhc, let mhcCfg = geometry.mhcConfig,
               let mhcStream = window.mhcStream, let mhcControls = window.mhcControls {
                let prevP = layer > 0 ? plan[layer - 1] : -1
                let hasDeferredPrevFFN = (p != mhcCfg.start) && (prevP >= mhcCfg.start && prevP < mhcCfg.end - 1)
                if p == mhcCfg.start {
                    try encodeMHCEnter(enc, x: window.hidden, stream: mhcStream, rows: activeRows)
                    enc.memoryBarrier(scope: .buffers)
                }
                let prevBranch = hasDeferredPrevFFN
                    ? prefixRows(mtpBatchedMLP.downRows, rows: activeRows, cols: h, dt: .f16)
                    : nil
                try encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.attnW, b: mhc.attnB,
                                        gamma: weights.ln1, controls: mhcControls, out: normalized,
                                        rows: activeRows, mergeBranchOut: prevBranch)
                enc.memoryBarrier(scope: .buffers)
            }
            try beginStage(.inputProjection)
            let activeNormalized = prefixRows(normalized, rows: activeRows, cols: h, dt: .f16)
            let isGDN = (weights.kind == .gdn || weights.kind == .gdnMoe)
            try encodeVerificationLayerInputProjection(
                enc, weights: weights, activeNormalized: activeNormalized,
                mtpBatchedTrunk: mtpBatchedTrunk, mtpBatchedMLP: mtpBatchedMLP,
                activeRows: activeRows)
            try beginStage(isGDN ? .gdn : .attention)
            try encodeVerificationLayerCore(
                enc, session: session, layer: layer, weights: weights,
                activeNormalized: activeNormalized,
                checkpoint: checkpoint, mtpBatchedTrunk: mtpBatchedTrunk,
                mtpBatchedMLP: mtpBatchedMLP, partials: partials,
                activeRows: activeRows)
            try beginStage(.outputProjection)
            let weight: QWWeightView
            let outCols: Int
            switch weights.kind {
            case .gdn, .gdnMoe:
                weight = weights.gdn!.out
                outCols = geometry.gdnZChannels
            case .hca, .hcaMoe:
                guard let hca = hcaOperator, let woB = weights.hca?.woB else {
                    throw QWEngineDriverError.state("HCA verification layer missing operator or woB")
                }
                weight = woB
                outCols = hca.zOutStride
            case .fullAttention, .moe:
                weight = weights.attn!.o
                outCols = weights.attn!.o.k
            }
            try mtpBatchedMLP.encodeProjection(enc,
                input: prefixRows(mtpBatchedTrunk.outputInputs, rows: activeRows,
                                  cols: outCols, dt: .f16),
                weight: weight, output: mtpBatchedMLP.downRows,
                activeRows: activeRows, preparedSums: isGDN)
            if let mhc = weights.mhc, geometry.mhcConfig != nil,
               let mhcStream = window.mhcStream, let mhcControls = window.mhcControls {
                enc.memoryBarrier(scope: .buffers)
                try encodeMHCRouterNorm(enc, stream: mhcStream, w: mhc.ffnW, b: mhc.ffnB,
                                        gamma: weights.ln2, controls: mhcControls, out: normalized,
                                        rows: activeRows,
                                        mergeBranchOut: prefixRows(mtpBatchedMLP.downRows, rows: activeRows, cols: h, dt: .f16))
                enc.memoryBarrier(scope: .buffers)
            } else {
                try encodeRMSNormResidualFast(enc, x: mtpBatchedMLP.downRows, gamma: weights.ln2,
                    out: normalized, residual: window.hidden, rows: activeRows)
                enc.memoryBarrier(scope: .buffers)
            }
            if let moe = weights.moe {
                guard let moeScratch = window.moeScratch, let moeDn32 = window.moeDn32 else {
                    throw QWEngineDriverError.state("MTP verification MoE scratch is absent")
                }
                try beginStage(.mlpGateUp)
                try beginStage(.mlpDown)
                self.activeCB = currentCB
                self.activeEnc = enc
                try QWMoEChain.encode(
                    enc: &enc, hal: hal, moe: moe, s: moeScratch,
                    x16: prefixRows(normalized, rows: activeRows, cols: h, dt: .f16),
                    dn32: prefixRows(moeDn32, rows: activeRows, cols: h, dt: .f32),
                    dn16: prefixRows(mtpBatchedMLP.downRows, rows: activeRows, cols: h, dt: .f16),
                    rows: activeRows, geometry: geometry, driver: self)
                if let active = self.activeEnc { enc = active }
                if let activeBuf = self.activeCB { currentCB = activeBuf }
                enc.memoryBarrier(scope: .buffers)
            } else {
                try beginStage(.mlpGateUp)
                let activeMlpRows = prefixRows(mlpRows, rows: activeRows,
                                               cols: geometry.intermediate, dt: .f16)
                let fuseFwht = !weights.gu.view.isIncumbentContainer && geometry.intermediate > 0 && (geometry.intermediate % 32 == 0)
                try mtpBatchedMLP.encodeGateUp(enc,
                    input: prefixRows(normalized, rows: activeRows, cols: h, dt: .f16),
                    weight: weights.gu.view,
                    output: activeMlpRows,
                    activeRows: activeRows,
                    fuseFwht32: fuseFwht)
                if !fuseFwht {
                    encodeFwht32Fast(enc, x: activeMlpRows)
                    enc.memoryBarrier(scope: .buffers)
                }
                try beginStage(.mlpDown)
                try mtpBatchedMLP.encodeDown(enc,
                    input: activeMlpRows,
                    weight: weights.down, activeRows: activeRows)
            }
            try beginStage(.norm)
            if let _ = weights.mhc, let mhcCfg = geometry.mhcConfig,
               let mhcStream = window.mhcStream, let mhcControls = window.mhcControls {
                let isLeave = (p == mhcCfg.end - 1)
                let nextIsMHC = nextP.map { $0 >= mhcCfg.start && $0 < mhcCfg.end } ?? false
                let deferPostFFN = !isLeave && nextIsMHC && (nextP != mhcCfg.start)
                if !deferPostFFN {
                    enc.memoryBarrier(scope: .buffers)
                    if isLeave, mhcCfg.hasExitHead, let exitH = mhcExitHead {
                        try encodeMHCMerge(enc, stream: mhcStream, branchOut: mtpBatchedMLP.downRows,
                                           controls: mhcControls, xOut: window.hidden,
                                           nextGamma: gamma, nextNormOut: normalized,
                                           rows: activeRows, isLeave: false, sinkhornIters: mhcCfg.iterations,
                                           hasNextNorm: false)
                        enc.memoryBarrier(scope: .buffers)
                        try encodeMHCExitHead(enc, stream: mhcStream, headW: exitH.headW, headB: exitH.headB,
                                              normGamma: gamma, outNorm: normalized,
                                              rows: activeRows, applyNorm: true)
                    } else {
                        try encodeMHCMerge(enc, stream: mhcStream, branchOut: mtpBatchedMLP.downRows,
                                           controls: mhcControls, xOut: window.hidden,
                                           nextGamma: gamma, nextNormOut: normalized,
                                           rows: activeRows, isLeave: isLeave, sinkhornIters: mhcCfg.iterations,
                                           hasNextNorm: isLeave)
                    }
                    enc.memoryBarrier(scope: .buffers)
                }
            } else {
                try encodeRMSNormResidualFast(enc, x: mtpBatchedMLP.downRows, gamma: gamma,
                    out: normalized, residual: window.hidden, rows: activeRows)
                enc.memoryBarrier(scope: .buffers)
            }
            if let dflashDrafter, p < dflashDrafter.targetLayerSlotMap.count {
                let slot = dflashDrafter.targetLayerSlotMap[p]
                if slot >= 0 {
                    dflashDrafter.encodeCaptureTargetHidden(
                        enc, source: window.hidden, srcRowOffset: 0, dstRowOffset: 0, rows: activeRows, slot: slot)
                    enc.memoryBarrier(scope: .buffers)
                }
            }
        }
        try QWVerifyWindow.encodeCopy(enc, hal: hal,
            source: prefixRows(normalized, rows: activeRows, cols: h, dt: .f16),
            destination: prefixRows(window.hidden, rows: activeRows, cols: h, dt: .f16))
        enc.memoryBarrier(scope: .buffers)
    }

}
