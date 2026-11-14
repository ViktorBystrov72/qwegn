// FNGPUBatchScheduler+Drive.swift — Multi-slot decode scheduler loop for FNGPUEngine.

import Foundation
import QWAlloc
import QWCore
import QWHAL
import QWHead

extension FNGPUBatchScheduler {
    func driveLoop() {
        while true {
            // Micro-coalesce concurrent requests currently tokenizing when starting a new wave.
            let activeCount0 = activeJobs.compactMap { $0 }.count
            if activeCount0 == 0 {
                var waitedUs = 0
                while waitedUs < 15_000 {
                    lock.lock()
                    let qCount = pendingQueue.count
                    lock.unlock()
                    let prep = FNBatchAdmissionTracker.preparingCount
                    if qCount >= Self.maxSlots { break }
                    if prep == 0 && waitedUs >= 1_500 { break }
                    usleep(250)
                    waitedUs += 250
                }
            }

            // Admit waiting jobs into free slots.
            let sCap4 = slotCapacity(totalSlots: Self.maxSlots)
            var newlyAdmitted: [Int] = []
            lock.lock()
            if !pendingQueue.isEmpty {
                let currentActive = activeJobs.compactMap { $0 }.count
                let totalCandidate = currentActive + pendingQueue.count
                let firstFitsIn4 = (pendingQueue[0].prompt.count + pendingQueue[0].maxNew + 4 <= sCap4)
                if currentActive == 0 && (totalCandidate == 1 || !firstFitsIn4) {
                    let job = pendingQueue.removeFirst()
                    job.inBatchMode = false
                    activeJobs[0] = job
                    newlyAdmitted.append(0)
                } else {
                    // Never transition an already-running single-stream Slot 0 job into partitioned batch mode
                    // mid-flight: let it finish its current single-stream turn cleanly first.
                    let slot0CanBatch: Bool
                    if let j0 = activeJobs[0] {
                        slot0CanBatch = j0.inBatchMode
                            && (j0.session.history.count + (j0.maxNew - j0.generatedTokens.count) + 4 <= sCap4)
                    } else {
                        slot0CanBatch = true
                    }
                    if slot0CanBatch {
                        for s in 0..<Self.maxSlots where activeJobs[s] == nil && !pendingQueue.isEmpty {
                            let cand = pendingQueue[0]
                            let fitsIn4 = (cand.prompt.count + cand.maxNew + 4 <= sCap4)
                            if fitsIn4 || (s == 0 && activeJobs.compactMap({ $0 }).isEmpty) {
                                let job = pendingQueue.removeFirst()
                                job.inBatchMode = fitsIn4 && (totalCandidate > 1 || s > 0)
                                activeJobs[s] = job
                                newlyAdmitted.append(s)
                                if !job.inBatchMode { break }
                            } else {
                                break
                            }
                        }
                    }
                }
            }
            lock.unlock()

            if !newlyAdmitted.isEmpty {
                let allActiveSlots = (0..<Self.maxSlots).filter { activeJobs[$0] != nil }
                let useMultiSlot = allActiveSlots.count > 1 || allActiveSlots.contains(where: { $0 > 0 }) || activeJobs[allActiveSlots[0]]!.inBatchMode
                if useMultiSlot {
                    do {
                        try ensureBatchResources()
                    } catch {
                        for s in newlyAdmitted {
                            if let j = activeJobs[s] { finishJob(slot: s, job: j, error: error) }
                        }
                        continue
                    }
                    for s in allActiveSlots { activeJobs[s]?.inBatchMode = true }
                }

                // Prefill each newly admitted slot.
                for s in newlyAdmitted {
                    guard let job = activeJobs[s] else { continue }
                    do {
                        if job.inBatchMode {
                            activateSlot(s, totalSlots: Self.maxSlots)
                            job.session.useSpec = (s == 0) ? job.useSpec : false
                            if !engine.hasPrefixCacheTier {
                                try engine.preparePrefill(forTokenCount: job.prompt.count)
                            }
                            engine.beginSession(&job.session, maxNew: job.maxNew)
                            let prefillStart = Date()
                            var chunkStart = 0
                            var firstTok = 0
                            let chunkRows = engine.dynamicPrefillChunkRows(forTotalTokens: job.prompt.count)
                            while chunkStart < job.prompt.count {
                                let end = min(chunkStart + chunkRows, job.prompt.count)
                                let isFirst = chunkStart == 0
                                firstTok = try engine.encodePrefillChunk(&job.session, chunk: chunkStart..<end,
                                                                         runsHead: end == job.prompt.count)
                                if isFirst {
                                    onCacheEvaluated?(job.session)
                                    onSlot0CacheEvaluated?(job.session)
                                }
                                chunkStart = end
                            }
                            job.prefillSeconds = Date().timeIntervalSince(prefillStart)
                            job.session.useSpec = false
                            syncSlotFromEngine(s)
                            job.position = job.prompt.count
                            if job.maxNew <= 0 {
                                engine.flushPendingPrefixCache(&job.session)
                                job.decodeStart = Date()
                                finishJob(slot: s, job: job)
                            } else {
                                _ = emitAndCheckDone(token: firstTok, slot: s, job: job)
                                engine.flushPendingPrefixCache(&job.session)
                                job.decodeStart = Date()
                            }
                        } else {
                            cleanupExtraSlotsBeforeSingleMode()
                            activateSlot(0, totalSlots: 1)
                            if !engine.hasPrefixCacheTier {
                                try engine.preparePrefill(forTokenCount: job.prompt.count)
                            }
                            engine.beginSession(&job.session, maxNew: job.maxNew)
                            let prefillStart = Date()
                            var chunkStart = 0
                            var firstTok = 0
                            let chunkRows = engine.dynamicPrefillChunkRows(forTotalTokens: job.prompt.count)
                            while chunkStart < job.prompt.count {
                                let end = min(chunkStart + chunkRows, job.prompt.count)
                                let isFirst = chunkStart == 0
                                firstTok = try engine.encodePrefillChunk(&job.session, chunk: chunkStart..<end,
                                                                         runsHead: end == job.prompt.count)
                                if isFirst {
                                    onCacheEvaluated?(job.session)
                                    onSlot0CacheEvaluated?(job.session)
                                }
                                chunkStart = end
                            }
                            job.prefillSeconds = Date().timeIntervalSince(prefillStart)
                            syncSlotFromEngine(0)
                            job.position = job.prompt.count
                            if job.maxNew <= 0 {
                                engine.flushPendingPrefixCache(&job.session)
                                job.decodeStart = Date()
                                finishJob(slot: 0, job: job)
                            } else {
                                _ = emitAndCheckDone(token: firstTok, slot: 0, job: job)
                                engine.flushPendingPrefixCache(&job.session)
                                job.decodeStart = Date()
                            }
                        }
                    } catch {
                        finishJob(slot: s, job: job, error: error)
                    }
                }
                // Re-check pendingQueue in case new requests arrived while prefill was executing on the GPU.
                lock.lock()
                let moreWaiting = !pendingQueue.isEmpty
                lock.unlock()
                if moreWaiting && activeJobs.contains(where: { $0 == nil }) {
                    continue
                }
            }

            let activeSlots = (0..<Self.maxSlots).filter { activeJobs[$0] != nil }
            if activeSlots.isEmpty {
                lock.lock()
                if pendingQueue.isEmpty {
                    cleanupExtraSlotsBeforeSingleMode()
                    activateSlot(0, totalSlots: 1)
                    isDriving = false
                    lock.unlock()
                    return
                }
                lock.unlock()
                continue
            }

            // Recover Slot 0 to full-capacity single-stream mode once other batch slots drain.
            if activeSlots.count == 1 && activeSlots[0] == 0, let job0 = activeJobs[0], job0.inBatchMode {
                lock.lock()
                let canRecoverSingle = pendingQueue.isEmpty && FNBatchAdmissionTracker.preparingCount == 0
                lock.unlock()
                if canRecoverSingle {
                    cleanupExtraSlotsBeforeSingleMode()
                    job0.inBatchMode = false
                    if job0.generatedTokens.count <= 1 {
                        job0.session.useSpec = job0.useSpec
                    }
                }
            }

            // Single-stream fast path on Slot 0 (full capacity + speculative decoding).
            if activeSlots.count == 1 && activeSlots[0] == 0 && !activeJobs[0]!.inBatchMode {
                let job = activeJobs[0]!
                activateSlot(0, totalSlots: 1)
                do {
                    let prevRounds = job.session.roundTokenCounts.count
                    let started = Date()
                    let nextTok = try engine.encodeDecodeToken(&job.session, token: job.lastToken, position: job.position)
                    job.position += 1
                    let measured = Date().timeIntervalSince(started) * 1000.0
                    let reported = job.session.stepMs > 0 ? job.session.stepMs : measured
                    let gpuMs = job.session.lastStepGPUMillis
                    if gpuMs > 0 && reported > 0 {
                        onStepTiming?(gpuMs, reported)
                    }
                    job.stepMs.append(reported)
                    if job.session.roundTokenCounts.count > prevRounds {
                        job.stepTokenCounts = job.session.roundTokenCounts
                    } else if job.session.roundTokenCounts.isEmpty {
                        job.stepTokenCounts.append(1)
                    } else {
                        job.stepTokenCounts = job.session.roundTokenCounts
                    }
                    syncSlotFromEngine(0)
                    _ = emitAndCheckDone(token: nextTok, slot: 0, job: job)
                } catch {
                    finishJob(slot: 0, job: job, error: error)
                }
                continue
            }

            // Multi-session batched decode path across all `B` active slots (`1 <= B <= 4`).
            var validSlots: [Int] = []
            var histories: [[Int64]] = []
            for s in activeSlots {
                guard let job = activeJobs[s] else { continue }
                if job.session.history.count + 1 >= sCap4 {
                    finishJob(slot: s, job: job,
                              error: FNGPUError.contextDoesNotFit(requested: job.session.history.count + 1,
                                                                  maxFitting: sCap4 - 1))
                    continue
                }
                let prevCount = job.session.history.count
                let tok = job.lastToken
                job.session.history.append(Int64(tok))
                if engine.residentKVHistoryBySlot[s]?.count == prevCount {
                    engine.residentKVHistoryBySlot[s]?.append(Int64(tok))
                }
                if job.session.sampler.config.repetitionPenalty != 1.0 {
                    job.session.recordSeenToken(tok, vocab: engine.vocab)
                }
                slotStates[s].highWaterPos = max(slotStates[s].highWaterPos, job.session.history.count)
                validSlots.append(s)
                histories.append(job.session.history)
            }
            guard !validSlots.isEmpty, let bStack = batchStack else { continue }

            do {
                let started = Date()
                let rawArgmaxes = try bStack.forwardBatch(
                    histories: histories,
                    activeSlots: validSlots,
                    totalSlots: Self.maxSlots,
                    slotStates: &slotStates,
                    embedTable: engine.embedTable,
                    head: engine.head
                )
                let wallMs = Date().timeIntervalSince(started) * 1000.0
                let gpuMs = bStack.lastGPUMillis
                if gpuMs > 0 && wallMs > 0 {
                    onStepTiming?(gpuMs, wallMs)
                }
                for (b, s) in validSlots.enumerated() {
                    guard let job = activeJobs[s] else { continue }
                    let nextTok = try engine.sampleFromHead(
                        argmaxToken: rawArgmaxes[b],
                        seenTokens: job.session.seenTokens,
                        seenTokenList: job.session.seenTokenList,
                        sampler: job.session.sampler,
                        row: b,
                        constraint: job.session.constraint
                    )
                    if job.session.collectLogprobs {
                        engine.recordTokenLogprob(&job.session, token: nextTok, row: b)
                    }
                    job.position += 1
                    job.stepMs.append(wallMs)
                    job.stepTokenCounts.append(1)
                    _ = emitAndCheckDone(token: nextTok, slot: s, job: job)
                }
            } catch {
                for s in validSlots {
                    if let job = activeJobs[s] { finishJob(slot: s, job: job, error: error) }
                }
            }
        }
    }
}
