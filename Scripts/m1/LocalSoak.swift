// LocalSoak.swift — The CPU-only reference soak.
//
// The m1 gate has to run on a machine with no model and no arena. This is
// what it measures there: a real, deterministic, allocation-free CPU
// workload, timed step by step, with the process's own RSS, fd count and
// allocation delta sampled per window. It is NOT a decode soak — the
// artifact says so, and the engine's decode laws are skipped on it.

import Darwin
import Foundation
import QWCli

/// What the process itself can be asked: its resident size, its open
/// files, and how many times it has allocated.
enum ProcessTelemetry {
    /// A monotonic nanosecond clock.
    static func clockGetNanos() -> UInt64 {
        clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    }

    /// The resident size of this process, in KiB.
    static func rssKb() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int64(info.resident_size) / 1024 : -1
    }

    /// The number of file descriptors this process holds open.
    static func fdCount() -> Int {
        let pid = Int32(ProcessInfo.processInfo.processIdentifier)
        let n = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, nil, 0)
        return Int(n) / MemoryLayout<proc_fdinfo>.stride
    }

    /// The number of live heap blocks this process holds. The delta
    /// across a window is what the steady state retained: zero when
    /// the window allocated and freed, the leak when it held on.
    static func liveHeapBlocks() -> Int {
        var stats = malloc_statistics_t()
        malloc_zone_statistics(malloc_default_zone(), &stats)
        return Int(stats.blocks_in_use)
    }
}

/// The CPU-only reference soak: no model, no GPU, no arena.
enum LocalSoak {
    /// What to run.
    struct Options {
        /// The steps the soak performs in total.
        var steps: Int
        /// The steps in one window.
        var window: Int
        /// The side of the square matrix one step multiplies.
        var rows: Int
        /// What this run is, in words.
        var label: String

        init(steps: Int = 10_000, window: Int = 1_000, rows: Int = 2_048,
             label: String = "CPU-only reference step — no model, no GPU "
                 + "(steadyAllocs = live heap-block delta)") {
            self.steps = steps
            self.window = window
            self.rows = rows
            self.label = label
        }
    }

    /// Runs the soak and returns one measured window per window of steps.
    static func run(_ options: Options) -> [QWCli.SoakSample] {
        let rows = max(16, options.rows)
        let window = max(1, options.window)
        let total = max(window, options.steps)
        let weights = makeWeights(rows: rows)
        var input = [Float](repeating: 0.5, count: rows)
        var output = [Float](repeating: 0, count: rows)
        var checksum: Float = 0
        for _ in 0..<min(2_000, total) {
            checksum += step(weights: weights, input: &input, output: &output)
        }
        var latencies = [Double](repeating: 0, count: window)
        for i in 0..<min(window, total) {
            let t0 = ProcessTelemetry.clockGetNanos()
            checksum += step(weights: weights, input: &input, output: &output)
            latencies[i] = Double(ProcessTelemetry.clockGetNanos() - t0) / 1_000_000
        }
        var samples: [QWCli.SoakSample] = []
        var done = 0
        while done < total {
            let n = min(window, total - done)
            let allocsBefore = ProcessTelemetry.liveHeapBlocks()
            let wallStart = ProcessTelemetry.clockGetNanos()
            for i in 0..<n {
                let t0 = ProcessTelemetry.clockGetNanos()
                checksum += step(weights: weights, input: &input, output: &output)
                latencies[i] = Double(ProcessTelemetry.clockGetNanos() - t0) / 1_000_000
            }
            let wallEnd = ProcessTelemetry.clockGetNanos()
            let allocsAfter = ProcessTelemetry.liveHeapBlocks()
            let wallSeconds = Double(wallEnd - wallStart) / 1_000_000_000
            let ordered = latencies[0..<n].sorted()
            done += n
            samples.append(QWCli.SoakSample(
                steps: done,
                windowTps: wallSeconds > 0 ? Double(n) / wallSeconds : 0,
                windowP50Ms: percentile(ordered, 0.50),
                windowP95Ms: percentile(ordered, 0.95),
                steadyAllocs: allocsAfter - allocsBefore,
                rssKb: ProcessTelemetry.rssKb(),
                fds: ProcessTelemetry.fdCount(),
                cbErrors: nil,
                kvFreePages: nil,
                source: QWCli.SoakSample.harnessSource,
                sampledAt: Date(),
                label: options.label,
                kind: QWCli.SoakSample.windowKind))
        }
        if checksum == 0 { print("[m1] local soak: checksum \(checksum)") }
        return samples
    }

    @inline(never)
    private static func step(weights: [Float], input: inout [Float], output: inout [Float]) -> Float {
        let rows = input.count
        var checksum: Float = 0
        weights.withUnsafeBufferPointer { w in
            input.withUnsafeMutableBufferPointer { x in
                for i in 0..<rows {
                    var acc: Float = 0
                    let row = w.baseAddress! + i * rows
                    for j in 0..<rows { acc += row[j] * x[j] }
                    output[i] = acc
                    checksum += acc
                }
                x[0] = x[0] + 0.001
            }
        }
        return checksum
    }

    private static func makeWeights(rows: Int) -> [Float] {
        var weights = [Float](repeating: 0, count: rows * rows)
        var state: UInt32 = 0x1234_5678
        for i in 0..<(rows * rows) {
            state = state &* 1_664_525 &+ 1_013_904_223
            weights[i] = Float(state & 0xFFFF) / 65535.0 - 0.5
        }
        return weights
    }

    private static func percentile(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let idx = min(sorted.count - 1, max(0, Int((q * Double(sorted.count)).rounded(.up)) - 1))
        return sorted[idx]
    }
}
