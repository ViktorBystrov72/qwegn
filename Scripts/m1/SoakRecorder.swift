// SoakRecorder.swift — The one writer of the measured windows.
//
// Every window that reaches the gate goes through here. The schema is
// owned by QWCli.SoakSample; the recorder only owns the file.

import Foundation
import QWCli

/// Writes measured windows into a JSONL file, one window per line.
struct SoakRecorder {
    /// Where the windows go.
    let url: URL
    /// How many windows were written.
    private(set) var written: Int = 0

    init(url: URL) {
        self.url = url
    }

    /// The export path: `m1_export_<stamp>.jsonl` in a directory.
    static func exportURL(in directory: URL, stamp: Date = Date()) -> URL {
        directory.appendingPathComponent("m1_export_\(stampStamp(stamp)).jsonl")
    }

    /// The stamp every soak file carries: seconds since the epoch.
    private static func stampStamp(_ date: Date) -> Int {
        Int(date.timeIntervalSince1970)
    }

    /// Writes one window.
    mutating func record(_ window: QWCli.SoakSample) throws {
        try window.append(to: url)
        written += 1
    }

    /// Writes a run of windows.
    mutating func record(_ windows: [QWCli.SoakSample]) throws {
        for window in windows { try record(window) }
    }
}
