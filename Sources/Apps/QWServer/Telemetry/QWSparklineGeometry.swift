// QWSparklineGeometry.swift — Pure dual-band sparkline geometry layout
// A nil sample breaks the line into a gap instead of flattening to zero.

import Foundation

/// A positioned sparkline point for an individually measured telemetry turn.
public struct QWSparklinePoint: Equatable, Sendable {
    public let index: Int
    public let x: Double
    public let y: Double
    public let value: Double

    public init(index: Int, x: Double, y: Double, value: Double) {
        self.index = index
        self.x = x
        self.y = y
        self.value = value
    }
}

/// Geometry primitives extracted for a single sparkline series.
public struct QWSparklineSeriesGeometry: Equatable, Sendable {
    public let runs: [[QWSparklinePoint]]
    public let isolatedPoints: [QWSparklinePoint]
    public let latestPoint: QWSparklinePoint?
    public let singleSlotGuideY: Double?

    public init(
        runs: [[QWSparklinePoint]],
        isolatedPoints: [QWSparklinePoint],
        latestPoint: QWSparklinePoint?,
        singleSlotGuideY: Double?
    ) {
        self.runs = runs
        self.isolatedPoints = isolatedPoints
        self.latestPoint = latestPoint
        self.singleSlotGuideY = singleSlotGuideY
    }
}

/// Combined dual-series sparkline layout shared by native and web renderers.
public struct QWSparklinePlot: Equatable, Sendable {
    public let prefill: QWSparklineSeriesGeometry
    public let decode: QWSparklineSeriesGeometry
    public let totalSlots: Int

    public init(
        prefill: QWSparklineSeriesGeometry,
        decode: QWSparklineSeriesGeometry,
        totalSlots: Int
    ) {
        self.prefill = prefill
        self.decode = decode
        self.totalSlots = totalSlots
    }
}

/// Computes non-overlapping vertical bands and gap-preserving runs for dual traces.
public enum QWSparklineGeometry {
    /// Returns per-series peak heights within the plot so prefill and decode stay distinct.
    public static func ceilings(
        prefillPeak peakP: Double,
        decodePeak peakD: Double
    ) -> (prefill: Double, decode: Double) {
        if peakP > 0 && peakD > 0 {
            if peakP > peakD * 1.5 { return (0.84, 0.60) }
            if peakD > peakP * 1.5 { return (0.60, 0.84) }
            return (0.84, 0.70)
        }
        return (0.80, 0.80)
    }

    /// Maps prefill and decode series into canvas coordinates while preserving nil gaps.
    public static func layout(
        prefill: [Double?],
        decode: [Double?],
        width: Double,
        height: Double,
        insetX: Double = 6.0,
        insetY: Double = 5.0
    ) -> QWSparklinePlot {
        let totalSlots = max(prefill.count, decode.count)
        let usableW = max(1.0, width - 2.0 * insetX)
        let usableH = max(1.0, height - 2.0 * insetY)
        let peakP = prefill.compactMap { $0 }.filter { $0 > 0 }.max() ?? 0.0
        let peakD = decode.compactMap { $0 }.filter { $0 > 0 }.max() ?? 0.0
        let (ceilP, ceilD) = ceilings(prefillPeak: peakP, decodePeak: peakD)

        return QWSparklinePlot(
            prefill: buildSeries(
                data: prefill, companion: decode, peak: peakP, companionPeak: peakD,
                ceiling: ceilP, totalSlots: totalSlots, width: width, height: height,
                insetX: insetX, insetY: insetY, usableW: usableW, usableH: usableH
            ),
            decode: buildSeries(
                data: decode, companion: prefill, peak: peakD, companionPeak: peakP,
                ceiling: ceilD, totalSlots: totalSlots, width: width, height: height,
                insetX: insetX, insetY: insetY, usableW: usableW, usableH: usableH
            ),
            totalSlots: totalSlots
        )
    }

    private static func buildSeries(
        data: [Double?],
        companion: [Double?],
        peak: Double,
        companionPeak: Double,
        ceiling: Double,
        totalSlots: Int,
        width: Double,
        height: Double,
        insetX: Double,
        insetY: Double,
        usableW: Double,
        usableH: Double
    ) -> QWSparklineSeriesGeometry {
        var runs: [[QWSparklinePoint]] = []
        var isolated: [QWSparklinePoint] = []
        var current: [QWSparklinePoint] = []
        var latest: QWSparklinePoint?

        func flushCurrent() {
            if current.count >= 2 {
                runs.append(current)
            } else if current.count == 1 {
                isolated.append(current[0])
            }
            current.removeAll(keepingCapacity: true)
        }

        let maxRef = max(1.0, peak > 0 ? peak / ceiling : 1.0)
        for (index, raw) in data.enumerated() {
            guard let raw else {
                // Skip nil slots while the companion phase is actively running so 1Hz decode
                // windows do not fragment prefill measurements into isolated dots.
                if index < companion.count, let comp = companion[index], comp > 0 {
                    continue
                }
                flushCurrent()
                continue
            }
            let val = max(0.0, raw)
            let x: Double = totalSlots <= 1
                ? (width - insetX)
                : (insetX + (Double(index) / Double(totalSlots - 1)) * usableW)
            let norm = peak > 0 ? min(1.0, max(0.0, val / maxRef)) : 0.0
            let y = height - insetY - norm * usableH
            let pt = QWSparklinePoint(index: index, x: x, y: y, value: val)
            current.append(pt)
            latest = pt
        }
        flushCurrent()

        let showGuide = runs.isEmpty && isolated.count == 1 && (totalSlots == 1 || companionPeak > 0)
        let guideY = showGuide ? isolated[0].y : nil
        return QWSparklineSeriesGeometry(
            runs: runs, isolatedPoints: isolated,
            latestPoint: latest, singleSlotGuideY: guideY
        )
    }
}
