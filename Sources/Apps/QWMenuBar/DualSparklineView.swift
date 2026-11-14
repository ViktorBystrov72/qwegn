// DualSparklineView.swift — prefill/decode sparkline card
// A sample that was not measured is a gap in the line, never a zero on
// the floor: a flat line at zero would claim the engine did nothing.

import SwiftUI
import QWServer

/// Immediate-mode dual-band Canvas sparkline preserving nil telemetry gaps.
public struct DualSparklineView: View {
    public var prefillData: [Double?]
    public var decodeData: [Double?]

    public init(prefillData: [Double?], decodeData: [Double?]) {
        self.prefillData = prefillData
        self.decodeData = decodeData
    }

    public var body: some View {
        let prefill = prefillData
        let decode = decodeData
        Canvas { context, size in
            guard size.width > 4, size.height > 4 else { return }
            let frameRect = CGRect(origin: .zero, size: size)
            let framePath = RoundedRectangle(cornerRadius: 7).path(in: frameRect)
            context.fill(framePath, with: .color(Color.black.opacity(0.20)))
            context.stroke(framePath, with: .color(Color.white.opacity(0.07)), lineWidth: 0.6)

            let gridStyle = StrokeStyle(lineWidth: 0.6, dash: [3, 3])
            for fraction in [0.25, 0.50, 0.75] {
                let y = size.height * fraction
                let grid = Path { path in
                    path.move(to: CGPoint(x: 4, y: y))
                    path.addLine(to: CGPoint(x: size.width - 4, y: y))
                }
                context.stroke(grid, with: .color(Color.white.opacity(0.05)), style: gridStyle)
            }

            let plot = QWSparklineGeometry.layout(
                prefill: prefill,
                decode: decode,
                width: size.width,
                height: size.height,
                insetX: 7.0,
                insetY: 6.0
            )
            Self.drawSeries(plot.prefill, color: TelemetryTheme.prefill, in: &context, size: size)
            Self.drawSeries(plot.decode, color: TelemetryTheme.accent, in: &context, size: size)
        }
        .frame(maxWidth: .infinity)
        .accessibilityLabel("Throughput trace")
    }

    private static func drawSeries(
        _ series: QWSparklineSeriesGeometry,
        color: Color,
        in context: inout GraphicsContext,
        size: CGSize
    ) {
        let bottomY = size.height - 6.0
        let leftX: CGFloat = 7.0
        let rightX = max(leftX + 1.0, size.width - 7.0)

        if let guideY = series.singleSlotGuideY {
            let y = CGFloat(guideY)
            let band = Path { path in
                path.move(to: CGPoint(x: leftX, y: y))
                path.addLine(to: CGPoint(x: rightX, y: y))
                path.addLine(to: CGPoint(x: rightX, y: bottomY))
                path.addLine(to: CGPoint(x: leftX, y: bottomY))
                path.closeSubpath()
            }
            context.fill(
                band,
                with: .linearGradient(
                    Gradient(colors: [color.opacity(0.14), .clear]),
                    startPoint: CGPoint(x: 0, y: y),
                    endPoint: CGPoint(x: 0, y: bottomY)
                )
            )
            let guideLine = Path { path in
                path.move(to: CGPoint(x: leftX, y: y))
                path.addLine(to: CGPoint(x: rightX, y: y))
            }
            context.stroke(
                guideLine,
                with: .color(color.opacity(0.60)),
                style: StrokeStyle(lineWidth: 1.25, lineCap: .round, dash: [4, 3])
            )
        }

        let lineStyle = StrokeStyle(lineWidth: 1.65, lineCap: .round, lineJoin: .round)
        for run in series.runs where run.count >= 2 {
            let area = Self.areaPath(run, bottomY: bottomY)
            let topY = run.map { CGFloat($0.y) }.min() ?? 0
            context.fill(
                area,
                with: .linearGradient(
                    Gradient(colors: [color.opacity(0.26), .clear]),
                    startPoint: CGPoint(x: 0, y: topY),
                    endPoint: CGPoint(x: 0, y: bottomY)
                )
            )
            context.stroke(Self.linePath(run), with: .color(color), style: lineStyle)
        }

        var markers = series.isolatedPoints
        if let latest = series.latestPoint, !markers.contains(where: { $0.index == latest.index }) {
            markers.append(latest)
        }
        for pt in markers {
            let haloRect = CGRect(x: pt.x - 4.75, y: pt.y - 4.75, width: 9.5, height: 9.5)
            let coreRect = CGRect(x: pt.x - 2.2, y: pt.y - 2.2, width: 4.4, height: 4.4)
            context.fill(Path(ellipseIn: haloRect), with: .color(color.opacity(0.26)))
            context.fill(Path(ellipseIn: coreRect), with: .color(color))
        }
    }

    private static func areaPath(_ points: [QWSparklinePoint], bottomY: CGFloat) -> Path {
        Path { path in
            guard let first = points.first, let last = points.last else { return }
            path.move(to: CGPoint(x: first.x, y: bottomY))
            for pt in points { path.addLine(to: CGPoint(x: pt.x, y: pt.y)) }
            path.addLine(to: CGPoint(x: last.x, y: bottomY))
            path.closeSubpath()
        }
    }

    private static func linePath(_ points: [QWSparklinePoint]) -> Path {
        Path { path in
            guard let first = points.first else { return }
            path.move(to: CGPoint(x: first.x, y: first.y))
            for pt in points.dropFirst() { path.addLine(to: CGPoint(x: pt.x, y: pt.y)) }
        }
    }
}
