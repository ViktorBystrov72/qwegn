// QGProcessorLogoView.swift — Unified QWG processor mark and hex Color helper

import AppKit
import SwiftUI

/// Renders the layered Q·W·G silicon die mark with optical micro sizing.
public struct QGProcessorLogoView: View {
    public init() {}

    public var body: some View {
        Canvas { context, size in
            let side = min(size.width, size.height)
            guard side > 0 else { return }
            let isMicro = side <= 20
            let scale = side / 600.0
            let offsetX = (size.width - side) * 0.5
            let offsetY = (size.height - side) * 0.5
            let transform = CGAffineTransform(translationX: offsetX, y: offsetY)
                .scaledBy(x: scale, y: scale)

            let bracketStroke: CGFloat = (isMicro ? 32 : 24) * scale
            let waveStroke: CGFloat = (isMicro ? 30 : 26) * scale
            let glyphStroke: CGFloat = (isMicro ? 34 : 28) * scale
            let waveOpacity: Double = isMicro ? 0.42 : 0.38
            let style = StrokeStyle(lineWidth: bracketStroke, lineCap: .round, lineJoin: .round)

            if !isMicro {
                for pin in Self.pinsPath() {
                    context.stroke(Path(pin).applying(transform), with: .foreground, style: style)
                }
            }
            for bracket in Self.bracketsPath() {
                context.stroke(Path(bracket).applying(transform), with: .foreground, style: style)
            }
            var waveContext = context
            waveContext.opacity = waveOpacity
            waveContext.stroke(
                Path(Self.wavePath()).applying(transform),
                with: .foreground,
                style: StrokeStyle(lineWidth: waveStroke, lineCap: .round, lineJoin: .round)
            )
            let glyphStyle = StrokeStyle(lineWidth: glyphStroke, lineCap: .round, lineJoin: .round)
            for glyph in Self.glyphPaths() {
                context.stroke(Path(glyph).applying(transform), with: .foreground, style: glyphStyle)
            }
        }
        .accessibilityLabel("Qwegn")
    }

    /// Draws the QWG mark into a top-left-origin CoreGraphics context.
    public static func draw(in ctx: CGContext, size: CGSize, strokeColor: NSColor) {
        let side = min(size.width, size.height)
        guard side > 0 else { return }
        let isMicro = side <= 20
        let scale = side / 600.0

        ctx.saveGState()
        ctx.translateBy(x: (size.width - side) * 0.5, y: (size.height - side) * 0.5)
        ctx.scaleBy(x: scale, y: scale)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)

        ctx.setStrokeColor(strokeColor.cgColor)
        ctx.setLineWidth(isMicro ? 32 : 24)
        if !isMicro {
            for pin in pinsPath() {
                ctx.addPath(pin)
                ctx.strokePath()
            }
        }
        for bracket in bracketsPath() {
            ctx.addPath(bracket)
            ctx.strokePath()
        }

        let waveAlpha = strokeColor.alphaComponent * (isMicro ? 0.42 : 0.38)
        ctx.setStrokeColor(strokeColor.withAlphaComponent(waveAlpha).cgColor)
        ctx.setLineWidth(isMicro ? 30 : 26)
        ctx.addPath(wavePath())
        ctx.strokePath()

        ctx.setStrokeColor(strokeColor.cgColor)
        ctx.setLineWidth(isMicro ? 34 : 28)
        for glyph in glyphPaths() {
            ctx.addPath(glyph)
            ctx.strokePath()
        }
        ctx.restoreGState()
    }

    /// Builds a dynamic Light/Dark NSStatusItem icon with optional status badge.
    public static func statusBarImage(
        pointSize: CGFloat = 18,
        statusDotColor: NSColor?,
        dimmed: Bool = false
    ) -> NSImage {
        let image = NSImage(size: NSSize(width: pointSize, height: pointSize), flipped: true) { rect in
            guard let ctx = NSGraphicsContext.current?.cgContext else { return false }
            let inkAlpha: CGFloat = dimmed ? 0.45 : 1.0
            let stroke = (statusDotColor == nil ? NSColor.black : NSColor.labelColor)
                .withAlphaComponent(inkAlpha)
            let markRect = rect.insetBy(dx: 0.5, dy: 0.5)
            ctx.saveGState()
            ctx.translateBy(x: markRect.minX, y: markRect.minY)
            draw(in: ctx, size: markRect.size, strokeColor: stroke)
            ctx.restoreGState()

            if let dotColor = statusDotColor {
                let dotSize: CGFloat = 5.0
                let dotRect = CGRect(
                    x: rect.maxX - dotSize - 0.5,
                    y: rect.maxY - dotSize - 0.5,
                    width: dotSize,
                    height: dotSize
                )
                ctx.saveGState()
                ctx.setBlendMode(.clear)
                ctx.fillEllipse(in: dotRect.insetBy(dx: -1.1, dy: -1.1))
                ctx.setBlendMode(.normal)
                ctx.setFillColor(dotColor.cgColor)
                ctx.fillEllipse(in: dotRect)
                ctx.restoreGState()
            }
            return true
        }
        image.isTemplate = (statusDotColor == nil)
        return image
    }

    private static func pinsPath() -> [CGPath] {
        [220, 300, 380].flatMap { pos -> [CGPath] in
            let p = CGFloat(pos)
            return [
                segment(CGPoint(x: p, y: 62), CGPoint(x: p, y: 106)),
                segment(CGPoint(x: p, y: 494), CGPoint(x: p, y: 538)),
                segment(CGPoint(x: 62, y: p), CGPoint(x: 106, y: p)),
                segment(CGPoint(x: 494, y: p), CGPoint(x: 538, y: p)),
            ]
        }
    }

    private static func bracketsPath() -> [CGPath] {
        let tl = CGMutablePath()
        tl.move(to: CGPoint(x: 272, y: 132))
        tl.addLine(to: CGPoint(x: 178, y: 132))
        tl.addQuadCurve(to: CGPoint(x: 132, y: 178), control: CGPoint(x: 132, y: 132))
        tl.addLine(to: CGPoint(x: 132, y: 272))

        let tr = CGMutablePath()
        tr.move(to: CGPoint(x: 328, y: 132))
        tr.addLine(to: CGPoint(x: 422, y: 132))
        tr.addQuadCurve(to: CGPoint(x: 468, y: 178), control: CGPoint(x: 468, y: 132))
        tr.addLine(to: CGPoint(x: 468, y: 272))

        let bl = CGMutablePath()
        bl.move(to: CGPoint(x: 132, y: 328))
        bl.addLine(to: CGPoint(x: 132, y: 422))
        bl.addQuadCurve(to: CGPoint(x: 178, y: 468), control: CGPoint(x: 132, y: 468))
        bl.addLine(to: CGPoint(x: 272, y: 468))

        let br = CGMutablePath()
        br.move(to: CGPoint(x: 468, y: 328))
        br.addLine(to: CGPoint(x: 468, y: 422))
        br.addQuadCurve(to: CGPoint(x: 422, y: 468), control: CGPoint(x: 468, y: 468))
        br.addLine(to: CGPoint(x: 328, y: 468))
        return [tl, tr, bl, br]
    }

    private static func wavePath() -> CGPath {
        let wave = CGMutablePath()
        wave.move(to: CGPoint(x: 164, y: 206))
        wave.addLine(to: CGPoint(x: 232, y: 414))
        wave.addLine(to: CGPoint(x: 300, y: 264))
        wave.addLine(to: CGPoint(x: 368, y: 414))
        wave.addLine(to: CGPoint(x: 436, y: 206))
        return wave
    }

    private static func glyphPaths() -> [CGPath] {
        let qLoop = CGPath(
            roundedRect: CGRect(x: 172, y: 182, width: 106, height: 232),
            cornerWidth: 38,
            cornerHeight: 38,
            transform: nil
        )
        let qTail = segment(CGPoint(x: 238, y: 350), CGPoint(x: 294, y: 426))

        let gPath = CGMutablePath()
        gPath.move(to: CGPoint(x: 426, y: 226))
        gPath.addQuadCurve(to: CGPoint(x: 382, y: 182), control: CGPoint(x: 426, y: 182))
        gPath.addLine(to: CGPoint(x: 366, y: 182))
        gPath.addQuadCurve(to: CGPoint(x: 322, y: 226), control: CGPoint(x: 322, y: 182))
        gPath.addLine(to: CGPoint(x: 322, y: 370))
        gPath.addQuadCurve(to: CGPoint(x: 366, y: 414), control: CGPoint(x: 322, y: 414))
        gPath.addLine(to: CGPoint(x: 386, y: 414))
        gPath.addQuadCurve(to: CGPoint(x: 428, y: 372), control: CGPoint(x: 428, y: 414))
        gPath.addLine(to: CGPoint(x: 428, y: 310))
        gPath.addLine(to: CGPoint(x: 372, y: 310))
        return [qLoop, qTail, gPath]
    }

    private static func segment(_ a: CGPoint, _ b: CGPoint) -> CGPath {
        let path = CGMutablePath()
        path.move(to: a)
        path.addLine(to: b)
        return path
    }
}

public extension Color {
    /// Initializes a Color from a CSS hex string (`#RRGGBB` or `#RRGGBBAA`).
    init(hex: String) {
        let cleaned = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var value: UInt64 = 0
        Scanner(string: cleaned).scanHexInt64(&value)
        let r, g, b, a: UInt64
        switch cleaned.count {
        case 6:
            (r, g, b, a) = (value >> 16, (value >> 8) & 0xFF, value & 0xFF, 0xFF)
        case 8:
            (r, g, b, a) = (value >> 24, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
        default:
            (r, g, b, a) = (0, 0, 0, 0xFF)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255.0,
            green: Double(g) / 255.0,
            blue: Double(b) / 255.0,
            opacity: Double(a) / 255.0
        )
    }
}
