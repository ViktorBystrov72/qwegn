// gen_app_icon.swift — Renders Qwegn AppIcon.iconset PNGs using CoreGraphics.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func drawQWGMark(in ctx: CGContext, size: CGFloat) {
    let isMicro = size <= 32
    let pad = size * (isMicro ? 0.14 : 0.18)
    let scale = (size - 2 * pad) / 600.0

    ctx.saveGState()
    // Flip into top-left SVG space inside the padded square.
    ctx.translateBy(x: pad, y: size - pad)
    ctx.scaleBy(x: scale, y: -scale)
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)

    let outerColor = CGColor(srgbRed: 0.78, green: 0.88, blue: 1.0, alpha: 0.90)
    let waveColor = CGColor(srgbRed: 0.24, green: 0.56, blue: 1.0, alpha: isMicro ? 0.68 : 0.58)
    let fgColor = CGColor(srgbRed: 0.94, green: 0.97, blue: 1.0, alpha: 1.0)

    ctx.setStrokeColor(outerColor)
    ctx.setLineWidth(isMicro ? 32 : 24)
    if !isMicro {
        for pos in [220, 300, 380] {
            let p = CGFloat(pos)
            ctx.move(to: CGPoint(x: p, y: 62)); ctx.addLine(to: CGPoint(x: p, y: 106))
            ctx.move(to: CGPoint(x: p, y: 494)); ctx.addLine(to: CGPoint(x: p, y: 538))
            ctx.move(to: CGPoint(x: 62, y: p)); ctx.addLine(to: CGPoint(x: 106, y: p))
            ctx.move(to: CGPoint(x: 494, y: p)); ctx.addLine(to: CGPoint(x: 538, y: p))
        }
        ctx.strokePath()
    }

    let brackets: [(CGPoint, CGPoint, CGPoint, CGPoint, CGPoint)] = [
        (CGPoint(x: 272, y: 132), CGPoint(x: 178, y: 132), CGPoint(x: 132, y: 132), CGPoint(x: 132, y: 178), CGPoint(x: 132, y: 272)),
        (CGPoint(x: 328, y: 132), CGPoint(x: 422, y: 132), CGPoint(x: 468, y: 132), CGPoint(x: 468, y: 178), CGPoint(x: 468, y: 272)),
        (CGPoint(x: 132, y: 328), CGPoint(x: 132, y: 422), CGPoint(x: 132, y: 468), CGPoint(x: 178, y: 468), CGPoint(x: 272, y: 468)),
        (CGPoint(x: 468, y: 328), CGPoint(x: 468, y: 422), CGPoint(x: 468, y: 468), CGPoint(x: 422, y: 468), CGPoint(x: 328, y: 468)),
    ]
    for (start, lineEnd, ctrl, cornerEnd, end) in brackets {
        ctx.move(to: start)
        ctx.addLine(to: lineEnd)
        ctx.addQuadCurve(to: cornerEnd, control: ctrl)
        ctx.addLine(to: end)
    }
    ctx.strokePath()

    ctx.setStrokeColor(waveColor)
    ctx.setLineWidth(isMicro ? 30 : 26)
    ctx.move(to: CGPoint(x: 164, y: 206))
    ctx.addLine(to: CGPoint(x: 232, y: 414))
    ctx.addLine(to: CGPoint(x: 300, y: 264))
    ctx.addLine(to: CGPoint(x: 368, y: 414))
    ctx.addLine(to: CGPoint(x: 436, y: 206))
    ctx.strokePath()

    ctx.setStrokeColor(fgColor)
    ctx.setLineWidth(isMicro ? 34 : 28)
    let qRect = CGRect(x: 172, y: 182, width: 106, height: 232)
    ctx.addPath(CGPath(roundedRect: qRect, cornerWidth: 38, cornerHeight: 38, transform: nil))
    ctx.move(to: CGPoint(x: 238, y: 350))
    ctx.addLine(to: CGPoint(x: 294, y: 426))

    ctx.move(to: CGPoint(x: 426, y: 226))
    ctx.addQuadCurve(to: CGPoint(x: 382, y: 182), control: CGPoint(x: 426, y: 182))
    ctx.addLine(to: CGPoint(x: 366, y: 182))
    ctx.addQuadCurve(to: CGPoint(x: 322, y: 226), control: CGPoint(x: 322, y: 182))
    ctx.addLine(to: CGPoint(x: 322, y: 370))
    ctx.addQuadCurve(to: CGPoint(x: 366, y: 414), control: CGPoint(x: 322, y: 414))
    ctx.addLine(to: CGPoint(x: 386, y: 414))
    ctx.addQuadCurve(to: CGPoint(x: 428, y: 372), control: CGPoint(x: 428, y: 414))
    ctx.addLine(to: CGPoint(x: 428, y: 310))
    ctx.addLine(to: CGPoint(x: 372, y: 310))
    ctx.strokePath()
    ctx.restoreGState()
}

func renderIcon(pixels: Int, to url: URL) throws {
    let cs = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    guard let ctx = CGContext(
        data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0,
        space: cs, bitmapInfo: bitmapInfo
    ) else {
        throw NSError(domain: "gen_app_icon", code: 1, userInfo: [NSLocalizedDescriptionKey: "CGContext failed"])
    }
    let s = CGFloat(pixels)
    let inset = s * 0.098
    let tileRect = CGRect(x: inset, y: inset, width: s - 2 * inset, height: s - 2 * inset)
    let tilePath = CGPath(roundedRect: tileRect, cornerWidth: s * 0.18, cornerHeight: s * 0.18, transform: nil)

    ctx.saveGState()
    ctx.addPath(tilePath)
    ctx.clip()
    let colors = [
        CGColor(srgbRed: 0.11, green: 0.14, blue: 0.21, alpha: 1.0),
        CGColor(srgbRed: 0.05, green: 0.06, blue: 0.09, alpha: 1.0),
    ] as CFArray
    let grad = CGGradient(colorsSpace: cs, colors: colors, locations: [0.0, 1.0])!
    ctx.drawLinearGradient(grad, start: CGPoint(x: s * 0.5, y: s), end: CGPoint(x: s * 0.5, y: 0), options: [])
    ctx.restoreGState()

    ctx.setStrokeColor(CGColor(srgbRed: 0.35, green: 0.50, blue: 0.75, alpha: 0.32))
    ctx.setLineWidth(max(1.0, s * 0.012))
    ctx.addPath(tilePath)
    ctx.strokePath()

    drawQWGMark(in: ctx, size: s)

    guard let image = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "gen_app_icon", code: 2, userInfo: [NSLocalizedDescriptionKey: "PNG encode failed"])
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "gen_app_icon", code: 3, userInfo: [NSLocalizedDescriptionKey: "PNG write failed"])
    }
}

guard CommandLine.arguments.count == 2 else {
    fputs("Usage: swift Scripts/gen_app_icon.swift <AppIcon.iconset>\n", stderr)
    exit(1)
}
let outDir = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let specs: [(String, Int)] = [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
]
for (name, px) in specs {
    try renderIcon(pixels: px, to: outDir.appendingPathComponent(name))
}
