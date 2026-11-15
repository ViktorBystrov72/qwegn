// MarqueeText.swift — Bounded single-line running text for long model names and traits

import AppKit
import SwiftUI

/// Pure timing and wrap math for bounded horizontal marquee scrolling.
enum MarqueeMath: Sendable {
    static func offset(
        elapsed: TimeInterval,
        textWidth: CGFloat,
        containerWidth: CGFloat,
        gap: CGFloat = 28,
        holdSeconds: Double = 1.4,
        speedPtPerSec: Double = 28
    ) -> CGFloat {
        guard containerWidth > 0,
              textWidth > containerWidth + 0.5,
              elapsed > 0,
              speedPtPerSec > 0 else {
            return 0
        }
        let cycleDistance = Double(textWidth + max(0, gap))
        let scrollDuration = cycleDistance / speedPtPerSec
        let period = max(0.01, holdSeconds + scrollDuration)
        let phase = elapsed.truncatingRemainder(dividingBy: period)
        guard phase >= holdSeconds else { return 0 }
        return -CGFloat((phase - holdSeconds) * speedPtPerSec)
    }
}

/// Bounded single-line label that scrolls smoothly when wider than its slot.
struct MarqueeText: View {
    let text: String
    let fontSize: CGFloat
    let weight: Font.Weight
    let nsWeight: NSFont.Weight
    let color: Color
    var shrinksToFit: Bool = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var cycleStart = Date()
    @State private var measuredCopyWidth: CGFloat = 0

    private static let loopGap: CGFloat = 28

    var body: some View {
        MarqueeSlotLayout(shrinksToFit: shrinksToFit) {
            Text(text)
                .font(.system(size: fontSize, weight: weight))
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .hidden()
                .accessibilityHidden(true)

            ViewThatFits(in: .horizontal) {
                staticCopy
                    .fixedSize(horizontal: true, vertical: false)

                scrollingSlot
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(text)
        .help(text)
        .onChange(of: text) { _, _ in
            cycleStart = Date()
            measuredCopyWidth = 0
        }
    }

    private var staticCopy: some View {
        Text(text)
            .font(.system(size: fontSize, weight: weight))
            .foregroundStyle(color)
            .lineLimit(1)
    }

    private var fallbackCopyWidth: CGFloat {
        let font = NSFont.systemFont(ofSize: fontSize, weight: nsWeight)
        return ceil((text as NSString).size(withAttributes: [.font: font]).width)
    }

    private var scrollingSlot: some View {
        GeometryReader { geo in
            let slotWidth = max(0, geo.size.width)
            let slotHeight = max(0, geo.size.height)
            let singleWidth = max(measuredCopyWidth, fallbackCopyWidth)
            if reduceMotion {
                staticCopy
                    .truncationMode(.tail)
                    .frame(width: slotWidth, height: slotHeight, alignment: .leading)
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { context in
                    let elapsed = context.date.timeIntervalSince(cycleStart)
                    let xOffset = MarqueeMath.offset(
                        elapsed: elapsed,
                        textWidth: singleWidth,
                        containerWidth: slotWidth,
                        gap: Self.loopGap
                    )
                    HStack(spacing: Self.loopGap) {
                        staticCopy
                            .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { newWidth in
                                if abs(measuredCopyWidth - newWidth) > 0.25 {
                                    measuredCopyWidth = newWidth
                                }
                            }
                        staticCopy
                            .accessibilityHidden(true)
                    }
                    .fixedSize(horizontal: true, vertical: false)
                    .offset(x: xOffset)
                    .frame(width: slotWidth, height: slotHeight, alignment: .leading)
                    .mask(edgeFadeMask(width: slotWidth, isLeadingFaded: xOffset < -0.5))
                }
            }
        }
        .clipped()
    }

    private func edgeFadeMask(width: CGFloat, isLeadingFaded: Bool) -> some View {
        let fadeFraction = width > 0 ? min(0.14, 10.0 / width) : 0
        let leadingColor: Color = isLeadingFaded ? .clear : .black
        let stops: [Gradient.Stop] = fadeFraction > 0 ? [
            .init(color: leadingColor, location: 0),
            .init(color: .black, location: fadeFraction),
            .init(color: .black, location: 1 - fadeFraction),
            .init(color: .clear, location: 1),
        ] : [.init(color: .black, location: 0), .init(color: .black, location: 1)]
        return LinearGradient(gradient: Gradient(stops: stops), startPoint: .leading, endPoint: .trailing)
    }
}

/// Constrains slot width synchronously to parent bounds while keeping exact line height.
private struct MarqueeSlotLayout: Layout {
    let shrinksToFit: Bool

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let ideal = subviews.first?.sizeThatFits(.unspecified) ?? .zero
        guard let w = proposal.width, w.isFinite else { return ideal }
        let bounded = shrinksToFit ? min(ideal.width, max(0, w)) : max(0, w)
        return CGSize(width: bounded, height: ideal.height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        guard subviews.count >= 2 else { return }
        subviews[0].place(at: bounds.origin, anchor: .topLeading, proposal: .unspecified)
        subviews[1].place(
            at: CGPoint(x: bounds.minX, y: bounds.midY),
            anchor: .leading,
            proposal: ProposedViewSize(width: bounds.width, height: bounds.height)
        )
    }
}
