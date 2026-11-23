// ReleaseMarkdownView.swift — Native block and inline Markdown parser and Cupertino renderer.

import AppKit
import Foundation
import SwiftUI

enum ReleaseMarkdownAlignment: Sendable, Equatable {
    case leading
    case center
    case trailing
}

enum ReleaseMarkdownListMarker: Sendable, Equatable {
    case unordered
    case ordered(Int)
    case task(checked: Bool)

    var isOrdered: Bool {
        if case .ordered = self { return true }
        return false
    }
}

struct ReleaseMarkdownListItem: Sendable, Equatable {
    let depth: Int
    let marker: ReleaseMarkdownListMarker
    let text: String
}

enum ReleaseMarkdownBlock: Sendable, Equatable {
    case heading(level: Int, text: String)
    case paragraph(text: String)
    case blockquote(blocks: [ReleaseMarkdownBlock])
    case list(items: [ReleaseMarkdownListItem])
    case codeBlock(language: String?, code: String)
    case table(headers: [String], alignments: [ReleaseMarkdownAlignment], rows: [[String]])
    case thematicBreak
}

enum ReleaseMarkdownParser {
    static func parse(_ markdown: String, maxQuoteDepth: Int = 4) -> [ReleaseMarkdownBlock] {
        let normalized = markdown
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        return parseLines(lines, maxQuoteDepth: maxQuoteDepth)
    }

    private static func parseLines(_ lines: [String], maxQuoteDepth: Int) -> [ReleaseMarkdownBlock] {
        var blocks: [ReleaseMarkdownBlock] = []
        var i = 0
        while i < lines.count {
            let line = lines[i]
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                i += 1
                continue
            }
            if let fence = matchFence(line) {
                i += 1
                var codeLines: [String] = []
                while i < lines.count {
                    if isClosingFence(lines[i], char: fence.char, minLength: fence.length) {
                        i += 1
                        break
                    }
                    codeLines.append(lines[i])
                    i += 1
                }
                blocks.append(.codeBlock(language: fence.language, code: codeLines.joined(separator: "\n")))
                continue
            }
            if let heading = matchHeading(line) {
                blocks.append(.heading(level: heading.level, text: heading.text))
                i += 1
                continue
            }
            if line.contains("|"), i + 1 < lines.count, let aligns = parseTableDelimiter(lines[i + 1]) {
                let rawHeaders = splitTableCells(line)
                let colCount = max(rawHeaders.count, aligns.count)
                let headers = padCells(rawHeaders, count: colCount)
                let normAligns = aligns + Array(repeating: .leading, count: max(0, colCount - aligns.count))
                i += 2
                var rows: [[String]] = []
                while i < lines.count {
                    let rowLine = lines[i]
                    let trimmed = rowLine.trimmingCharacters(in: .whitespaces)
                    if trimmed.isEmpty || !rowLine.contains("|") || matchFence(rowLine) != nil
                        || matchHeading(rowLine) != nil || isThematicBreak(rowLine) {
                        break
                    }
                    rows.append(padCells(splitTableCells(rowLine), count: colCount))
                    i += 1
                }
                blocks.append(.table(headers: headers, alignments: Array(normAligns.prefix(colCount)), rows: rows))
                continue
            }
            if isThematicBreak(line) {
                blocks.append(.thematicBreak)
                i += 1
                continue
            }
            if matchQuoteLine(line) != nil {
                var qLines: [String] = []
                while i < lines.count, let inner = matchQuoteLine(lines[i]) {
                    qLines.append(inner)
                    i += 1
                }
                let innerBlocks = maxQuoteDepth > 0
                    ? parseLines(qLines, maxQuoteDepth: maxQuoteDepth - 1)
                    : [.paragraph(text: joinLines(qLines))]
                if !innerBlocks.isEmpty {
                    blocks.append(.blockquote(blocks: innerBlocks))
                }
                continue
            }
            if matchListLine(line) != nil {
                var items: [ReleaseMarkdownListItem] = []
                var indentStack: [Int] = []
                var baseIsOrdered: Bool? = nil
                while i < lines.count {
                    if let parsed = matchListLine(lines[i]), !isThematicBreak(lines[i]) {
                        if let rootOrdered = baseIsOrdered, let rootIndent = indentStack.first,
                           parsed.indent <= rootIndent, parsed.marker.isOrdered != rootOrdered {
                            break
                        }
                        if indentStack.isEmpty {
                            indentStack = [parsed.indent]
                            baseIsOrdered = parsed.marker.isOrdered
                        } else if parsed.indent > indentStack[indentStack.count - 1] {
                            indentStack.append(parsed.indent)
                        } else {
                            while indentStack.count > 1 && parsed.indent < indentStack[indentStack.count - 1] {
                                indentStack.removeLast()
                            }
                        }
                        let depth = min(indentStack.count - 1, 4)
                        var itemLines = [parsed.text]
                        i += 1
                        while i < lines.count, !lines[i].trimmingCharacters(in: .whitespaces).isEmpty, !isBlockStart(lines, at: i) {
                            itemLines.append(lines[i].trimmingCharacters(in: .whitespaces))
                            i += 1
                        }
                        items.append(ReleaseMarkdownListItem(depth: depth, marker: parsed.marker, text: joinLines(itemLines)))
                        continue
                    }
                    if lines[i].trimmingCharacters(in: .whitespaces).isEmpty,
                       i + 1 < lines.count, !isThematicBreak(lines[i + 1]), let nextParsed = matchListLine(lines[i + 1]) {
                        if let rootOrdered = baseIsOrdered, let rootIndent = indentStack.first,
                           nextParsed.indent <= rootIndent, nextParsed.marker.isOrdered != rootOrdered {
                            break
                        }
                        i += 1
                        continue
                    }
                    break
                }
                if !items.isEmpty { blocks.append(.list(items: items)) }
                continue
            }
            var pLines: [String] = []
            while i < lines.count, !lines[i].trimmingCharacters(in: .whitespaces).isEmpty, !isBlockStart(lines, at: i) {
                pLines.append(lines[i])
                i += 1
            }
            if !pLines.isEmpty { blocks.append(.paragraph(text: joinLines(pLines))) }
        }
        return blocks
    }

    private static func isBlockStart(_ lines: [String], at idx: Int) -> Bool {
        let line = lines[idx]
        if matchFence(line) != nil || matchHeading(line) != nil || isThematicBreak(line)
            || matchQuoteLine(line) != nil || matchListLine(line) != nil {
            return true
        }
        return line.contains("|") && idx + 1 < lines.count && parseTableDelimiter(lines[idx + 1]) != nil
    }

    private static func stripUpToThreeLeadingSpaces(_ line: String) -> Substring? {
        var idx = line.startIndex
        var spaces = 0
        while idx < line.endIndex && line[idx] == " " && spaces < 4 {
            spaces += 1
            idx = line.index(after: idx)
        }
        return spaces <= 3 ? line[idx...] : nil
    }

    private static func matchFence(_ line: String) -> (char: Character, length: Int, language: String?)? {
        guard let sub = stripUpToThreeLeadingSpaces(line), let first = sub.first, first == "`" || first == "~" else {
            return nil
        }
        let ticks = sub.prefix(while: { $0 == first })
        guard ticks.count >= 3 else { return nil }
        let rest = String(sub.dropFirst(ticks.count)).trimmingCharacters(in: .whitespaces)
        if first == "`" && rest.contains("`") { return nil }
        let lang = rest.split(whereSeparator: { $0.isWhitespace }).first.map(String.init)
        return (first, ticks.count, lang?.isEmpty == false ? lang : nil)
    }

    private static func isClosingFence(_ line: String, char: Character, minLength: Int) -> Bool {
        guard let sub = stripUpToThreeLeadingSpaces(line) else { return false }
        let marker = sub.prefix(while: { $0 == char })
        return marker.count >= minLength && sub.dropFirst(marker.count).trimmingCharacters(in: .whitespaces).isEmpty
    }

    private static func matchHeading(_ line: String) -> (level: Int, text: String)? {
        guard let sub = stripUpToThreeLeadingSpaces(line) else { return nil }
        let hashes = sub.prefix(while: { $0 == "#" })
        guard (1...6).contains(hashes.count) else { return nil }
        let afterHashes = sub.dropFirst(hashes.count)
        guard let sep = afterHashes.first, sep == " " || sep == "\t" else { return nil }
        var text = String(afterHashes).trimmingCharacters(in: .whitespaces)
        if text.hasSuffix("#") {
            let trimmedRight = String(text.reversed().drop(while: { $0 == "#" }).reversed())
            if trimmedRight.isEmpty || trimmedRight.last?.isWhitespace == true {
                text = trimmedRight.trimmingCharacters(in: .whitespaces)
            }
        }
        return text.isEmpty ? nil : (hashes.count, text)
    }

    private static func isThematicBreak(_ line: String) -> Bool {
        guard let sub = stripUpToThreeLeadingSpaces(line) else { return false }
        let compact = sub.filter { $0 != " " && $0 != "\t" }
        guard compact.count >= 3, let marker = compact.first, marker == "-" || marker == "*" || marker == "_" else {
            return false
        }
        return compact.allSatisfy { $0 == marker }
    }

    private static func matchQuoteLine(_ line: String) -> String? {
        guard let sub = stripUpToThreeLeadingSpaces(line), sub.first == ">" else { return nil }
        let rest = sub.dropFirst()
        return rest.first == " " ? String(rest.dropFirst()) : String(rest)
    }

    private static func matchListLine(_ line: String) -> (indent: Int, marker: ReleaseMarkdownListMarker, text: String)? {
        var idx = line.startIndex
        var indent = 0
        while idx < line.endIndex && (line[idx] == " " || line[idx] == "\t") {
            indent += (line[idx] == "\t") ? 4 : 1
            idx = line.index(after: idx)
        }
        let rest = line[idx...]
        guard let first = rest.first else { return nil }
        if (first == "-" || first == "*" || first == "+"),
           let second = rest.dropFirst().first, second == " " || second == "\t" {
            let body = String(rest.dropFirst().drop(while: { $0 == " " || $0 == "\t" }))
            return (indent, taskMarker(from: body).marker, taskMarker(from: body).text)
        }
        let digits = rest.prefix(while: { $0.isASCII && $0.isNumber })
        if !digits.isEmpty && digits.count <= 9 {
            let afterDigits = rest.dropFirst(digits.count)
            if let punct = afterDigits.first, punct == "." || punct == ")",
               let space = afterDigits.dropFirst().first, space == " " || space == "\t" {
                let startNum = Int(digits) ?? 1
                let body = String(afterDigits.dropFirst().drop(while: { $0 == " " || $0 == "\t" }))
                return (indent, .ordered(startNum), body)
            }
        }
        return nil
    }

    private static func taskMarker(from text: String) -> (marker: ReleaseMarkdownListMarker, text: String) {
        if text.hasPrefix("[ ] ") { return (.task(checked: false), String(text.dropFirst(4))) }
        if text.hasPrefix("[x] ") || text.hasPrefix("[X] ") { return (.task(checked: true), String(text.dropFirst(4))) }
        return (.unordered, text)
    }

    private static func splitTableCells(_ line: String) -> [String] {
        var s = line.trimmingCharacters(in: .whitespaces)
        if s.hasPrefix("|") { s.removeFirst() }
        if s.hasSuffix("|") && !s.hasSuffix("\\|") { s.removeLast() }
        var cells: [String] = []
        var cur = ""
        var backtickRun = 0
        var i = s.startIndex
        while i < s.endIndex {
            let ch = s[i]
            if ch == "\\" && backtickRun == 0 {
                let next = s.index(after: i)
                if next < s.endIndex && s[next] == "|" {
                    cur.append("|")
                    i = s.index(after: next)
                    continue
                }
            }
            if ch == "`" {
                var runEnd = s.index(after: i)
                var count = 1
                while runEnd < s.endIndex && s[runEnd] == "`" {
                    count += 1
                    runEnd = s.index(after: runEnd)
                }
                backtickRun = backtickRun == 0 ? count : (backtickRun == count ? 0 : backtickRun)
                cur.append(contentsOf: s[i..<runEnd])
                i = runEnd
                continue
            }
            if ch == "|" && backtickRun == 0 {
                cells.append(cur.trimmingCharacters(in: .whitespaces))
                cur = ""
            } else {
                cur.append(ch)
            }
            i = s.index(after: i)
        }
        cells.append(cur.trimmingCharacters(in: .whitespaces))
        return cells
    }

    private static func parseTableDelimiter(_ line: String) -> [ReleaseMarkdownAlignment]? {
        guard line.contains("|") else { return nil }
        let cells = splitTableCells(line)
        guard !cells.isEmpty else { return nil }
        var aligns: [ReleaseMarkdownAlignment] = []
        for cell in cells {
            let left = cell.hasPrefix(":")
            let right = cell.hasSuffix(":")
            let middle = cell.dropFirst(left ? 1 : 0).dropLast(right ? 1 : 0)
            guard middle.count >= 2 && middle.allSatisfy({ $0 == "-" }) else { return nil }
            aligns.append(left && right ? .center : (right ? .trailing : .leading))
        }
        return aligns
    }

    private static func padCells(_ cells: [String], count: Int) -> [String] {
        cells.count >= count ? Array(cells.prefix(count)) : cells + Array(repeating: "", count: count - cells.count)
    }

    private static func joinLines(_ lines: [String]) -> String {
        var joined = ""
        for idx in lines.indices {
            var seg = lines[idx]
            let hardBreak = seg.hasSuffix("  ") || seg.hasSuffix("\\")
            if hardBreak {
                seg = seg.hasSuffix("  ") ? String(seg.dropLast(2)) : String(seg.dropLast(1))
            }
            if idx > 0 {
                let prevHard = lines[idx - 1].hasSuffix("  ") || lines[idx - 1].hasSuffix("\\")
                joined += prevHard ? "\n" : " "
            }
            joined += seg
        }
        return joined
    }
}

enum ReleaseMarkdownInline {
    static func isSafeURL(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http" || scheme == "mailto" else {
            return false
        }
        return !url.absoluteString.unicodeScalars.contains { $0.value <= 0x1F || $0.value == 0x7F }
    }

    static func attributedString(
        for text: String,
        baseSize: CGFloat,
        baseWeight: Font.Weight = .regular,
        increasedContrast: Bool = false
    ) -> AttributedString {
        guard !text.isEmpty else { return AttributedString("") }
        let opts = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        var result = (try? AttributedString(markdown: text, options: opts)) ?? AttributedString(text)
        let runs = result.runs.map { ($0.range, $0.inlinePresentationIntent, $0.link) }
        for (range, intent, link) in runs {
            let intentSet = intent ?? []
            if intentSet.contains(.code) {
                let codeWeight: Font.Weight = intentSet.contains(.stronglyEmphasized) ? .semibold : .regular
                result[range].swiftUI.font = .system(size: max(10, baseSize - 0.5), weight: codeWeight, design: .monospaced)
                result[range].swiftUI.foregroundColor = TelemetryTheme.highlight
                result[range].swiftUI.backgroundColor = TelemetryTheme.cool.opacity(increasedContrast ? 0.18 : 0.11)
            } else if intentSet.contains(.stronglyEmphasized) || intentSet.contains(.emphasized) {
                let weight: Font.Weight = (baseWeight == .bold) ? .bold : (intentSet.contains(.stronglyEmphasized) ? .semibold : baseWeight)
                var font = Font.system(size: baseSize, weight: weight)
                if intentSet.contains(.emphasized) { font = font.italic() }
                result[range].swiftUI.font = font
            }
            if intentSet.contains(.strikethrough) {
                result[range].swiftUI.strikethroughStyle = .single
            }
            if let url = link {
                if isSafeURL(url) {
                    result[range].swiftUI.foregroundColor = TelemetryTheme.accent
                    result[range].swiftUI.underlineStyle = .single
                } else {
                    result[range].link = nil
                }
            }
        }
        return result
    }
}

struct ReleaseMarkdownView: View {
    let markdown: String
    var increasedContrast: Bool = false

    private var blocks: [ReleaseMarkdownBlock] {
        ReleaseMarkdownParser.parse(markdown)
    }

    var body: some View {
        Group {
            if blocks.isEmpty {
                Text("No release notes provided.")
                    .font(TelemetryTheme.ui(11.5))
                    .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(blocks.enumerated()), id: \.offset) { idx, block in
                        blockView(block, isFirst: idx == 0)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }

    private func blockView(_ block: ReleaseMarkdownBlock, isFirst: Bool) -> AnyView {
        switch block {
        case let .heading(level, text):
            return AnyView(headingView(level: level, text: text, isFirst: isFirst))
        case let .paragraph(text):
            return AnyView(
                Text(ReleaseMarkdownInline.attributedString(for: text, baseSize: 11.5, increasedContrast: increasedContrast))
                    .font(TelemetryTheme.ui(11.5))
                    .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                    .lineSpacing(2.5)
                    .fixedSize(horizontal: false, vertical: true)
            )
        case let .blockquote(inner):
            return AnyView(blockquoteView(inner))
        case let .list(items):
            return AnyView(listView(items))
        case let .codeBlock(language, code):
            return AnyView(codeBlockView(language: language, code: code))
        case let .table(headers, alignments, rows):
            return AnyView(tableView(headers: headers, alignments: alignments, rows: rows))
        case .thematicBreak:
            return AnyView(
                Rectangle()
                    .fill(TelemetryTheme.hairline(increasedContrast: increasedContrast))
                    .frame(height: 1)
                    .padding(.vertical, 3)
            )
        }
    }

    private func headingView(level: Int, text: String, isFirst: Bool) -> some View {
        let size: CGFloat = level == 1 ? 14.5 : (level == 2 ? 13 : (level == 3 ? 12 : 11.5))
        let weight: Font.Weight = level == 1 ? .bold : .semibold
        let color: Color = level <= 2
            ? TelemetryTheme.primary(increasedContrast: increasedContrast)
            : (level == 3 ? TelemetryTheme.cool : TelemetryTheme.secondary(increasedContrast: increasedContrast))
        return VStack(alignment: .leading, spacing: 4) {
            Text(ReleaseMarkdownInline.attributedString(for: text, baseSize: size, baseWeight: weight, increasedContrast: increasedContrast))
                .font(TelemetryTheme.ui(size, weight))
                .foregroundStyle(color)
                .fixedSize(horizontal: false, vertical: true)
            if level == 1 {
                Rectangle().fill(TelemetryTheme.hairline(increasedContrast: increasedContrast)).frame(height: 1)
            }
        }
        .padding(.top, isFirst ? 0 : (level <= 3 ? 4 : 2))
    }

    private func blockquoteView(_ inner: [ReleaseMarkdownBlock]) -> some View {
        HStack(alignment: .top, spacing: 8) {
            RoundedRectangle(cornerRadius: 1.5)
                .fill(TelemetryTheme.accent)
                .frame(width: 3)
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(inner.enumerated()), id: \.offset) { idx, child in
                    blockView(child, isFirst: idx == 0)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 7)
        .padding(.horizontal, 10)
        .background(TelemetryTheme.accent.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small))
    }

    private func listView(_ items: [ReleaseMarkdownListItem]) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    listMarkerView(item.marker, depth: item.depth)
                    Text(ReleaseMarkdownInline.attributedString(for: item.text, baseSize: 11.5, increasedContrast: increasedContrast))
                        .font(TelemetryTheme.ui(11.5))
                        .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                        .lineSpacing(2.5)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.leading, CGFloat(item.depth) * 14)
            }
        }
    }

    @ViewBuilder
    private func listMarkerView(_ marker: ReleaseMarkdownListMarker, depth: Int) -> some View {
        switch marker {
        case .unordered:
            Text(depth == 0 ? "•" : (depth == 1 ? "◦" : "▪"))
                .font(TelemetryTheme.ui(12, .semibold))
                .foregroundStyle(depth == 0 ? TelemetryTheme.cool : TelemetryTheme.secondary(increasedContrast: increasedContrast))
                .frame(width: 10, alignment: .center)
        case let .ordered(num):
            Text("\(num).")
                .font(TelemetryTheme.numeric(11, .medium))
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                .frame(minWidth: 16, alignment: .trailing)
        case let .task(checked):
            Image(systemName: checked ? "checkmark.square.fill" : "square")
                .font(.system(size: 11))
                .foregroundStyle(checked ? TelemetryTheme.active : TelemetryTheme.secondary(increasedContrast: increasedContrast))
                .frame(width: 14, alignment: .center)
        }
    }

    private func codeBlockView(language: String?, code: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if let language {
                Text(language)
                    .font(TelemetryTheme.numeric(9.5, .medium))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(TelemetryTheme.substrate)
                Rectangle().fill(TelemetryTheme.hairline(increasedContrast: increasedContrast)).frame(height: 0.5)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(TelemetryTheme.numeric(10.5))
                    .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                    .padding(9)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(TelemetryTheme.rowSubstrate)
        .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small))
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                .strokeBorder(TelemetryTheme.hairline(increasedContrast: increasedContrast), lineWidth: 0.75)
        )
    }

    private func tableView(headers: [String], alignments: [ReleaseMarkdownAlignment], rows: [[String]]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(Array(headers.enumerated()), id: \.offset) { col, cell in
                        tableCell(cell, alignment: alignments[col], isHeader: true)
                    }
                }
                .background(TelemetryTheme.rowSubstrate)
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    Rectangle()
                        .fill(TelemetryTheme.hairline(increasedContrast: increasedContrast))
                        .frame(height: 0.5)
                        .gridCellUnsizedAxes(.horizontal)
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { col, cell in
                            tableCell(cell, alignment: alignments[col], isHeader: false)
                        }
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                    .strokeBorder(TelemetryTheme.hairline(increasedContrast: increasedContrast), lineWidth: 0.75)
            )
        }
    }

    private func tableCell(_ text: String, alignment: ReleaseMarkdownAlignment, isHeader: Bool) -> some View {
        let colAlign: HorizontalAlignment = alignment == .center ? .center : (alignment == .trailing ? .trailing : .leading)
        let weight: Font.Weight = isHeader ? .semibold : .regular
        return Text(ReleaseMarkdownInline.attributedString(for: text, baseSize: 10.5, baseWeight: weight, increasedContrast: increasedContrast))
            .font(TelemetryTheme.ui(10.5, weight))
            .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .gridColumnAlignment(colAlign)
    }
}
