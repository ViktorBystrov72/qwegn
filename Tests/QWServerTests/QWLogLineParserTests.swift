// Verifies structured HTTP and system log line parsing and filter matching.

import QWServer
import XCTest

final class QWLogLineParserTests: XCTestCase {
    func testParsesHttp200LineAsHttpInfo() {
        let line = "[2026-11-14 00:00:00] POST /v1/chat/completions -> 200 OK (12.4ms)"
        let entry = QWLogLineParser.parse(id: 1, line: line)
        XCTAssertEqual(entry.id, 1)
        XCTAssertEqual(entry.kind, .http)
        XCTAssertEqual(entry.level, .info)
        XCTAssertEqual(entry.timestamp, "2026-11-14 00:00:00")
        XCTAssertFalse(entry.body.isEmpty)
    }

    func testParsesHttp500AndErrLinesAsHttpError() {
        let line500 = "[2026-11-14 00:00:01] POST /v1/chat/completions -> 500 Internal Server Error"
        let entry500 = QWLogLineParser.parse(id: 2, line: line500)
        XCTAssertEqual(entry500.kind, .http)
        XCTAssertEqual(entry500.level, .error)

        let lineErr = "[2026-11-14 00:00:02] GET /v1/models -> ERR | connection reset"
        let entryErr = QWLogLineParser.parse(id: 3, line: lineErr)
        XCTAssertEqual(entryErr.kind, .http)
        XCTAssertEqual(entryErr.level, .error)
    }

    func testParsesTimestampedVsBracketOnlySystemLines() {
        let withTimestamp = "[2026-11-14 00:00:03] [ERROR] failed to bind socket"
        let parsedWithTs = QWLogLineParser.parse(id: 4, line: withTimestamp)
        XCTAssertEqual(parsedWithTs.kind, .system)
        XCTAssertEqual(parsedWithTs.level, .error)
        XCTAssertEqual(parsedWithTs.timestamp, "2026-11-14 00:00:03")

        let bracketOnly = "[ERROR] missing configuration file"
        let parsedNoTs = QWLogLineParser.parse(id: 5, line: bracketOnly)
        XCTAssertEqual(parsedNoTs.kind, .system)
        XCTAssertEqual(parsedNoTs.level, .error)
        XCTAssertNil(parsedNoTs.timestamp)
    }

    func testMatchesAllFilterLevelsAndQuery() {
        let httpInfo = QWLogLineParser.parse(id: 10, line: "[2026-11-14 00:00:10] GET /v1/models -> 200 OK")
        let sysInfo = QWLogLineParser.parse(id: 11, line: "[2026-11-14 00:00:11] [INFO] engine ready")
        let sysWarn = QWLogLineParser.parse(id: 12, line: "[2026-11-14 00:00:12] [WARN] kv cache pressure")
        let sysError = QWLogLineParser.parse(id: 13, line: "[2026-11-14 00:00:13] [ERROR] oom guard triggered")
        let sysDebug = QWLogLineParser.parse(id: 14, line: "[2026-11-14 00:00:14] [DEBUG] trace dispatch")

        XCTAssertTrue(httpInfo.matches(filter: .all, query: ""))
        XCTAssertTrue(httpInfo.matches(filter: .http, query: ""))
        XCTAssertFalse(sysInfo.matches(filter: .http, query: ""))

        XCTAssertTrue(sysInfo.matches(filter: .info, query: ""))
        XCTAssertTrue(sysWarn.matches(filter: .warn, query: ""))
        XCTAssertTrue(sysError.matches(filter: .error, query: ""))
        XCTAssertTrue(sysDebug.matches(filter: .debug, query: ""))

        XCTAssertFalse(sysDebug.matches(filter: .info, query: ""))
        XCTAssertTrue(sysWarn.matches(filter: .all, query: "pressure"))
        XCTAssertFalse(sysWarn.matches(filter: .all, query: "nonexistent"))
    }
}
