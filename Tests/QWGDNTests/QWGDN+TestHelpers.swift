// QWGDN+TestHelpers.swift — Test-only scratch element helpers for QWGDNTests.
@testable import QWGDN

extension QWGDNPrefill {
    static func uwabElems(T: Int) -> Int { T * 6144 }
    static func oScratchElems(T: Int) -> Int { T * 6144 }
}

extension QWGDNDecode {
    static var oScratchElems: Int { vHeads * headDim }
}
