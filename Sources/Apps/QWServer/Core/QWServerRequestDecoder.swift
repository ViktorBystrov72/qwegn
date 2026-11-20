// QWServerRequestDecoder.swift — Shared JSON request decoder preserving key
// insertion order for nested QWJson payloads across all HTTP endpoints.

import Foundation
import QWToken

public enum QWServerRequestDecoder {
    public static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let decoder = JSONDecoder()
        if let orderedRoot = try? QWJson.parse(data) {
            decoder.userInfo[QWJson.orderedRootUserInfoKey] = orderedRoot
        }
        return try decoder.decode(type, from: data)
    }
}
