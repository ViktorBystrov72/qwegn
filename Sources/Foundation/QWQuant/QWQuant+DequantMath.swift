// QWQuant+DequantMath.swift — Scalar f16/bf16 conversion, HQ4 dequantization, and FWHT-32.
import CryptoKit
import Foundation

extension QWQuant {
    @inline(__always)
    public static func f16tof32Bits(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let e = UInt32((h >> 10) & 0x1F)
        let m = UInt32(h & 0x3FF)
        if e == 0 {
            if m == 0 { return Float(bitPattern: sign) }
            var mb = m, mbits = 0
            while (mb & 0x400) == 0 { mb <<= 1; mbits += 1 }
            let exp = UInt32(113 - mbits)
            return Float(bitPattern: sign | (exp << 23) | ((mb << 13) & 0x7F_FFFF))
        }
        if e == 31 { return Float(bitPattern: sign | 0x7F80_0000 | (m << 13)) }
        return Float(bitPattern: sign | ((e &+ 112) << 23) | (m << 13))
    }

    @inline(__always)
    public static func f32tof16Bits(_ x: Float) -> UInt16 {
        Float16(x).bitPattern
    }

    @inline(__always)
    public static func bf16ToF16Array(
        src: UnsafePointer<UInt16>, dst: UnsafeMutablePointer<UInt16>, count: Int
    ) {
        var i = 0
        while i + 8 <= count {
            dst[i]     = Float16(Float(bitPattern: UInt32(src[i]) << 16)).bitPattern
            dst[i + 1] = Float16(Float(bitPattern: UInt32(src[i + 1]) << 16)).bitPattern
            dst[i + 2] = Float16(Float(bitPattern: UInt32(src[i + 2]) << 16)).bitPattern
            dst[i + 3] = Float16(Float(bitPattern: UInt32(src[i + 3]) << 16)).bitPattern
            dst[i + 4] = Float16(Float(bitPattern: UInt32(src[i + 4]) << 16)).bitPattern
            dst[i + 5] = Float16(Float(bitPattern: UInt32(src[i + 5]) << 16)).bitPattern
            dst[i + 6] = Float16(Float(bitPattern: UInt32(src[i + 6]) << 16)).bitPattern
            dst[i + 7] = Float16(Float(bitPattern: UInt32(src[i + 7]) << 16)).bitPattern
            i += 8
        }
        while i < count {
            dst[i] = Float16(Float(bitPattern: UInt32(src[i]) << 16)).bitPattern
            i += 1
        }
    }

    @inline(never)
    public static func mulRn(_ a: Float, _ b: Float) -> Float { a * b }

    @inline(never)
    public static func addRn(_ a: Float, _ b: Float) -> Float { a + b }

    @inline(never)
    public static func subRn(_ a: Float, _ b: Float) -> Float { a - b }

    /// Two bytes, little-endian, as the producer wrote them.
    @inline(__always)
    public static func u16le(_ bytes: UnsafeRawPointer, _ offset: Int) -> UInt16 {
        UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
    }

    /// Four bytes, little-endian, as the producer wrote them.
    @inline(__always)
    public static func u32le(_ bytes: UnsafeRawPointer, _ offset: Int) -> UInt32 {
        UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }

    /// Decodes one `Block_HQ4_G32` (20 B) into 32 values, in the block's
    /// own domain: the magic-bias form and **no** Walsh-Hadamard step.
    /// The transform belongs to the weight form, which is multiplied; a
    /// row that is added to the residual stream is stored as it is.
    @inline(__always)
    public static func dequantHQ4Group(_ blk: UnsafeRawPointer,
                                       into dst: UnsafeMutablePointer<Float>) {
        let s = QWQuant.f16tof32Bits(QWQuant.u16le(blk, 0))
        let bias = QWQuant.f16tof32Bits(QWQuant.u16le(blk, 2))
        for j in 0..<4 {
            let qs = QWQuant.u32le(blk, 4 + 4 * j)
            let b = 8 * j
            dst[b + 0] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 0) & 0xF) + 1024.0, s), bias)
            dst[b + 1] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 16) & 0xF) + 1024.0, s), bias)
            dst[b + 2] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 4) & 0xF) + 1024.0, s), bias)
            dst[b + 3] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 20) & 0xF) + 1024.0, s), bias)
            dst[b + 4] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 8) & 0xF) + 1024.0, s), bias)
            dst[b + 5] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 24) & 0xF) + 1024.0, s), bias)
            dst[b + 6] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 12) & 0xF) + 1024.0, s), bias)
            dst[b + 7] = QWQuant.addRn(QWQuant.mulRn(Float((qs >> 28) & 0xF) + 1024.0, s), bias)
        }
    }

    public static func fwht32OrthoBlock(_ p: UnsafeMutablePointer<Float>) {
        var h = 1
        while h < 32 {
            var i0 = 0
            while i0 < 32 {
                for j in 0..<h {
                    let u = p[i0 + j]
                    let v = p[i0 + j + h]
                    p[i0 + j] = addRn(u, v)
                    p[i0 + j + h] = subRn(u, v)
                }
                i0 += 2 * h
            }
            h *= 2
        }
        let scale = invSqrt32
        for i in 0..<32 {
            p[i] = mulRn(p[i], scale)
        }
    }

    static func sha256Hex(_ s: String) -> String {
        SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
