import XCTest
import Metal
@testable import QWAlloc
import QWHAL

/// C03 gate (specs/03-qwalloc.md): sub-alloc ≥20× cheaper than makeBuffer,
/// ring wrap correctness under 100k ops, page accounting exact,
/// steady-state alloc counters provable.
final class QWAllocTests: XCTestCase {

    static let hal: QWHal = try! {
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void qw_probe(device float* out [[buffer(0)]]) { out[0] = 1; }
        """
        let tmp = FileManager.default.temporaryDirectory
        let msl = tmp.appendingPathComponent("qw_alloc_probe.metal")
        let lib = tmp.appendingPathComponent("qw_alloc_probe.metallib")
        try! src.write(to: msl, atomically: true, encoding: .utf8)
        try? FileManager.default.removeItem(at: lib)
        let air = tmp.appendingPathComponent("qw_alloc_probe.air")
        let c = Process(); c.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        c.arguments = ["-sdk", "macosx", "metal", "-c", msl.path, "-o", air.path]
        try! c.run(); c.waitUntilExit()
        let l = Process(); l.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        l.arguments = ["-sdk", "macosx", "metallib", air.path, "-o", lib.path]
        try! l.run(); l.waitUntilExit()
        // Throwing, not `try!`: this HAL names an explicit library, and a
        // library in the wrong basis is a failure the test should report.
        return try QWHal(metallibPath: lib.path, geometry: .qwen38_27b)
    }()

    /// 512 MB arena — big enough to matter, small enough for CI.
    static func arena(_ mb: Int = 512) -> QWArena {
        try! QWArena(hal: hal, weights: mb << 20 / 2, activations: mb << 20 / 8,
                     kv: mb << 20 / 4, gdnStates: mb << 20 / 8)
    }

    func testSubAllocIsCheaperThanMakeBuffer() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
                      "makeBuffer timing ratio skipped on virtualized CI runner")
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable,
                          "makeBuffer timing ratio requires bare-metal GPU")
        let a = Self.arena()
        let dev = Self.hal.device   // sub-alloc timing happens below via sub2
        let sz = 1 << 20
        // baseline: makeBuffer
        var t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<200 { _ = dev.makeBuffer(length: sz, options: .storageModeShared) }
        let base = CFAbsoluteTimeGetCurrent() - t0
        // arena sub-alloc (weights region = 256 MB in the 512 MB test arena)
        let sub2 = a.weightsRegion()
        t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<200 { _ = try sub2.allocate(bytes: sz) }
        let ours = CFAbsoluteTimeGetCurrent() - t0
        let ratio = base / max(ours, 1e-9)
        print(String(format: "makeBuffer %.0f µs/op vs arena %.3f µs/op → %.0f×",
                     base / 200 * 1e6, ours / 200 * 1e6, ratio))
        XCTAssertGreaterThan(ratio, 20, "gate: ≥20× cheaper than makeBuffer")
    }

    func testAlignmentLaw() throws {
        let sub = Self.arena().weightsRegion()
        let o1 = try sub.allocate(bytes: 1)
        let o2 = try sub.allocate(bytes: 513)
        XCTAssertEqual(o1 % 256, 0)
        XCTAssertEqual(o2 % 256, 0)
        XCTAssertGreaterThan(o2, o1)
    }

    func testExactCapacityAndRelocatedRegionBases() throws {
        let a = try QWArena(hal: Self.hal, weights: 768, activations: 256,
                            kv: 256, gdnStates: 256)
        let sub = a.weightsRegion()
        let first = try sub.allocate(QWTensorShape([129], .f16))
        let second = try sub.allocate(QWTensorShape([1], .f32))
        XCTAssertEqual(first.offset, 0)
        XCTAssertEqual(second.offset, 512)
        XCTAssertEqual(sub.usedBytes, 768)
        XCTAssertEqual(a.activationsRegion().region.base, 768)
        XCTAssertEqual(a.kvRegion().region.base, 1024)
        XCTAssertEqual(a.gdnRegion().region.base, 1280)
        XCTAssertEqual(a.committedBytes, 1536)
        let regions = [sub.region, a.activationsRegion().region,
                       a.kvRegion().region, a.gdnRegion().region]
        for (index, region) in regions.enumerated() {
            XCTAssertEqual(region.base % 256, 0)
            region.pointer.storeBytes(of: UInt32(index + 1), as: UInt32.self)
        }
        for (index, region) in regions.enumerated() {
            XCTAssertEqual(region.pointer.load(as: UInt32.self), UInt32(index + 1))
        }
        XCTAssertThrowsError(try sub.allocate(bytes: 1))
        XCTAssertEqual(sub.usedBytes, 768)
        sub.markSteadyState()
        XCTAssertEqual(sub.steadyStateAllocs, 0)
    }

    func testRingWrapCorrectness() throws {
        let a = Self.arena()
        let ring = try a.ring(slabBytes: 4096, depth: 8)
        let shape = QWTensorShape([1024], .f16)      // 2 KB
        var offsets: [Int] = []
        for _ in 0..<100_000 { offsets.append(ring.acquire(shape).offset) }
        // wraps exactly on depth
        XCTAssertEqual(Set(offsets).count, 8)
        XCTAssertEqual(offsets[0], offsets[8])
        // every offset 256-aligned and inside the activation region
        let regionBase = a.regionSizes.weights
        for off in offsets {
            XCTAssertEqual((regionBase + off) % 256, 0)
            XCTAssertLessThan(off, a.regionSizes.act)
        }
    }

    func testPageAccounting() throws {
        let a = Self.arena()
        let pages = try a.pages(pageSize: 65_536, count: 1024)
        XCTAssertEqual(pages.freePages, 1024)
        var taken: [Int] = []
        for _ in 0..<1024 { taken.append(pages.allocPage()!) }
        XCTAssertEqual(Set(taken).count, 1024, "no double-handout")
        XCTAssertNil(pages.allocPage(), "exhaustion returns nil")
        pages.freePage(taken[5])
        XCTAssertEqual(pages.allocPage(), 5)
        // 128 B page alignment law
        for i in 0..<8 {
            XCTAssertEqual(QWTensorValue(region: pages.region, offset: i * pages.pageSize, shape: QWTensorShape([1], .f16)).offset % 128, 0)
        }
    }

    func testSteadyStateCounterArmsAndCounts() throws {
        let sub = Self.arena().weightsRegion()
        _ = try sub.allocate(bytes: 1024)          // setup alloc — not counted
        sub.markSteadyState()
        _ = try sub.allocate(bytes: 1024)          // hot-path alloc — counted
        _ = try sub.allocate(bytes: 1024)
        XCTAssertEqual(sub.steadyStateAllocs, 2, "counter must prove hot-loop allocs")
    }

    func testOutOfSpaceIsClean() throws {
        let sub = Self.arena().weightsRegion()
        XCTAssertThrowsError(try sub.allocate(bytes: 1 << 40)) { err in
            guard case QWAllocError.outOfSpace = err else { return XCTFail("\(err)") }
        }
    }

    func testCpuPointerZeroCopy() throws {
        let a = Self.arena()
        let sub = a.gdnRegion()
        let off = try sub.allocate(bytes: 4)
        let p = a.cpuPointer(in: sub, offset: off).assumingMemoryBound(to: Float.self)
        p.pointee = 7.5
        XCTAssertEqual(p.pointee, 7.5)
    }

    func testMultiSlabAllocateBytesDoesNotCorruptCursorOnOOM() throws {
        let a = Self.arena(32)
        let buf = a.allBuffers[0]
        let r0 = QWMRegion(buffer: buf, base: 0, length: 1024)
        let r1 = QWMRegion(buffer: buf, base: 1024, length: 1024)
        let sub = QWSubAllocator(regions: [r0, r1])

        let off0 = try sub.allocate(bytes: 768)
        XCTAssertEqual(off0, 0)
        // Oversized request fails cleanly without mutating slabIndex/cursor.
        XCTAssertThrowsError(try sub.allocate(bytes: 2048))
        XCTAssertThrowsError(try sub.allocate(bytes: -1))
        XCTAssertThrowsError(try sub.allocate(bytes: Int.max))
        // Subsequent request that fits in slab 1 still succeeds!
        let off1 = try sub.allocate(bytes: 512)
        XCTAssertEqual(off1, 0)
        XCTAssertEqual(sub.usedBytes, 1280)
        XCTAssertEqual(sub.freeBytes, 768)
    }

    func testWeightSlabCapNeverCollapsesAndPageArenaAndCopyReadSafety() throws {
        XCTAssertEqual(QWArena.weightSlabCap(maxBufferLength: 1024 << 20), 1024 << 20)
        XCTAssertEqual(QWArena.weightSlabCap(maxBufferLength: 1536 << 20), 1536 << 20)
        XCTAssertEqual(QWArena.weightSlabCap(maxBufferLength: 3072 << 20), 1536 << 20)

        let shape = QWTensorShape([1 << 30, 8], .q4g)
        XCTAssertEqual(shape.totalBytes, (1 << 30) * 5)
        let overflowShape = QWTensorShape([Int.max, 2], .f16)
        XCTAssertNil(overflowShape.checkedTotalBytes)

        let a = Self.arena(16)
        let sub = a.weightsRegion()
        let tv = try sub.allocate(QWTensorShape([8], .f16))
        a.copyIn(tv, from: Data()) // empty Data must not crash
        let bytes = withUnsafeBytes(of: Float(3.25).bitPattern.littleEndian) {
            (raw: UnsafeRawBufferPointer) in Data(raw)
        }
        let unalignedSlice = tv.slice(byteOffset: 2, count: 2, dtype: .f32)
        a.copyIn(unalignedSlice, from: bytes)
        let readBack = a.read(unalignedSlice, as: Float.self, count: 1)
        XCTAssertEqual(readBack, [3.25])

        let pages = try a.pages(pageSize: 4096, count: 4)
        let p0 = try XCTUnwrap(pages.allocPage())
        pages.freePage(-1)
        pages.freePage(99)
        pages.freePage(p0)
        pages.freePage(p0) // double-free ignored
        XCTAssertEqual(pages.freePages, 4)
    }

    func testContainerKindSourceOfTruthAndLegacyCompatibility() throws {
        let a = Self.arena(16)
        let sub = a.weightsRegion()
        let big = try sub.allocate(QWTensorShape([2048], .f16)) // 4096 B

        let inc = try XCTUnwrap(QWWeightView(blocks: big, n: 4, k: 128, containerKind: .incumbentHQ4G32))
        XCTAssertEqual(inc.containerKind, .incumbentHQ4G32)
        XCTAssertTrue(inc.isIncumbentContainer)
        XCTAssertTrue(inc.isRotated)
        XCTAssertFalse(inc.isSHVQ)
        XCTAssertEqual(inc.bits, 4)
        XCTAssertEqual(inc.groupSize, 32)
        XCTAssertEqual(inc.formatName, "HQ4_G32")

        let legacyInc = try XCTUnwrap(QWWeightView(blocks: big, n: 4, k: 128, isRotated: true))
        XCTAssertEqual(legacyInc.containerKind, .incumbentHQ4G32)
        XCTAssertEqual(legacyInc, inc)

        let rotHQ = try XCTUnwrap(QWWeightView(blocks: big, n: 4, k: 128, bits: 4, groupSize: 32,
                                               isRotated: true, isIncumbentContainer: false))
        XCTAssertEqual(rotHQ.containerKind, .rotatedHQ(bits: 4, group: 32))
        XCTAssertFalse(rotHQ.isIncumbentContainer)
        XCTAssertTrue(rotHQ.isRotated)
        XCTAssertNotEqual(rotHQ, inc)

        let shvq = try XCTUnwrap(QWWeightView(blocks: big, n: 4, k: 128,
                                              containerKind: .shvq(bits: 4, group: 32)))
        XCTAssertTrue(shvq.isSHVQ)
        XCTAssertTrue(shvq.isRotated)
        XCTAssertFalse(shvq.isIncumbentContainer)
        XCTAssertEqual(shvq.formatName, "SHVQ4_G32")
        let legacySHVQ = try XCTUnwrap(QWWeightView(blocks: big, n: 4, k: 128, bits: 4, groupSize: 32,
                                                    isRotated: true, isIncumbentContainer: false, isSHVQ: true))
        XCTAssertEqual(shvq.rowBytes, legacySHVQ.rowBytes)
        XCTAssertEqual(legacySHVQ, shvq)

        let affine = try XCTUnwrap(QWWeightView(blocks: big, n: 4, k: 128,
                                                containerKind: .nativeAffine(bits: 3, group: 64)))
        XCTAssertFalse(affine.isRotated)
        XCTAssertFalse(affine.isSHVQ)
        XCTAssertFalse(affine.isIncumbentContainer)
        XCTAssertEqual(affine.formatName, "AQ3_G64")
        XCTAssertNil(QWWeightView(blocks: big, n: 4, k: 128, bits: 4, groupSize: 32,
                                  isRotated: false, isSHVQ: true))
    }
}
