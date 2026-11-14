// DriverWeightLoading.swift — pack-to-GPU weight loading helpers used by
// QWEngineDriver.init (split from QWEngineDriver.swift; same allocation
// order as the original init-local functions).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWLoad
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    struct WeightLoader {
        let pk: QWQuant.Pack?
        let st: QWSafeTensorsContainer?
        let hl: QWHal
        let psoCvt: any MTLComputePipelineState
        let psoQuantBF16: (any MTLComputePipelineState)?
        let quantStaging: (any MTLBuffer)?
        let wsub: QWSubAllocator
        let plan: QWWeightAllocationPlan
        let geometry: QWModelGeometry
        let format: QWWeightFormat
        var weightRequest = 0
        var sharedColdBanks: (gu: QWWeightView, down: QWWeightView)?
        var hcaLsaByLayer: [Int: (wq: QWTensorValue, wk: QWTensorValue)] = [:]
        // Shared mover (QWLoad): F_NOCACHE pread straight into the destination,
        // single-memcpy mmap fallback; loading does not duplicate the pack in RAM.
        let io: PlaneIO?

        init(pk: QWQuant.Pack, hl: QWHal, psoCvt: any MTLComputePipelineState,
             wsub: QWSubAllocator, plan: QWWeightAllocationPlan, geometry: QWModelGeometry,
             format: QWWeightFormat = .hq4G32) {
            self.pk = pk; self.st = nil; self.hl = hl; self.psoCvt = psoCvt
            self.psoQuantBF16 = nil; self.quantStaging = nil
            self.wsub = wsub; self.plan = plan; self.geometry = geometry
            self.format = format
            self.io = PlaneIO(pack: pk, policy: LoadPolicy(descriptor: pk.descriptor))
        }

        init(st: QWSafeTensorsContainer, hl: QWHal, psoCvt: any MTLComputePipelineState,
             psoQuantBF16: any MTLComputePipelineState,
             wsub: QWSubAllocator, plan: QWWeightAllocationPlan, geometry: QWModelGeometry,
             format: QWWeightFormat = .hq4G32) {
            self.pk = nil; self.st = st; self.hl = hl; self.psoCvt = psoCvt
            self.psoQuantBF16 = psoQuantBF16
            self.quantStaging = format.isNativeAffine
                ? nil
                : hl.device.makeBuffer(length: 64 * 1024 * 1024, options: .storageModeShared)
            self.wsub = wsub; self.plan = plan; self.geometry = geometry
            self.format = format
            self.io = nil
        }

        static var prefixAlternates: [String] { QWCheckpointTensorLoader.prefixAlternates }

        func rawExists(_ candidate: String) -> Bool {
            if let pk { return (try? pk.entry(candidate)) != nil }
            if let st { return st.tensor(candidate) != nil }
            return false
        }

        /// The pack's name for `name`, under whichever prefix the checkpoint
        /// used. The canonical name is returned when none of them is present,
        /// so the caller's error names what the engine asked for.
        func resolveName(_ name: String) -> String {
            QWCheckpointTensorLoader.resolveName(
                name,
                defaultPrefix: QWEngineDriver.P,
                tieWordEmbeddings: geometry.descriptor.tieWordEmbeddings,
                rawExists: rawExists
            )
        }

        /// True when the pack carries `name` (under any accepted prefix).
        func hasTensor(_ name: String) -> Bool {
            rawExists(resolveName(name))
        }

        var isMLXCheckpoint: Bool {
            st?.isMLXCheckpoint == true
        }

        mutating func allocateWeight(_ shape: QWTensorShape) throws -> QWTensorValue {
            try plan.check(shape, at: weightRequest)
            let tensor = try wsub.allocate(shape)
            weightRequest += 1
            return tensor
        }
        // ── W2: embed + lm_head bf16 → f16 (device gather, chunked upload) ──
        mutating func convertBF16(_ name: String) throws -> QWTensorValue {
            let name = resolveName(name)
            guard let pk, let io else {
                throw QWEngineDriverError.weightMissing(name, "convertBF16 requires a QWP pack")
            }
            let e = try pk.entry(name)
            guard e.rawOff != nil else {
                throw QWEngineDriverError.weightMissing(name, "not a bf16 raw plane")
            }
            let total = e.shape.reduce(1, *)
            let dst = try allocateWeight(QWTensorShape([total], .f16))
            // THE GRID LAW: threadgroups-per-dimension ≤ 65535 (Metal cap;
            // an over-cap dispatch silently no-ops — half a matrix would
            // upload as zeros). 65535·256 elems = 32 MB bf16 per chunk.
            let chunk = 65535 * 256
            let tmp = hl.device.makeBuffer(length: chunk * 2, options: .storageModeShared)!
            var done = 0
            while done < total {
                let n = min(chunk, total - done)
                try io.loadRaw(name, elemOffset: done, elems: n, into: tmp.contents())
                guard let (cb, enc) = hl.begin() else { throw QWEngineDriverError.state("cmdbuffer") }
                var start = UInt32(0), nn = UInt32(n)
                enc.setComputePipelineState(psoCvt)
                enc.setBuffer(tmp, offset: 0, index: 0)          // bf16 bytes (ushort)
                enc.setBuffer(dst.region.buffer,
                              offset: dst.region.base + dst.offset + done * 2, index: 1)
                enc.setBytes(&start, length: 4, index: 2)
                enc.setBytes(&nn, length: 4, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                hl.end(cb, enc); cb.waitUntilCompleted()
                // H3 FIX (load path): a failed conversion chunk would
                // silently upload zeros into embed/head — throw.
                if let e = cb.error { throw QWEngineDriverError.state("bf16→f16 cb: \(e.localizedDescription)") }
                done += n
            }
            return dst
        }

        func quantizeSafetensorsToHQ4(_ name: String, n: Int, k: Int,
                                      into dstBuffer: any MTLBuffer,
                                      dstByteOffset: Int) throws {
            let resolved = resolveName(name)
            guard let st, let staging = quantStaging else {
                throw QWEngineDriverError.weightMissing(name, "safetensors online quantizer not configured")
            }
            do {
                try QWHQ4Quantizer.quantizeSafeTensorsInto(
                    hal: hl,
                    st: st,
                    name: resolved,
                    srcRowOffset: 0,
                    rowCount: n,
                    k: k,
                    dstBuffer: dstBuffer,
                    dstByteOffset: dstByteOffset,
                    staging: staging,
                    quantPSO: psoQuantBF16
                )
            } catch let e as QWHQ4Quantizer.QuantizeError {
                throw QWEngineDriverError.weightMissing(name, e.description)
            }
        }

        func quantizeSafetensorsSwitchGateUp(
            gateName: String, upName: String, experts: Int, i: Int, h: Int,
            into dstBuffer: any MTLBuffer, dstByteOffset: Int
        ) throws {
            let gResolved = resolveName(gateName)
            let uResolved = resolveName(upName)
            guard let st, let staging = quantStaging else {
                throw QWEngineDriverError.weightMissing(gateName, "safetensors online quantizer not configured")
            }
            do {
                try QWHQ4Quantizer.quantizeSwitchGateUpInto(
                    hal: hl,
                    st: st,
                    gateName: gResolved,
                    upName: uResolved,
                    experts: experts,
                    inter: i,
                    hidden: h,
                    dstBuffer: dstBuffer,
                    dstByteOffset: dstByteOffset,
                    staging: staging,
                    quantPSO: psoQuantBF16
                )
            } catch let e as QWHQ4Quantizer.QuantizeError {
                throw QWEngineDriverError.weightMissing(gateName, e.description)
            }
        }

        // ── helpers: planes verbatim (W1); bf16→f16 / →f32 CPU (W4) ────────
        // The plane views themselves live in DriverWeightLoadingAQ.swift: the
        // family's container is sized and filled differently.

        func rawFloats(_ name: String, _ elems: Int) throws -> [Float] {
            let resolved = resolveName(name)
            if let pk {
                let src = try pk.rawBf16(resolved)
                guard src.count == elems else {
                    throw QWEngineDriverError.weightMissing(name, "shape \(src.count) ≠ \(elems)")
                }
                return src
            }
            guard let st else {
                throw QWEngineDriverError.weightMissing(name, "missing in safetensors")
            }
            do {
                let out = try st.readFloats(resolved, expectedElementCount: elems)
                guard out.count == elems else {
                    throw QWEngineDriverError.weightMissing(name, "shape \(out.count) ≠ \(elems)")
                }
                return out
            } catch let err as QWEngineDriverError {
                throw err
            } catch {
                throw QWEngineDriverError.weightMissing(name, "\(error)")
            }
        }

        /// A bf16 plane, verbatim. Private on purpose: a norm gamma must
        /// not come through here, so the engine cannot load one the wrong
        /// way. The norm law is `f16wNorm`, and the raw plane is for the
        /// weights that have no law (the GDN's conv and gated norm).
        mutating func f16w(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let src = try rawFloats(name, elems)
            let t = try allocateWeight(QWTensorShape([elems], .f16))
            var bits = [UInt16](repeating: 0, count: elems)
            for i in 0..<elems { bits[i] = QWQuant.f32tof16Bits(src[i]) }
            bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .copyMemory(from: raw.baseAddress!, byteCount: elems * 2) }
            return t
        }
        /// A norm gamma's bits, in the law the engine's `1 + w` norms read:
        /// the 27B's checkpoint stores `w = γ − 1` under that law, a standard
        /// HF checkpoint stores γ itself, so it becomes `γ − 1` here.
        func normGammaBits(_ name: String, _ elems: Int) throws -> [UInt16] {
            let src = try rawFloats(name, elems)
            let shift: Float = (geometry.normGammaIsPlusOne && !isMLXCheckpoint) ? 0.0 : 1.0
            return src.map { QWQuant.f32tof16Bits($0 - shift) }
        }

        /// A norm gamma, stored the way the engine's norm law reads it.
        mutating func f16wNorm(_ name: String, _ elems: Int) throws -> QWTensorValue {
            guard !geometry.normGammaIsPlusOne || isMLXCheckpoint else { return try f16w(name, elems) }
            let bits = try normGammaBits(name, elems)
            let t = try allocateWeight(QWTensorShape([elems], .f16))
            bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .copyMemory(from: raw.baseAddress!, byteCount: elems * 2) }
            return t
        }

        /// The model's final RMSNorm gamma. It is a `1 + w` norm gamma like
        /// every other, so it is converted like every other.
        mutating func finalNorm(_ elems: Int) throws -> QWTensorValue {
            try f16wNorm(QWEngineDriver.P + "norm.weight", elems)
        }
        mutating func f32w(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let src = try rawFloats(name, elems)
            let t = try allocateWeight(QWTensorShape([elems], .f32))
            src.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .copyMemory(from: raw.baseAddress!, byteCount: elems * 4) }
            return t
        }
        mutating func f32wOrDequant(_ name: String, _ elems: Int) throws -> QWTensorValue {
            let resolved = resolveName(name)
            let src: [Float]
            if let pk {
                let e = try pk.entry(resolved)
                if e.rawOff != nil {
                    src = try pk.rawBf16(resolved)
                } else {
                    src = try pk.dequantQ4g(resolved)
                }
                guard src.count == elems else {
                    throw QWEngineDriverError.weightMissing(name, "shape \(src.count) ≠ \(elems)")
                }
            } else {
                src = try rawFloats(name, elems)
            }
            let t = try allocateWeight(QWTensorShape([elems], .f32))
            src.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in t.region.buffer.contents()
                .advanced(by: t.region.base + t.offset)
                .copyMemory(from: raw.baseAddress!, byteCount: elems * 4) }
            return t
        }
        // ── W3: plane concatenation helper (pure plumbing, zero math change) ──
        func resolvedPlaneLayout(_ firstName: String?, k: Int) -> (bits: Int, group: Int, incumbent: Bool, shvq: Bool, rowBytes: Int) {
            if let firstName, let pk, let entry = try? pk.entry(resolveName(firstName)), entry.hq4Off != nil {
                return (entry.hqBits, entry.hqGroup, entry.isIncumbentHQ4, entry.isSHVQ,
                        (k / entry.hqGroup) * entry.hqBlockBytes)
            }
            if let firstName, firstName.contains(".experts.") && !firstName.contains("shared_expert") {
                if firstName.contains("down_proj") {
                    return (format.expertDownBits, format.expertDownGroupSize,
                            format.expertDownIsIncumbentContainer, format.expertDownIsSHVQ,
                            format.expertDownRowBytes(k: k))
                }
                return (format.expertBits, format.expertGroupSize,
                        format.expertIsIncumbentContainer, format.expertIsSHVQ,
                        format.expertRowBytes(k: k))
            }
            return (format.bits, format.groupSize, format.denseIsIncumbentContainer,
                    format.denseIsSHVQ, format.rowBytes(k: k))
        }

        mutating func concatPlane(_ tensors: [(name: String, n: Int)], k: Int,
                         key: KeyPath<QWQTensorEntry, Int?>, extraZeroRows: Int = 0) throws -> QWTensorValue {
            if format.isNativeAffine {
                return try aqConcatPlane(tensors, k: k)
            }
            let rowBytes = resolvedPlaneLayout(tensors.first?.name, k: k).rowBytes
            let totalRows = tensors.reduce(0) { $0 + $1.n } + extraZeroRows
            let totalElems = (totalRows * rowBytes) / 2
            let t = try allocateWeight(QWTensorShape([totalElems], .f16))
            let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            var curByteOffset = 0
            if let pk, let io {
                var batch: [(name: String, dst: UnsafeMutableRawPointer)] = []
                batch.reserveCapacity(tensors.count)
                for (name, n) in tensors {
                    let resolved = resolveName(name)
                    let entry = try pk.entry(resolved)
                    guard entry.rawOff == nil, entry[keyPath: key] != nil else {
                        throw QWEngineDriverError.weightMissing(name, "needs hq4_off plane")
                    }
                    batch.append((name: resolved, dst: p.advanced(by: curByteOffset)))
                    curByteOffset += n * rowBytes
                }
                try io.loadBatch(batch, layout: .hq4)
            } else {
                for (name, n) in tensors {
                    let resolved = resolveName(name)
                    try quantizeSafetensorsToHQ4(
                        resolved, n: n, k: k,
                        into: t.region.buffer,
                        dstByteOffset: t.region.base + t.offset + curByteOffset
                    )
                    curByteOffset += n * rowBytes
                }
            }
            if extraZeroRows > 0 {
                memset(p.advanced(by: curByteOffset), 0, extraZeroRows * rowBytes)
            }
            return t
        }
        mutating func concatQ4(_ tensors: [(name: String, n: Int)], k: Int,
                               extraZeroRows: Int = 0) throws -> (QWWeightView, [QWWeightView]) {
            let layout = resolvedPlaneLayout(tensors.first?.name, k: k)
            let rowBytes = layout.rowBytes
            let totalN = tensors.reduce(0) { $0 + $1.n } + extraZeroRows
            let blocks = try concatPlane(tensors, k: k, key: \QWQTensorEntry.hq4Off, extraZeroRows: extraZeroRows)
            guard let bigView = QWWeightView(blocks: blocks, n: totalN, k: k,
                                                 bits: layout.bits, groupSize: layout.group,
                                                 isRotated: hl.weightBasis.rotatesActivations,
                                                 isIncumbentContainer: layout.incumbent,
                                                 isSHVQ: layout.shvq) else {
                throw QWEngineDriverError.weightMissing(tensors.first?.name ?? "concat", "concat view invalid")
            }
            var subViews: [QWWeightView] = []
            var row0 = 0
            for (name, n) in tensors {
                let subBlocks = QWTensorValue(
                    region: blocks.region,
                    offset: blocks.offset + row0 * rowBytes,
                    shape: QWTensorShape([n * rowBytes / 2], .f16)
                )
                guard let sv = bigView.subview(blocks: subBlocks, n: n, k: k) else {
                    throw QWEngineDriverError.weightMissing(name, "subview invalid")
                }
                subViews.append(sv)
                row0 += n
            }
            return (bigView, subViews)
        }
        mutating func packedGU(_ blk: String) throws -> QWMLPWeight {
            let gate = blk + "mlp.gate_proj.weight", up = blk + "mlp.up_proj.weight"
            let (bigView, _) = try concatQ4([(gate, geometry.intermediate), (up, geometry.intermediate)], k: geometry.hidden)
            guard let v = QWMLPWeight(blocks: bigView.blocks, n: 2 * geometry.intermediate, k: geometry.hidden,
                                    bits: bigView.bits, groupSize: bigView.groupSize,
                                    isRotated: bigView.isRotated,
                                    isIncumbentContainer: bigView.isIncumbentContainer,
                                    isSHVQ: bigView.isSHVQ) else {
                throw QWEngineDriverError.weightMissing(gate, "gu view invalid")
            }
            return v
        }

    }
}
