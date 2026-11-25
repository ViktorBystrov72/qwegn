// QwgReplVisionTests.swift — Regression tests for qwg repl --no-load-vision
// and /image slash command rejection when visionController == nil.
import ArgumentParser
@testable import qwg
import Testing

@Suite("qwg repl vision policy")
struct QwgReplVisionTests {
    @Test("Repl refuses --image with --no-load-vision during argument validation")
    func replRejectsImageWithNoLoadVision() throws {
        #expect(throws: (any Error).self) {
            _ = try Repl.parse(["--no-load-vision", "--image", "/tmp/sample.png", "dummy-model"])
        }
        #expect(throws: (any Error).self) {
            try Repl.validateImageOption(image: "/tmp/sample.png", allowVision: false)
        }
        #expect(throws: Never.self) {
            try Repl.validateImageOption(image: nil, allowVision: false)
            _ = try Repl.parse(["--no-load-vision", "dummy-model"])
            _ = try Repl.parse(["--image", "/tmp/sample.png", "dummy-model"])
        }
    }

    @Test("REPL /image command is rejected cleanly when visionController is nil")
    func replSlashImageRejectedWithoutVisionController() {
        let rejectedWithArg = Repl.resolveSlashImageCommand(
            arg: "/tmp/photo.png", hasVisionController: false, currentPending: nil
        )
        #expect(!rejectedWithArg.accepted)
        #expect(rejectedWithArg.nextPending == nil)
        #expect(rejectedWithArg.notice.contains("--no-load-vision"))

        let rejectedNoArg = Repl.resolveSlashImageCommand(
            arg: "", hasVisionController: false, currentPending: "/tmp/old.png"
        )
        #expect(!rejectedNoArg.accepted)
        #expect(rejectedNoArg.nextPending == nil)

        let accepted = Repl.resolveSlashImageCommand(
            arg: "/tmp/photo.png", hasVisionController: true, currentPending: nil
        )
        #expect(accepted.accepted)
        #expect(accepted.nextPending == "/tmp/photo.png")
    }

    @Test("Repl refuses --video with --no-load-vision and handles /video slash command")
    func replRejectsVideoWithNoLoadVisionAndHandlesSlashVideo() throws {
        #expect(throws: (any Error).self) {
            _ = try Repl.parse(["--no-load-vision", "--video", "/tmp/clip.mp4", "dummy-model"])
        }
        #expect(throws: (any Error).self) {
            try Repl.validateVideoOption(video: "/tmp/clip.mp4", allowVision: false)
        }
        #expect(throws: Never.self) {
            try Repl.validateVideoOption(video: nil, allowVision: false)
            _ = try Repl.parse(["--video", "/tmp/clip.mp4", "dummy-model"])
        }

        let rejected = Repl.resolveSlashVideoCommand(
            arg: "/tmp/clip.mp4", hasVisionController: false, currentPending: nil
        )
        #expect(!rejected.accepted)
        #expect(rejected.nextPending == nil)
        #expect(rejected.notice.contains("--no-load-vision"))

        let accepted = Repl.resolveSlashVideoCommand(
            arg: "/tmp/clip.mp4", hasVisionController: true, currentPending: nil
        )
        #expect(accepted.accepted)
        #expect(accepted.nextPending == "/tmp/clip.mp4")
    }
}
