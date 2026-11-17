// QWResponsesStore.swift — Thread-safe bounded LRU store for /v1/responses state,
// supporting previous_response_id chaining, background status updates, GET, and DELETE.

import Foundation

public final class QWResponsesStore: @unchecked Sendable {
    public struct StoredResponse: Sendable {
        public var resource: OpenAIResponseResource
        public var historyMessages: [ChatMessage]
        public var task: Task<Void, Never>?

        public init(
            resource: OpenAIResponseResource,
            historyMessages: [ChatMessage],
            task: Task<Void, Never>? = nil
        ) {
            self.resource = resource
            self.historyMessages = historyMessages
            self.task = task
        }
    }

    private let lock = NSLock()
    private let capacity: Int
    private var entries: [String: StoredResponse] = [:]
    private var lruOrder: [String] = []

    public init(capacity: Int = 256) {
        self.capacity = max(1, capacity)
    }

    public func put(_ entry: StoredResponse) {
        let id = entry.resource.id
        var evictedTasks: [Task<Void, Never>] = []
        lock.withLock {
            entries[id] = entry
            if let idx = lruOrder.firstIndex(of: id) {
                lruOrder.remove(at: idx)
            }
            lruOrder.append(id)
            while lruOrder.count > capacity {
                let oldest = lruOrder.removeFirst()
                if let removed = entries.removeValue(forKey: oldest), let t = removed.task {
                    evictedTasks.append(t)
                }
            }
        }
        for t in evictedTasks { t.cancel() }
    }

    public func get(_ id: String) -> StoredResponse? {
        lock.withLock {
            guard let existing = entries[id] else { return nil }
            if let idx = lruOrder.firstIndex(of: id) {
                lruOrder.remove(at: idx)
                lruOrder.append(id)
            }
            return existing
        }
    }

    public func update(id: String, resource: OpenAIResponseResource, historyMessages: [ChatMessage]? = nil) {
        lock.withLock {
            guard var existing = entries[id] else { return }
            existing.resource = resource
            if let historyMessages {
                existing.historyMessages = historyMessages
            }
            existing.task = nil
            entries[id] = existing
            if let idx = lruOrder.firstIndex(of: id) {
                lruOrder.remove(at: idx)
                lruOrder.append(id)
            }
        }
    }

    public func attachTask(id: String, task: Task<Void, Never>) {
        let shouldCancel: Bool = lock.withLock {
            guard var existing = entries[id] else { return true }
            existing.task = task
            entries[id] = existing
            return false
        }
        if shouldCancel {
            task.cancel()
        }
    }

    @discardableResult
    public func delete(_ id: String) -> Bool {
        let outcome: (removed: Bool, task: Task<Void, Never>?) = lock.withLock {
            guard let removed = entries.removeValue(forKey: id) else { return (false, nil) }
            if let idx = lruOrder.firstIndex(of: id) {
                lruOrder.remove(at: idx)
            }
            return (true, removed.task)
        }
        outcome.task?.cancel()
        return outcome.removed
    }

    public var count: Int {
        lock.withLock { entries.count }
    }
}
