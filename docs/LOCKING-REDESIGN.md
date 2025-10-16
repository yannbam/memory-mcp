# Locking System Redesign: True Reader-Writer Locks

**Date**: 2025-10-16
**Author**: Claude (Session: pr-review-1c890f75 + continuation)
**Status**: Architecture Design Phase

## Executive Summary

This document outlines the redesign of memory-mcp's locking system to fix three critical concurrency issues found during PR review:

1. **Read locks are actually exclusive** (performance bug) → Migrate to true RW locks
2. **Rename doesn't lock destination** (race condition) → Multi-path atomic locking
3. **exists() swallows all errors** (correctness bug) → Proper error handling

## Critical Issues Analysis

### Issue 1: Fake Shared Read Locks

**Current Code** (`src/memory/locking.ts:95-109`):
```typescript
export async function acquireReadLock(filePath: string): Promise<LockRelease> {
  // Comment claims: "Multiple readers can hold the lock simultaneously"
  // Reality: Uses lockfile.lock() which is exclusive
  return await lockfile.lock(fileToLock, {
    // ... exclusive lock configuration
  });
}
```

**Problem**: `proper-lockfile` doesn't support shared locks. All reads serialize unnecessarily.

**Impact**: Multiple Claude instances viewing `/memories` will wait for each other instead of reading concurrently. Severe performance degradation under concurrent access.

**Root Cause**: `proper-lockfile` is a file-based mutex library, not a reader-writer lock library.

---

### Issue 2: Rename Race Condition

**Current Code** (`src/memory/operations.ts:410-475`):
```typescript
export async function rename(command: RenameCommand, context: OperationsContext): Promise<string> {
  // Validate and convert both paths
  const sourceFullPath = validateAndConvertPath(command.old_path, context.memoryRoot);
  const destFullPath = validateAndConvertPath(command.new_path, context.memoryRoot);

  // Execute with write lock on source path (no concurrency check needed for rename)
  return withWriteLock(sourceFullPath, false, async () => {
    // ... rename logic that touches BOTH source and destination
  });
}
```

**Problem**: Only locks source path. Destination is unlocked during the operation.

**Race Condition Example**:
```
Time  | Process A                      | Process B
------|--------------------------------|----------------------------
T1    | rename("/a.txt", "/b.txt")     |
T2    | Lock("/a.txt") ✓              |
T3    | Check source exists            |
T4    |                                | create("/b.txt", "data")
T5    |                                | Lock("/b.txt") ✓
T6    | fs.link("/a.txt", "/b.txt")   |
      | → EEXIST error (dest exists)   |
T7    |                                | Write to /b.txt
T8    |                                | Unlock("/b.txt")
```

**Impact**:
- Rename can fail spuriously when concurrent create races
- Multiple renames to same destination can cause data corruption
- Unpredictable behavior under concurrent access

---

### Issue 3: exists() Swallows Errors

**Current Code** (`src/memory/operations.ts:69-76`):
```typescript
async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false; // Swallows ALL errors
  }
}
```

**Problem**: Permission denied (EACCES) → returns `false` → "file not found" error message

**Impact**: Misleading error messages make debugging permission issues nearly impossible.

**Correct Behavior**: Only catch ENOENT, rethrow permission and other filesystem errors.

---

## Solution Architecture

### Library Choice: @esfx/async-readerwriterlock

**Selected**: `@esfx/async-readerwriterlock` v1.0.0

**Rationale**:
- ✅ Purpose-built for reader-writer locks (not a general mutex)
- ✅ True shared read locks - multiple concurrent readers
- ✅ Exclusive write locks
- ✅ Upgradeable read locks (read → write atomically)
- ✅ TypeScript-first design
- ✅ Actively maintained (Microsoft TypeScript team member)
- ✅ Apache-2.0 license, 234 GitHub stars

**API Overview**:
```typescript
import { AsyncReaderWriterLock } from '@esfx/async-readerwriterlock';

const rwlock = new AsyncReaderWriterLock();

// Shared read lock (multiple concurrent readers)
const readLock = await rwlock.read();
try {
  // ... read operation
} finally {
  readLock.unlock();
}

// Exclusive write lock
const writeLock = await rwlock.write();
try {
  // ... write operation
} finally {
  writeLock.unlock();
}

// Upgradeable read lock (atomic promotion to write)
const upgradeLock = await rwlock.upgradeableRead();
try {
  // ... read to decide if write needed
  if (needsWrite) {
    const writeLock = await upgradeLock.upgrade();
    try {
      // ... write operation
    } finally {
      writeLock.unlock();
    }
  }
} finally {
  upgradeLock.unlock();
}
```

---

## Detailed Design

### 1. LockManager Class

**Purpose**: Manage a pool of AsyncReaderWriterLock instances, one per path.

**Why per-path locks?**
File locking operates at path granularity. Each unique path needs its own RW lock so operations on different paths don't block each other unnecessarily.

**Design**:

```typescript
/**
 * LockManager manages a pool of reader-writer locks, one per path.
 *
 * Provides coordinated access to filesystem paths with:
 * - Shared read locks (multiple concurrent readers)
 * - Exclusive write locks
 * - Multi-path atomic locking (for rename operations)
 *
 * Lock instances are cached and reused for the same path.
 * Lock cleanup happens automatically when no operations are pending.
 */
export class LockManager {
  // Map: normalized path → RW lock instance
  private locks: Map<string, AsyncReaderWriterLock>;

  // Map: normalized path → reference count (for cleanup)
  private refCounts: Map<string, number>;

  constructor() {
    this.locks = new Map();
    this.refCounts = new Map();
  }

  /**
   * Get or create lock for a path
   * Increments reference count
   */
  private getLock(path: string): AsyncReaderWriterLock {
    // Normalize path to canonical form (resolve symlinks, . and ..)
    const normalizedPath = path.resolve(path);

    // Get existing lock or create new one
    let lock = this.locks.get(normalizedPath);
    if (!lock) {
      lock = new AsyncReaderWriterLock();
      this.locks.set(normalizedPath, lock);
      this.refCounts.set(normalizedPath, 0);
    }

    // Increment reference count
    const refCount = this.refCounts.get(normalizedPath)!;
    this.refCounts.set(normalizedPath, refCount + 1);

    return lock;
  }

  /**
   * Release lock for a path
   * Decrements reference count, cleans up if zero
   */
  private releaseLock(path: string): void {
    const normalizedPath = path.resolve(path);

    // Decrement reference count
    const refCount = this.refCounts.get(normalizedPath);
    if (refCount === undefined) {
      throw new Error(`Release called on non-existent lock: ${normalizedPath}`);
    }

    const newRefCount = refCount - 1;
    this.refCounts.set(normalizedPath, newRefCount);

    // Clean up if no more references
    if (newRefCount === 0) {
      this.locks.delete(normalizedPath);
      this.refCounts.delete(normalizedPath);
    }
  }

  /**
   * Acquire read lock for a path
   * Returns unlock function
   */
  async acquireReadLock(path: string): Promise<LockRelease> {
    const lock = this.getLock(path);
    const lockHandle = await lock.read();

    return async () => {
      lockHandle.unlock();
      this.releaseLock(path);
    };
  }

  /**
   * Acquire write lock for a path
   * Returns unlock function and captured mtime (if requested)
   */
  async acquireWriteLock(
    path: string,
    captureMtime: boolean = false
  ): Promise<LockResult> {
    // Capture mtime BEFORE acquiring lock (for optimistic concurrency)
    let mtimeBefore: number | null = null;
    if (captureMtime) {
      try {
        const stats = await fs.stat(path);
        mtimeBefore = stats.mtimeMs;
      } catch (err: any) {
        if (err.code !== 'ENOENT') throw err;
        // File doesn't exist yet - that's ok for create operations
      }
    }

    const lock = this.getLock(path);
    const lockHandle = await lock.write();

    const release = async () => {
      lockHandle.unlock();
      this.releaseLock(path);
    };

    return { release, mtimeBefore };
  }

  /**
   * Acquire write locks for multiple paths atomically
   *
   * Acquires locks in sorted order to prevent deadlock:
   * - Process A: lock ["/a", "/b"]
   * - Process B: lock ["/b", "/a"]
   * - Both processes sort paths → both try ["/a", "/b"]
   * - No circular wait → no deadlock
   *
   * Returns unlock function that releases all locks
   */
  async acquireMultipleWriteLocks(paths: string[]): Promise<LockRelease> {
    // Deduplicate and sort paths to prevent deadlock
    const uniquePaths = [...new Set(paths)];
    const sortedPaths = uniquePaths.sort();

    // Acquire locks in order
    const lockHandles: Array<{ unlock: () => void }> = [];
    for (const path of sortedPaths) {
      const lock = this.getLock(path);
      const lockHandle = await lock.write();
      lockHandles.push(lockHandle);
    }

    // Return release function that unlocks in reverse order
    return async () => {
      for (let i = lockHandles.length - 1; i >= 0; i--) {
        lockHandles[i].unlock();
      }
      for (const path of sortedPaths) {
        this.releaseLock(path);
      }
    };
  }
}
```

---

### 2. High-Level API Functions

**Keep existing API patterns** (`withReadLock`, `withWriteLock`) but implement with LockManager:

```typescript
// Singleton lock manager instance
const lockManager = new LockManager();

/**
 * Execute an operation with read lock
 * Multiple concurrent readers allowed
 */
export async function withReadLock<T>(
  filePath: string,
  operation: () => Promise<T>
): Promise<T> {
  // Determine what to lock
  const pathToLock = await determinePathToLock(filePath);

  // Acquire read lock
  const release = await lockManager.acquireReadLock(pathToLock);

  try {
    // Execute operation while holding lock
    return await operation();
  } finally {
    // Always release lock
    await release();
  }
}

/**
 * Execute an operation with write lock and optimistic concurrency control
 */
export async function withWriteLock<T>(
  filePath: string,
  checkConcurrency: boolean,
  operation: () => Promise<T>
): Promise<T> {
  // Determine what to lock
  const pathToLock = await determinePathToLock(filePath);

  // Acquire lock and capture mtime if checking concurrency
  const { release, mtimeBefore } = await lockManager.acquireWriteLock(
    pathToLock,
    checkConcurrency
  );

  try {
    // Check if file was modified while waiting for lock
    if (checkConcurrency && mtimeBefore !== null) {
      const wasModified = await wasFileModified(filePath, mtimeBefore);
      if (wasModified) {
        throw new Error(
          `File was modified concurrently while waiting for lock: ${filePath}`
        );
      }
    }

    // Execute operation while holding lock
    return await operation();
  } finally {
    // Always release lock
    await release();
  }
}

/**
 * Execute an operation with write locks on multiple paths atomically
 * NEW FUNCTION for rename operations
 */
export async function withMultipleWriteLocks<T>(
  filePaths: string[],
  operation: () => Promise<T>
): Promise<T> {
  // Determine what to lock for each path
  const pathsToLock = await Promise.all(
    filePaths.map(fp => determinePathToLock(fp))
  );

  // Acquire locks atomically (sorted order prevents deadlock)
  const release = await lockManager.acquireMultipleWriteLocks(pathsToLock);

  try {
    // Execute operation while holding all locks
    return await operation();
  } finally {
    // Always release all locks
    await release();
  }
}

/**
 * Determine which path to lock
 * If file doesn't exist, lock parent directory instead
 */
async function determinePathToLock(filePath: string): Promise<string> {
  try {
    await fs.access(filePath);
    return filePath; // File exists, lock it directly
  } catch (err: any) {
    if (err.code === 'ENOENT') {
      // File doesn't exist, lock parent directory
      return path.dirname(filePath);
    }
    throw err; // Other errors (permission denied, etc.)
  }
}
```

---

### 3. Update rename() Operation

**Current** (locks only source):
```typescript
export async function rename(command: RenameCommand, context: OperationsContext): Promise<string> {
  const sourceFullPath = validateAndConvertPath(command.old_path, context.memoryRoot);
  const destFullPath = validateAndConvertPath(command.new_path, context.memoryRoot);

  return withWriteLock(sourceFullPath, false, async () => {
    // ... rename logic
  });
}
```

**New** (locks both source and destination atomically):
```typescript
export async function rename(command: RenameCommand, context: OperationsContext): Promise<string> {
  const sourceFullPath = validateAndConvertPath(command.old_path, context.memoryRoot);
  const destFullPath = validateAndConvertPath(command.new_path, context.memoryRoot);

  // Lock both paths atomically (prevents race conditions)
  return withMultipleWriteLocks([sourceFullPath, destFullPath], async () => {
    // ... rename logic
    // Both paths are locked, safe to proceed
  });
}
```

---

### 4. Fix exists() Helper

**Current** (swallows all errors):
```typescript
async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false; // ❌ Swallows permission errors
  }
}
```

**New** (proper error handling):
```typescript
async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch (err: any) {
    // File not found - expected case
    if (err.code === 'ENOENT') {
      return false;
    }

    // Permission denied - helpful error message
    if (err.code === 'EACCES') {
      throw new Error(
        `Permission denied accessing path: ${filePath}\n` +
        `Check filesystem permissions for the memory-mcp process.`
      );
    }

    // Other filesystem errors - rethrow with context
    throw new Error(
      `Filesystem error checking path: ${filePath}\n` +
      `Error code: ${err.code}\n` +
      `Message: ${err.message}`
    );
  }
}
```

---

## Migration Strategy

### Phase 1: Install Dependencies
```bash
npm install @esfx/async-readerwriterlock
```

### Phase 2: Rewrite locking.ts
1. Create LockManager class
2. Implement acquireReadLock, acquireWriteLock, acquireMultipleWriteLocks
3. Reimplement withReadLock, withWriteLock
4. Add new withMultipleWriteLocks function
5. Keep determinePathToLock and wasFileModified helpers

### Phase 3: Update operations.ts
1. Update rename() to use withMultipleWriteLocks
2. Fix exists() error handling
3. No changes needed to other operations (API stays the same)

### Phase 4: Update Tests
1. Update locking tests for new RW lock behavior
2. Add tests for concurrent reads
3. Add tests for multi-path locking
4. Verify all 85 tests pass

---

## Testing Strategy

### Unit Tests
- ✅ Existing tests should mostly pass unchanged (same high-level API)
- ✅ Add tests for concurrent reads (verify they don't block each other)
- ✅ Add tests for multi-path locking (verify atomic behavior)
- ✅ Update any tests that assume exclusive reads

### Integration Tests
- ☞ Spawn multiple processes calling view() concurrently
- ☞ Verify read throughput improves vs. current implementation
- ☞ Test rename() race conditions are prevented

### Manual Testing
- ☞ Connect multiple Claude Code instances
- ☞ Have them view /memories concurrently
- ☞ Verify they don't serialize

---

## Benefits

### Performance
- **Concurrent reads**: Multiple Claude instances can view files simultaneously
- **No serialization**: Read operations scale linearly with number of readers
- **Same write performance**: Write locks still exclusive (no change)

### Correctness
- **No rename races**: Both source and destination locked atomically
- **Clear error messages**: Permission errors surface properly
- **Proper semantics**: Read locks are actually shared, as documented

### Maintainability
- **Clear abstraction**: LockManager encapsulates lock lifecycle
- **Type-safe**: TypeScript RW lock library
- **Well-tested**: @esfx library is production-ready

---

## Risks and Mitigations

### Risk 1: Lock Map Grows Unbounded
**Mitigation**: Reference counting with automatic cleanup when refCount reaches zero.

### Risk 2: Deadlock with Multi-Path Locking
**Mitigation**: Always acquire locks in sorted path order (prevents circular wait).

### Risk 3: Breaking Changes to API
**Mitigation**: Keep existing withReadLock/withWriteLock API, only add new withMultipleWriteLocks.

---

## Success Criteria

✅ All 85 tests pass
✅ Concurrent reads don't block each other (manual test)
✅ Rename locks both source and destination
✅ exists() only catches ENOENT, rethrows other errors
✅ No performance regression on single-threaded operations

---

## Open Questions

1. **Lock timeouts**: Should we add timeout support to prevent deadlock? Current `proper-lockfile` has stale timeout (30s).
   - **Answer**: @esfx supports cancellation via CancelableTokens. Defer for now, add if needed.

2. **Lock upgrades**: Should we use upgradeable read locks for operations that might need to write?
   - **Answer**: Not needed for current use cases. All operations know upfront if they need read or write.

3. **Lock statistics**: Should we expose metrics (lock wait time, contention, etc.)?
   - **Answer**: Defer. Add debug logging first, metrics later if needed.

---

## Implementation Checklist

- [ ] Install @esfx/async-readerwriterlock
- [ ] Create LockManager class
- [ ] Implement acquireReadLock
- [ ] Implement acquireWriteLock
- [ ] Implement acquireMultipleWriteLocks
- [ ] Implement withReadLock
- [ ] Implement withWriteLock
- [ ] Implement withMultipleWriteLocks
- [ ] Update rename() to use withMultipleWriteLocks
- [ ] Fix exists() error handling
- [ ] Update tests for RW lock behavior
- [ ] Run all 85 tests
- [ ] Manual test concurrent reads
- [ ] Update ARCHITECTURE.md
- [ ] Update CLAUDE.md session handoff

---

**End of Design Document**
