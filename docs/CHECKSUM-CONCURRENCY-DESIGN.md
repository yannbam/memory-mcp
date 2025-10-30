# Checksum-Based Concurrency Detection

**Date**: 2025-10-30
**Author**: Claude (Session: 76621b6d-1a88-426b-972b-1a9363d47d53)
**Status**: Design Phase → Implementation Ready
**Branch**: `feature/checksum-concurrency-detection`

## Executive Summary

This document outlines the design and implementation strategy for replacing mtime-based concurrency detection with content checksums. This change solves a critical limitation: detecting file modifications that occur **between separate operations** (sequential modifications), not just during lock acquisition (concurrent modifications).

**Key improvement**: Detects when Claude reads a file, another process modifies it minutes later, then Claude tries to write based on stale understanding.

---

## Problem Statement

### Current Limitation: Mtime Only Detects Concurrent Modifications

**Current mtime-based detection window:**
```
Operation Timeline:
├─ T0: Capture mtime              ← mtimeBefore
│
├─ T1: Wait for lock...           ← Detection window (milliseconds to seconds)
│
├─ T2: Lock acquired
├─ T3: Check mtime                ← If different → ERROR
│
└─ T4: Proceed with operation
```

**What mtime DOES detect:**
- File modified by another process **while waiting for the lock**
- Timeframe: milliseconds to seconds

**What mtime DOES NOT detect:**
- File modified **before starting this operation**
- File modified **between a previous read and this write**

### Real-World Failure Scenario

```
Time   | Claude Session A              | Claude Session B
-------|-------------------------------|------------------
10:00  | view("/memories/notes.txt")   |
       | Returns: "TODO: Buy milk"     |
       | (No mtime cached)             |
       |                               |
10:05  |                               | str_replace(..., "milk", "eggs")
       |                               | File now: "TODO: Buy eggs"
       |                               |
10:10  | str_replace(..., "milk", "bread")
       | → Captures mtime: [current]   |
       | → Checks: [current] === [current] ✓
       | → Searches for "milk": NOT FOUND
       | → Error: "Text not found"     |
```

**Current behavior**: Claude gets a confusing "text not found" error, not realizing the file was modified by another process.

**Desired behavior**: Claude gets explicit notification that file was modified, with current contents shown.

---

## Solution: Content Checksum Caching

### Core Concept

**Store what we last saw, compare against current disk state:**

1. **After every read or write**: Compute SHA-256 checksum of file content → Store in RAM
2. **Before every write**:
   - Read current file from disk
   - Compute its checksum
   - Compare with cached checksum
   - If mismatch → File was modified → Error with current contents
   - If match → Proceed with operation

### Why This Works Across Separate Processes

**Key insight**: Each stdio MCP server process has its own memory space, but all use the **same filesystem** as source of truth.

```
┌─────────────────────┐         ┌─────────────────────┐
│ Server Process A    │         │ Server Process B    │
│ (stdio transport)   │         │ (stdio transport)   │
├─────────────────────┤         ├─────────────────────┤
│ Checksum Cache:     │         │ Checksum Cache:     │
│ notes.txt → abc123  │         │ notes.txt → xyz789  │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           └───────────┬───────────────────┘
                       ▼
              ┌────────────────┐
              │   Filesystem   │
              │  (Disk State)  │
              │ notes.txt:     │
              │ "actual data"  │
              └────────────────┘
```

**Detection mechanism:**
- Process A caches: `notes.txt → checksum of "old data"`
- Process B modifies file on disk
- Process A before next write:
  - Reads current disk state: "new data"
  - Computes checksum: different from cached!
  - Detects modification even though B is separate process

---

## Implementation Strategy

### 1. New Module: Checksum Utilities

**File**: `src/memory/checksums.ts`

```typescript
import crypto from 'crypto';
import * as path from 'path';

/**
 * In-memory checksum cache
 * Key: Normalized absolute file path
 * Value: SHA-256 hex digest (64 chars)
 */
const checksumCache = new Map<string, string>();

/**
 * Compute SHA-256 checksum of file content
 * @param content - File content as string
 * @returns Hex digest (64 characters)
 */
export function computeChecksum(content: string): string {
  return crypto.createHash('sha256').update(content, 'utf-8').digest('hex');
}

/**
 * Get cached checksum for a file path
 * @param filePath - Absolute file path
 * @returns Cached checksum or undefined if not cached
 */
export function getCachedChecksum(filePath: string): string | undefined {
  return checksumCache.get(path.resolve(filePath));
}

/**
 * Store checksum in cache
 * @param filePath - Absolute file path
 * @param checksum - SHA-256 hex digest
 */
export function setCachedChecksum(filePath: string, checksum: string): void {
  checksumCache.set(path.resolve(filePath), checksum);
}

/**
 * Remove checksum from cache
 * @param filePath - Absolute file path
 */
export function clearCachedChecksum(filePath: string): void {
  checksumCache.delete(path.resolve(filePath));
}

/**
 * Clear entire cache (for testing)
 */
export function clearAllCachedChecksums(): void {
  checksumCache.clear();
}

/**
 * Get cache statistics (for debugging)
 */
export function getChecksumCacheStats() {
  return {
    size: checksumCache.size,
    memoryEstimate: checksumCache.size * 82, // ~82 bytes per entry
  };
}
```

**Exports**: All functions above

---

### 2. Modified: locking.ts

**Replace mtime-based detection with checksum-based detection in `withWriteLock()`:**

```typescript
export async function withWriteLock<T>(
  filePath: string,
  checkConcurrency: boolean,
  operation: () => Promise<T>,
): Promise<T> {
  // For operations that don't need concurrency checking (e.g., create)
  if (!checkConcurrency) {
    const pathToLock = await determinePathToLock(filePath);
    const { release } = await lockManager.acquireWriteLock(pathToLock, false);
    try {
      return await operation();
    } finally {
      await release();
    }
  }

  // Read file and compute checksum BEFORE lock acquisition
  let contentBefore: string;
  let checksumBefore: string;

  try {
    contentBefore = await fs.readFile(filePath, 'utf-8');
    checksumBefore = computeChecksum(contentBefore);
  } catch (err) {
    const fsError = err as { code?: string };
    if (fsError.code === 'ENOENT') {
      // File doesn't exist - can't check concurrency
      // This should be rare (operations check exists() first)
      const pathToLock = await determinePathToLock(filePath);
      const { release } = await lockManager.acquireWriteLock(pathToLock, false);
      try {
        return await operation();
      } finally {
        await release();
      }
    }
    throw err; // Other errors (permission, etc.)
  }

  // Check against cached checksum (detects sequential modifications)
  const cachedChecksum = getCachedChecksum(filePath);
  if (cachedChecksum && cachedChecksum !== checksumBefore) {
    // File changed since last access by THIS server instance
    const preview = makeContentPreview(contentBefore, filePath);

    throw new Error(
      'File has been modified by another process.\n\n' +
      preview + '\n\n' +
      'Please review the current contents and retry if appropriate.'
    );
  }

  // Acquire exclusive write lock
  const pathToLock = await determinePathToLock(filePath);
  const { release } = await lockManager.acquireWriteLock(pathToLock, false);

  try {
    // Re-read and verify checksum (detects concurrent modifications during lock wait)
    const contentNow = await fs.readFile(filePath, 'utf-8');
    const checksumNow = computeChecksum(contentNow);

    if (checksumNow !== checksumBefore) {
      // File modified while waiting for lock
      const preview = makeContentPreview(contentNow, filePath);

      throw new Error(
        'File was modified while waiting for lock.\n\n' +
        preview + '\n\n' +
        'Please review the current contents and retry.'
      );
    }

    // Execute operation - file hasn't changed
    return await operation();
  } finally {
    await release();
  }
}

/**
 * Helper: Create content preview for error messages
 * Truncates large files with indicator
 */
function makeContentPreview(content: string, filePath: string): string {
  const maxLength = 5000;
  const lines = content.split('\n');

  let preview = `Current contents of ${filePath}:\n`;
  preview += '━'.repeat(60) + '\n';

  if (content.length <= maxLength) {
    preview += content;
  } else {
    preview += content.slice(0, maxLength);
    preview += '\n\n[... truncated, file is ' + content.length + ' bytes total]';
  }

  preview += '\n' + '━'.repeat(60);

  return preview;
}
```

**Remove**: All mtime-related code (mtimeBefore, wasFileModified function)

---

### 3. Modified: operations.ts

**Update all operations to cache checksums after reads and writes:**

#### **view() - Cache after reading**

```typescript
export async function view(command: ViewCommand, context: OperationsContext): Promise<string> {
  // ... existing code ...

  return withReadLock(fullPath, async () => {
    const stat = await fs.stat(fullPath);

    if (stat.isDirectory()) {
      return viewDirectory(fullPath, context);
    } else {
      const result = await viewFile(fullPath, command.view_range);

      // Cache checksum after reading file
      if (!command.view_range) {
        // Only cache if we read the entire file
        const content = await fs.readFile(fullPath, 'utf-8');
        setCachedChecksum(fullPath, computeChecksum(content));
      }

      return result;
    }
  });
}
```

**Note**: Only cache on full file reads, not partial (with view_range).

#### **create() - Cache after writing**

```typescript
export async function create(command: CreateCommand, context: OperationsContext): Promise<string> {
  // ... existing validation ...

  await withWriteLock(fullPath, false, async () => {
    // ... existing create logic ...
    const content = command.file_text ?? '';
    await fs.writeFile(fullPath, content, 'utf-8');

    // Cache checksum of newly created file
    setCachedChecksum(fullPath, computeChecksum(content));
  });

  // ... return message
}
```

#### **str_replace() - Cache after writing**

```typescript
export async function str_replace(command: StrReplaceCommand, context: OperationsContext): Promise<string> {
  // ... existing validation ...

  await withWriteLock(fullPath, true, async () => {
    // ... existing str_replace logic ...
    const newContent = content.replace(oldStr, newStr);
    await fs.writeFile(fullPath, newContent, 'utf-8');

    // Cache checksum of modified file
    setCachedChecksum(fullPath, computeChecksum(newContent));
  });

  return `File ${command.path} has been edited`;
}
```

**Same pattern for**:
- `insert()` - Cache after writing
- `deleteOp()` (line and text variants) - Cache after writing or clear if file deleted

#### **deleteOp() - Clear cache on file/directory deletion**

```typescript
export async function deleteOp(command: DeleteCommand, context: OperationsContext): Promise<string> {
  // ... existing validation ...

  await withWriteLock(fullPath, false, async () => {
    // ... existing delete logic ...

    if (stat.isFile()) {
      await fs.unlink(fullPath);
      clearCachedChecksum(fullPath); // Remove from cache
    } else {
      await fs.rm(fullPath, { recursive: true });
      clearCachedChecksum(fullPath); // Clear directory entry
      // Note: No need to clear files within - they're gone
    }
  });

  // ... return message
}
```

#### **rename() - Clear old path from cache**

```typescript
export async function rename(command: RenameCommand, context: OperationsContext): Promise<string> {
  // ... existing validation ...

  await withMultipleWriteLocks([sourceFullPath, destFullPath], async () => {
    // ... existing rename logic ...

    // Clear cache for old path (new path will be cached on next access)
    clearCachedChecksum(sourceFullPath);
  });

  // ... return message
}
```

---

### 4. Testing Strategy

#### **Unit Tests**

**New file**: `test/checksum-utilities.test.ts`
- Compute checksum consistency (same content → same checksum)
- Different content → different checksum
- Cache set/get/clear operations
- Cache statistics

**Modified file**: `test/locking.test.ts`
- Remove mtime-based tests
- Add checksum-based sequential modification tests
- Add checksum-based concurrent modification tests
- Verify error messages include content preview

**Modified file**: `test/memory-operations.test.ts`
- Add tests verifying checksums cached after operations
- Add tests for sequential modification detection (read → external modify → write)
- Verify cache cleared on delete/rename

#### **Integration Tests**

**New file**: `test/concurrent-access.test.ts`
- Spawn two separate server processes (stdio transport)
- Server A: view file
- Server B: modify file
- Server A: attempt to modify file
- Verify A receives "file modified" error with current contents

**Manual Testing**
- Two Claude Code sessions running in parallel
- Both connected to separate stdio MCP servers
- Session A: read memory file
- Session B: modify same file
- Session A: attempt to modify file
- Verify error message with current contents

---

## Performance Analysis

### Computational Cost

**Current (mtime-based):**
```
2 × fs.stat()      ~0.05ms each  = 0.1ms
1 × fs.readFile()  ~0.5ms        = 0.5ms
────────────────────────────────────────
Total: ~0.6ms per write operation
```

**Proposed (checksum-based):**
```
2 × fs.readFile()     ~0.5ms each   = 1.0ms
2 × SHA-256 hash      ~0.02ms each  = 0.04ms (10KB file)
────────────────────────────────────────
Total: ~1.04ms per write operation
```

**Overhead: +0.44ms** (~73% increase in absolute terms, negligible in practice)

### SHA-256 Hashing Performance

**Throughput**: ~500 MB/s on modern CPUs

| File Size | Hash Time |
|-----------|-----------|
| 1 KB | 0.002ms |
| 10 KB | 0.02ms |
| 50 KB | 0.1ms |
| 100 KB | 0.2ms |
| 1 MB | 2ms |

**Expected use case**: Human-readable notes (1-50 KB) → **negligible hashing cost**

### Memory Overhead

**Per cache entry:**
```
Key: "/home/user/.memory/memories/notes.txt"  ~50 bytes
Value: "a1b2c3d4e5f6..." (64 hex chars)       ~32 bytes
Map overhead:                                  ~20 bytes
──────────────────────────────────────────────────────
Total: ~102 bytes per cached file
```

**1000 files = ~102 KB of RAM**

**Verdict**: Trivial memory cost for the use case

---

## Edge Cases & Mitigations

### Large Files (>1 MB)

**Scenario**: User stores large file in /memories (unexpected use case)

**Current approach**: Hash it anyway
- 10 MB file = 20ms hashing time
- Still acceptable for rare operations

**Future optimization** (if needed):
```typescript
const MAX_CHECKSUM_SIZE = 1 * 1024 * 1024; // 1 MB

if (stat.size > MAX_CHECKSUM_SIZE) {
  // Skip checksumming for very large files
  // Fall back to no concurrency detection
  context.logger.warn('Large file - skipping checksum', { path, size: stat.size });
  return; // Don't cache
}
```

### Content Preview Truncation

**Issue**: Showing "full contents" in error might overwhelm terminal for large files

**Solution**: Implemented in `makeContentPreview()`:
- Truncate at 5000 characters
- Add indicator: `[... truncated, file is 12345 bytes total]`

### Cache Invalidation on Rename

**Scenario**: File renamed from A → B

**Approach**:
1. Clear cache entry for old path (A)
2. Don't create cache entry for new path (B)
3. Next operation on B will read and cache

**Why not copy cache entry?**
- Content might have changed during rename (race condition)
- Simpler to just re-read on next access
- Rename is infrequent operation

### Directory Operations

**Scenario**: view() on directory

**Approach**: Don't cache checksums for directories
```typescript
if (stat.isDirectory()) {
  return viewDirectory(...); // No caching
}
```

**Reason**: Directory content is list of entries, changes frequently, not worth caching

### Server Restart

**Behavior**: In-memory cache cleared automatically

**Implication**: First operation after restart won't detect modifications since previous session

**Acceptable?**: Yes - this is expected behavior for in-memory cache. Alternative would be persistent cache (file/DB), but adds significant complexity for minimal benefit.

---

## Migration Plan

### Phase 1: Implementation (This Session)
1. ✅ Create feature branch
2. ✅ Write design document
3. Create implementation plan (PlanAndTrack)
4. Implement checksum utilities
5. Modify locking.ts to use checksums
6. Update all operations to cache checksums
7. Update types/interfaces as needed

### Phase 2: Testing
1. Write unit tests for checksum utilities
2. Update locking tests (remove mtime, add checksum)
3. Update operations tests (verify caching)
4. Write integration tests (concurrent access)
5. Manual testing with two Claude Code sessions

### Phase 3: Documentation
1. Update ARCHITECTURE.md (concurrency section)
2. Update README.md (user-facing implications)
3. Update CHANGELOG.md
4. Update memory (handoff notes)

### Phase 4: Review & Merge
1. Code review
2. All tests passing
3. Performance verification
4. Merge to dev branch

---

## Success Criteria

✅ All existing tests pass (with modifications)
✅ New checksum tests pass
✅ Integration test: two processes, sequential modification detected
✅ Manual test: two Claude sessions, error message shows current content
✅ No performance regression for typical files (<50 KB)
✅ Documentation updated

---

## Backward Compatibility

**Breaking Changes**: None

**API Changes**: None (internal implementation only)

**Behavior Changes**:
- More accurate concurrency detection (catches sequential modifications)
- Better error messages (shows current content)
- Slightly slower write operations (+0.4ms, negligible)

**Migration**: None required (internal change)

---

## Future Enhancements (Out of Scope)

1. **Persistent cache** - Survive server restarts (complexity not justified)
2. **Size-based hashing strategy** - Different hash for large files (premature optimization)
3. **Cache eviction** - LRU cache with size limit (not needed for expected use case)
4. **Metrics** - Track cache hit/miss rates (defer until needed)

---

**End of Design Document**
