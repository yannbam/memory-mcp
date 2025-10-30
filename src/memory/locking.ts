/**
 * File Locking with True Reader-Writer Locks
 *
 * Provides file locking for concurrent access from multiple Claude instances.
 * Uses @esfx/async-readerwriterlock for true shared read locks and exclusive write locks.
 * Implements optimistic concurrency control using content checksums (SHA-256).
 *
 * Key Features:
 * - Shared read locks: Multiple readers can access concurrently
 * - Exclusive write locks: Only one writer at a time
 * - Multi-path atomic locking: Lock multiple paths for rename operations
 * - Optimistic concurrency: Detect modifications using cached content checksums
 * - Sequential modification detection: Detects changes across separate operations
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { AsyncReaderWriterLock } from '@esfx/async-readerwriterlock';
import {
  computeChecksum,
  getCachedChecksum,
} from './checksums.js';
import { formatFileContent } from './formatting.js';

/**
 * Lock release function
 */
export type LockRelease = () => Promise<void>;

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
  private getLock(filePath: string): AsyncReaderWriterLock {
    // Normalize path to canonical form (resolve symlinks, . and ..)
    const normalizedPath = path.resolve(filePath);

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
  private releaseLock(filePath: string): void {
    // Normalize path
    const normalizedPath = path.resolve(filePath);

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
   * Multiple readers can hold locks simultaneously
   *
   * @param filePath - Absolute filesystem path to lock
   * @returns Lock release function
   */
  async acquireReadLock(filePath: string): Promise<LockRelease> {
    // Get lock instance for this path
    const lock = this.getLock(filePath);

    // Acquire shared read lock
    const lockHandle = await lock.read();

    // Return release function
    return (): Promise<void> => {
      lockHandle.unlock();
      this.releaseLock(filePath);
      return Promise.resolve();
    };
  }

  /**
   * Acquire write lock for a path
   * Exclusive - only one writer at a time
   *
   * @param filePath - Absolute filesystem path to lock
   * @returns Lock release function
   */
  async acquireWriteLock(filePath: string): Promise<LockRelease> {
    // Get lock instance for this path
    const lock = this.getLock(filePath);

    // Acquire exclusive write lock
    const lockHandle = await lock.write();

    // Return release function
    const release = (): Promise<void> => {
      lockHandle.unlock();
      this.releaseLock(filePath);
      return Promise.resolve();
    };

    return release;
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
   * @param filePaths - Array of absolute filesystem paths to lock
   * @returns Lock release function that releases all locks
   */
  async acquireMultipleWriteLocks(filePaths: string[]): Promise<LockRelease> {
    // Deduplicate and sort paths to prevent deadlock
    const uniquePaths = [...new Set(filePaths)];
    const sortedPaths = uniquePaths.map(p => path.resolve(p)).sort();

    // Acquire locks in order
    const lockHandles: Array<{ unlock: () => void }> = [];
    const acquiredPaths: string[] = [];

    try {
      for (const filePath of sortedPaths) {
        const lock = this.getLock(filePath);
        const lockHandle = await lock.write();
        lockHandles.push(lockHandle);
        acquiredPaths.push(filePath);
      }
    } catch (error) {
      // If acquisition fails, release all acquired locks
      for (let i = lockHandles.length - 1; i >= 0; i--) {
        lockHandles[i].unlock();
      }
      for (const filePath of acquiredPaths) {
        this.releaseLock(filePath);
      }
      throw error;
    }

    // Return release function that unlocks in reverse order
    return (): Promise<void> => {
      for (let i = lockHandles.length - 1; i >= 0; i--) {
        lockHandles[i].unlock();
      }
      for (const filePath of sortedPaths) {
        this.releaseLock(filePath);
      }
      return Promise.resolve();
    };
  }
}

// Singleton lock manager instance
const lockManager = new LockManager();

/**
 * Determine which path to lock
 * If file doesn't exist, lock parent directory instead
 *
 * @param filePath - Absolute filesystem path
 * @returns Path to lock (file if exists, parent directory otherwise)
 */
async function determinePathToLock(filePath: string): Promise<string> {
  try {
    await fs.access(filePath);
    return filePath; // File exists, lock it directly
  } catch (err) {
    const fsError = err as { code?: string };
    if (fsError.code === 'ENOENT') {
      // File doesn't exist, lock parent directory
      const parentDir = path.dirname(filePath);

      // Ensure parent directory exists
      try {
        await fs.mkdir(parentDir, { recursive: true });
      } catch {
        // Directory already exists or creation failed, continue anyway
      }

      return parentDir;
    }
    throw err; // Other errors (permission denied, etc.)
  }
}

/**
 * Helper: Create content preview for error messages
 * Uses same formatting as view command for consistency
 *
 * @param content - File content as string
 * @param filePath - File path for display
 * @returns Formatted preview with line numbers
 */
function makeContentPreview(content: string, filePath: string): string {
  // Build preview header
  let preview = `Current contents of ${filePath}:\n`;
  preview += '━'.repeat(60) + '\n';

  // Use shared formatting function (same as view command)
  preview += formatFileContent(content);

  // Add footer
  preview += '\n' + '━'.repeat(60);

  return preview;
}

/**
 * Execute an operation with read lock
 * Multiple concurrent readers allowed
 *
 * @param filePath - Absolute filesystem path to lock
 * @param operation - Async operation to execute while holding lock
 * @returns Result from operation
 */
export async function withReadLock<T>(
  filePath: string,
  operation: () => Promise<T>,
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
 *
 * Uses content checksum comparison to detect modifications:
 * 1. Read file and compute checksum (before lock)
 * 2. Compare with cached checksum (detects sequential modifications)
 * 3. Acquire write lock
 * 4. Re-read and verify checksum (detects concurrent modifications during lock wait)
 *
 * @param filePath - Absolute filesystem path to lock
 * @param checkConcurrency - Whether to check for concurrent modifications
 * @param operation - Async operation to execute while holding lock
 * @returns Result from operation
 * @throws Error if file was modified sequentially or concurrently
 */
export async function withWriteLock<T>(
  filePath: string,
  checkConcurrency: boolean,
  operation: () => Promise<T>,
): Promise<T> {
  // For operations that don't need concurrency checking (e.g., create, delete)
  if (!checkConcurrency) {
    const pathToLock = await determinePathToLock(filePath);
    const release = await lockManager.acquireWriteLock(pathToLock);
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
    if (fsError.code === 'ENOENT' || fsError.code === 'EISDIR') {
      // File doesn't exist or is a directory - can't check concurrency
      // Let the operation handle validation and error messaging
      const pathToLock = await determinePathToLock(filePath);
      const release = await lockManager.acquireWriteLock(pathToLock);
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
        preview +
        '\n\n' +
        'Please review the current contents and retry if appropriate.',
    );
  }

  // Acquire exclusive write lock
  const pathToLock = await determinePathToLock(filePath);
  const release = await lockManager.acquireWriteLock(pathToLock);

  try {
    // Re-read and verify checksum (detects concurrent modifications during lock wait)
    const contentNow = await fs.readFile(filePath, 'utf-8');
    const checksumNow = computeChecksum(contentNow);

    if (checksumNow !== checksumBefore) {
      // File modified while waiting for lock
      const preview = makeContentPreview(contentNow, filePath);

      throw new Error(
        'File was modified while waiting for lock.\n\n' +
          preview +
          '\n\n' +
          'Please review the current contents and retry.',
      );
    }

    // Execute operation - file hasn't changed
    return await operation();
  } finally {
    await release();
  }
}

/**
 * Execute an operation with write locks on multiple paths atomically
 *
 * NEW FUNCTION for rename operations that need to lock both source and destination.
 * Acquires all locks atomically (in sorted order to prevent deadlock).
 *
 * @param filePaths - Array of absolute filesystem paths to lock
 * @param operation - Async operation to execute while holding all locks
 * @returns Result from operation
 */
export async function withMultipleWriteLocks<T>(
  filePaths: string[],
  operation: () => Promise<T>,
): Promise<T> {
  // Determine what to lock for each path
  const pathsToLock = await Promise.all(
    filePaths.map(fp => determinePathToLock(fp)),
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

// Export LockManager for testing
export { lockManager };
