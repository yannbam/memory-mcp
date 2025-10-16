/**
 * File Locking with True Reader-Writer Locks
 *
 * Provides file locking for concurrent access from multiple Claude instances.
 * Uses @esfx/async-readerwriterlock for true shared read locks and exclusive write locks.
 * Implements optimistic concurrency control using mtime (modification timestamp).
 *
 * Key Features:
 * - Shared read locks: Multiple readers can access concurrently
 * - Exclusive write locks: Only one writer at a time
 * - Multi-path atomic locking: Lock multiple paths for rename operations
 * - Optimistic concurrency: Detect modifications while waiting for locks
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { AsyncReaderWriterLock } from '@esfx/async-readerwriterlock';

/**
 * Lock release function
 */
export type LockRelease = () => Promise<void>;

/**
 * Lock acquisition result with mtime tracking
 */
export interface LockResult {
  release: LockRelease;
  mtimeBefore: number | null;
}

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
   * Optionally captures mtime before locking for optimistic concurrency control
   *
   * @param filePath - Absolute filesystem path to lock
   * @param captureMtime - Whether to capture mtime before acquiring lock
   * @returns Lock release function and mtime (if captured)
   */
  async acquireWriteLock(
    filePath: string,
    captureMtime: boolean = false,
  ): Promise<LockResult> {
    // Capture mtime BEFORE acquiring lock (for optimistic concurrency)
    let mtimeBefore: number | null = null;
    if (captureMtime) {
      try {
        const stats = await fs.stat(filePath);
        mtimeBefore = stats.mtimeMs;
      } catch (err) {
        const fsError = err as { code?: string };
        if (fsError.code !== 'ENOENT') throw err;
        // File doesn't exist yet - that's ok for create operations
      }
    }

    // Get lock instance for this path
    const lock = this.getLock(filePath);

    // Acquire exclusive write lock
    const lockHandle = await lock.write();

    // Return release function with captured mtime
    const release = (): Promise<void> => {
      lockHandle.unlock();
      this.releaseLock(filePath);
      return Promise.resolve();
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
 * Check if file was modified since captured mtime.
 *
 * Used for optimistic concurrency control - detects if file changed
 * while waiting for lock acquisition.
 *
 * @param filePath - Absolute filesystem path
 * @param mtimeBefore - Mtime captured before lock acquisition
 * @returns true if file was modified, false otherwise
 */
export async function wasFileModified(
  filePath: string,
  mtimeBefore: number | null,
): Promise<boolean> {
  // If mtime wasn't captured, can't check for modifications
  if (mtimeBefore === null) {
    return false;
  }

  // Get current mtime
  try {
    const stats = await fs.stat(filePath);
    const mtimeAfter = stats.mtimeMs;

    // Compare timestamps
    return mtimeAfter !== mtimeBefore;
  } catch {
    // File doesn't exist anymore - consider it modified
    return true;
  }
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
 * Acquires write lock, checks for concurrent modifications, executes operation.
 * Throws error if file was modified while waiting for lock.
 *
 * @param filePath - Absolute filesystem path to lock
 * @param checkConcurrency - Whether to check for concurrent modifications
 * @param operation - Async operation to execute while holding lock
 * @returns Result from operation
 * @throws Error if file was modified concurrently
 */
export async function withWriteLock<T>(
  filePath: string,
  checkConcurrency: boolean,
  operation: () => Promise<T>,
): Promise<T> {
  // Capture mtime BEFORE determining path to lock (for original filePath)
  let mtimeBefore: number | null = null;
  if (checkConcurrency) {
    try {
      const stats = await fs.stat(filePath);
      mtimeBefore = stats.mtimeMs;
    } catch (err) {
      const fsError = err as { code?: string };
      if (fsError.code !== 'ENOENT') throw err;
      // File doesn't exist yet - that's ok for create operations
    }
  }

  // Determine what to lock (parent directory if file doesn't exist)
  const pathToLock = await determinePathToLock(filePath);

  // Acquire lock (no mtime capture needed - already done above)
  const { release } = await lockManager.acquireWriteLock(pathToLock, false);

  try {
    // Check if file was modified while waiting for lock
    if (checkConcurrency && (await wasFileModified(filePath, mtimeBefore))) {
      throw new Error(
        'File has been modified by another process. ' +
          'Please read the file again and retry your operation.',
      );
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
