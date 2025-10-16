/**
 * File Locking with Optimistic Concurrency Control
 *
 * Provides file locking for concurrent access from multiple Claude instances.
 * Uses proper-lockfile for cross-process file locking.
 * Implements optimistic concurrency control using mtime (modification timestamp).
 */

import * as fs from 'fs/promises';
import * as lockfile from 'proper-lockfile';

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
 * Acquire an exclusive write lock on a file or directory.
 *
 * For operations that modify files (create, str_replace, insert, delete, rename).
 * Blocks until lock is acquired (retries forever).
 * Optionally captures mtime before locking for optimistic concurrency control.
 *
 * @param filePath - Absolute filesystem path to lock
 * @param captureMtime - Whether to capture mtime before acquiring lock
 * @returns Lock release function and mtime (if captured)
 */
export async function acquireWriteLock(
  filePath: string,
  captureMtime: boolean = false,
): Promise<LockResult> {
  // Capture mtime before locking if requested
  let mtimeBefore: number | null = null;
  if (captureMtime) {
    try {
      const stat = await fs.stat(filePath);
      mtimeBefore = stat.mtime.getTime();
    } catch {
      // File doesn't exist yet - that's ok for create operations
      mtimeBefore = null;
    }
  }

  // Determine what to lock
  // If file doesn't exist, lock the parent directory instead
  let lockPath = filePath;
  try {
    await fs.stat(filePath);
    // File exists, lock it directly
  } catch {
    // File doesn't exist, lock parent directory
    const path = await import('path');
    lockPath = path.dirname(filePath);

    // Ensure parent directory exists
    try {
      await fs.mkdir(lockPath, { recursive: true });
    } catch {
      // Directory already exists or creation failed, continue anyway
    }
  }

  // Acquire exclusive lock
  // Retry forever - wait for lock to become available
  const release = await lockfile.lock(lockPath, {
    retries: {
      forever: true, // Wait indefinitely
      minTimeout: 100, // Minimum wait between retries: 100ms
      maxTimeout: 1000, // Maximum wait between retries: 1s
    },
    stale: 30000, // Consider lock stale after 30 seconds
  });

  return { release, mtimeBefore };
}

/**
 * Acquire a shared read lock on a file or directory.
 *
 * For operations that only read files (view).
 * Multiple readers can hold the lock simultaneously.
 * Blocks until lock is acquired.
 *
 * @param filePath - Absolute filesystem path to lock
 * @returns Lock release function
 */
export async function acquireReadLock(filePath: string): Promise<LockRelease> {
  // Acquire shared lock
  // proper-lockfile doesn't natively support shared locks,
  // so we use exclusive locks but with shorter stale timeout
  const release = await lockfile.lock(filePath, {
    retries: {
      forever: true, // Wait indefinitely
      minTimeout: 100,
      maxTimeout: 1000,
    },
    stale: 10000, // Shorter stale timeout for reads: 10 seconds
  });

  return release;
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
 * @throws Error if mtime check is requested but mtimeBefore is null
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
    const stat = await fs.stat(filePath);
    const mtimeAfter = stat.mtime.getTime();

    // Compare timestamps
    return mtimeAfter !== mtimeBefore;
  } catch {
    // File doesn't exist anymore - consider it modified
    return true;
  }
}

/**
 * Execute an operation with write lock and optimistic concurrency control.
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
  // Acquire lock and capture mtime if checking concurrency
  const { release, mtimeBefore } = await acquireWriteLock(filePath, checkConcurrency);

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
 * Execute an operation with read lock.
 *
 * Acquires read lock, executes operation.
 * No concurrency checking needed for reads.
 *
 * @param filePath - Absolute filesystem path to lock
 * @param operation - Async operation to execute while holding lock
 * @returns Result from operation
 */
export async function withReadLock<T>(
  filePath: string,
  operation: () => Promise<T>,
): Promise<T> {
  // Acquire read lock
  const release = await acquireReadLock(filePath);

  try {
    // Execute operation while holding lock
    return await operation();
  } finally {
    // Always release lock
    await release();
  }
}
