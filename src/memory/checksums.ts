/**
 * Content Checksum Utilities for Concurrency Detection
 *
 * Provides in-memory caching of file content checksums to detect modifications
 * across separate MCP server operations (not just during lock acquisition).
 *
 * Key Features:
 * - SHA-256 content hashing for reliable change detection
 * - In-memory cache per server process (Map<path, checksum>)
 * - Cross-process detection via shared filesystem state
 * - Normalized path handling to prevent cache inconsistencies
 *
 * Usage Pattern:
 * 1. After read/write operations: setCachedChecksum(path, computeChecksum(content))
 * 2. Before write operations: compare getCachedChecksum(path) with current disk checksum
 * 3. On delete/rename: clearCachedChecksum(path) to remove stale entries
 */

import crypto from 'crypto';
import * as path from 'path';

/**
 * In-memory checksum cache
 * Key: Normalized absolute file path
 * Value: SHA-256 hex digest (64 characters)
 *
 * Each stdio MCP server process has its own cache instance.
 * Detection works because all processes check same filesystem state.
 */
const checksumCache = new Map<string, string>();

/**
 * Compute SHA-256 checksum of file content
 *
 * @param content - File content as string (UTF-8)
 * @returns Hex digest (64 characters, lowercase)
 *
 * Performance: Fast SHA-256 hashing using Node crypto library
 * - Small files (1-10 KB): Sub-millisecond
 * - Medium files (50 KB): ~0.1ms typical
 * - Performance scales linearly with content size
 * - Actual speed depends on CPU and Node.js version
 */
export function computeChecksum(content: string): string {
  // Create SHA-256 hash of content
  return crypto.createHash('sha256').update(content, 'utf-8').digest('hex');
}

/**
 * Get cached checksum for a file path
 *
 * @param filePath - Absolute file path
 * @returns Cached checksum or undefined if not cached
 *
 * Returns undefined for:
 * - Files never accessed by this server instance
 * - Files accessed with partial reads (view_range)
 * - Files deleted or renamed (cache cleared)
 */
export function getCachedChecksum(filePath: string): string | undefined {
  // Normalize path to canonical form to ensure cache hits
  // Resolves: symlinks, relative segments (. and ..)
  // Note: Case handling depends on filesystem (case-sensitive on Linux,
  //       case-insensitive on macOS/Windows)
  return checksumCache.get(path.resolve(filePath));
}

/**
 * Store checksum in cache
 *
 * @param filePath - Absolute file path
 * @param checksum - SHA-256 hex digest (64 chars)
 *
 * Call after:
 * - Reading entire file (not partial with view_range)
 * - Writing file (create, str_replace, insert, delete operations)
 *
 * This enables sequential modification detection on next access.
 */
export function setCachedChecksum(filePath: string, checksum: string): void {
  // Normalize path to prevent duplicate entries
  checksumCache.set(path.resolve(filePath), checksum);
}

/**
 * Remove checksum from cache
 *
 * @param filePath - Absolute file path
 *
 * Call when:
 * - Deleting file (no longer exists)
 * - Renaming file (old path no longer valid)
 * - Directory deletion (remove entry, children deleted on disk)
 *
 * Does nothing if path not in cache (safe to call unconditionally).
 */
export function clearCachedChecksum(filePath: string): void {
  // Normalize path before deletion
  checksumCache.delete(path.resolve(filePath));
}

/**
 * Clear entire cache
 *
 * Used for:
 * - Testing (reset state between tests)
 * - Manual cache invalidation if needed
 *
 * Cache automatically clears on server restart (in-memory only).
 */
export function clearAllCachedChecksums(): void {
  checksumCache.clear();
}

/**
 * Get all cached entries (for iteration)
 * Used internally for recursive cache clearing
 *
 * @returns Iterator over [path, checksum] pairs
 */
export function getAllCachedEntries(): IterableIterator<[string, string]> {
  return checksumCache.entries();
}

/**
 * Get cache statistics
 *
 * @returns Cache size and memory estimate
 *
 * Useful for:
 * - Debugging cache behavior
 * - Monitoring memory usage
 * - Performance analysis
 */
export function getChecksumCacheStats(): {
  size: number;
  memoryEstimate: number;
} {
  return {
    size: checksumCache.size,
    // APPROXIMATE memory per entry (actual varies by V8 version and path lengths):
    // - Path key: ~100 bytes average (varies widely: 20-500+ bytes)
    // - Checksum value: 64 chars × 2 bytes (UTF-16) = ~128 bytes
    // - Map overhead: ~40-80 bytes (V8 implementation detail)
    // Total estimate: ~270 bytes per entry (rough approximation)
    //
    // NOTE: This is an ORDER-OF-MAGNITUDE estimate for monitoring purposes.
    // Do not rely on this for precise memory accounting.
    memoryEstimate: checksumCache.size * 270,
  };
}
