/**
 * Path Security Module
 *
 * Validates and sanitizes memory paths to prevent directory traversal attacks.
 * All paths must start with /memories and remain within the memory root directory.
 */

import * as path from 'path';

/**
 * Validates a memory path and converts it to a filesystem path.
 *
 * Security requirements:
 * - Path must start with /memories
 * - Path must not escape the memory root directory
 * - Prevents traversal attacks: ../, ..\, %2e%2e%2f, etc.
 *
 * @param memoryPath - Virtual path starting with /memories (e.g., /memories/notes.txt)
 * @param memoryRoot - Absolute filesystem path to memory root directory
 * @returns Validated absolute filesystem path
 * @throws Error if path is invalid or would escape memory root
 */
export function validatePath(memoryPath: string, memoryRoot: string): string {
  // Check path starts with /memories
  if (!memoryPath.startsWith('/memories')) {
    throw new Error(`Path must start with /memories, got: ${memoryPath}`);
  }

  // Convert virtual path to relative filesystem path
  // Remove /memories prefix and leading slash
  const relativePath = memoryPath.slice('/memories'.length).replace(/^\//, '');

  // Build full filesystem path
  // If relativePath is empty, use memoryRoot directly (for /memories itself)
  const fullPath = relativePath ? path.join(memoryRoot, relativePath) : memoryRoot;

  // Resolve to canonical absolute path
  const resolvedPath = path.resolve(fullPath);
  const resolvedRoot = path.resolve(memoryRoot);

  // Verify resolved path is within memory root
  // This prevents all forms of directory traversal
  // Must check for path separator to prevent sibling directory bypass
  // (e.g., /tmp/memory vs /tmp/memory2)
  if (!resolvedPath.startsWith(resolvedRoot + path.sep) && resolvedPath !== resolvedRoot) {
    throw new Error(`Path ${memoryPath} would escape /memories directory`);
  }

  return resolvedPath;
}

/**
 * Converts a filesystem path back to a memory path.
 *
 * @param filesystemPath - Absolute filesystem path
 * @param memoryRoot - Absolute filesystem path to memory root directory
 * @returns Virtual memory path starting with /memories
 */
export function toMemoryPath(filesystemPath: string, memoryRoot: string): string {
  // Resolve both paths to canonical form
  const resolvedPath = path.resolve(filesystemPath);
  const resolvedRoot = path.resolve(memoryRoot);

  // Verify path is within memory root
  // Must check for path separator to prevent sibling directory bypass
  if (!resolvedPath.startsWith(resolvedRoot + path.sep) && resolvedPath !== resolvedRoot) {
    throw new Error(`Path ${filesystemPath} is not within memory root`);
  }

  // Get relative path from root
  const relativePath = path.relative(resolvedRoot, resolvedPath);

  // Convert to memory path
  return relativePath ? `/memories/${relativePath}` : '/memories';
}
