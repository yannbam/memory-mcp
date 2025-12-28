/**
 * Memory Operations
 *
 * Implements all 6 memory tool commands:
 * - view: Show directory contents or file contents with optional line ranges
 * - create: Create or overwrite files
 * - str_replace: Replace unique text in files
 * - insert: Insert text at specific lines
 * - delete: Delete files or directories
 * - rename: Rename or move files/directories
 *
 * All operations use file locking for concurrent access safety.
 *
 * ## Parameter Combinations Reference
 *
 * **view:**
 * - path (required)
 * - view_range (optional) - only for files, e.g., [1, 10] or [5, -1] for EOF
 *
 * **create:**
 * - path (required)
 * - file_text (optional) - omit or use "" to create empty file
 *
 * **str_replace:**
 * - path (required)
 * - old_str/old_string (required, use either variant)
 * - new_str/new_string (optional) - omit or use "" to delete text
 *
 * **insert:**
 * - path (required)
 * - insert_text (required)
 * - insert_line (optional) - omit to append to end of file
 *
 * **delete:**
 * - path (required)
 * - delete_line (optional) - delete specific line number
 * - old_str/old_string (optional) - delete unique text occurrence
 * - Note: Cannot mix delete_line with old_str (choose one approach)
 * - Note: Without either, deletes entire file/directory
 *
 * **rename:**
 * - old_path (required)
 * - new_path (required)
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { withReadLock, withWriteLock, withMultipleWriteLocks } from './locking.js';
import { validatePath } from './path-security.js';
import { renderDirectoryTree } from './tree-view.js';
import {
  computeChecksum,
  setCachedChecksum,
  clearCachedChecksum,
  getAllCachedEntries,
} from './checksums.js';
import { formatFileContent } from './formatting.js';
import type { Logger } from '../utils/logger.js';

/**
 * Memory command interfaces matching MCP memory tool specification
 */

export interface ViewCommand {
  path: string;
  view_range?: number[]; // Array of exactly 2 numbers [start, end], validated by Zod schema
}

export interface CreateCommand {
  path: string;
  file_text?: string;
}

export interface StrReplaceCommand {
  path: string;
  old_str?: string;
  old_string?: string;
  new_str?: string;
  new_string?: string;
}

export interface InsertCommand {
  path: string;
  insert_line?: number;
  insert_text: string;
}

export interface DeleteCommand {
  path: string;
  delete_line?: number;
  old_str?: string;
  old_string?: string;
}

export interface RenameCommand {
  old_path: string;
  new_path: string;
}

/**
 * Memory operations context
 */
export interface OperationsContext {
  memoryRoot: string;
  logger: Logger;
  treeView: boolean;
}

/**
 * Helper: Check if path exists
 * Only catches ENOENT (file not found) - rethrows other errors
 */
async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch (err) {
    const fsError = err as { code?: string; message?: string };

    // File not found - expected case
    if (fsError.code === 'ENOENT') {
      return false;
    }

    // Permission denied - helpful error message
    if (fsError.code === 'EACCES') {
      throw new Error(
        `Permission denied accessing path: ${filePath}\n` +
          `Check filesystem permissions for the memory-mcp process.`,
      );
    }

    // Other filesystem errors - rethrow with context
    throw new Error(
      `Filesystem error checking path: ${filePath}\n` +
        `Error code: ${fsError.code ?? 'UNKNOWN'}\n` +
        `Message: ${fsError.message ?? 'Unknown error'}`,
    );
  }
}

/**
 * View command: Show directory contents or file contents with optional line ranges
 *
 * @param command - View command parameters
 * @param context - Operations context with memory root and logger
 * @returns Formatted output string
 */
export async function view(command: ViewCommand, context: OperationsContext): Promise<string> {
  const startTime = Date.now();

  // Validate and convert path
  const fullPath = validatePath(command.path, context.memoryRoot);

  // Check if path exists before trying to lock
  if (!(await exists(fullPath))) {
    throw new Error(`Path not found: ${command.path}`);
  }

  // Execute with read lock
  const result = await withReadLock(fullPath, async () => {

    // Get file stats to determine type
    const stat = await fs.stat(fullPath);

    if (stat.isDirectory()) {
      // View directory contents (don't cache - directories change frequently)
      return await viewDirectory(fullPath, command.path, context);
    } else if (stat.isFile()) {
      // View file contents
      const fileContent = await viewFile(fullPath, command.view_range);

      // Cache checksum after reading entire file (not partial reads with view_range)
      if (!command.view_range) {
        // TODO: Optimize - viewFile() already read content, avoid double-read
        // Current: Read separately because viewFile() may format/transform content
        // Better: Refactor viewFile() to return both raw content and formatted output
        const content = await fs.readFile(fullPath, 'utf-8');
        setCachedChecksum(fullPath, computeChecksum(content));
      }

      return fileContent;
    } else {
      throw new Error(`Path not found: ${command.path}`);
    }
  });

  // Log operation
  await context.logger.debug('view', {
    path: command.path,
    view_range: command.view_range,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  return result;
}

/**
 * Helper: View directory contents
 */
async function viewDirectory(
  fullPath: string,
  memoryPath: string,
  context: OperationsContext,
): Promise<string> {
  // Use tree view if enabled
  if (context.treeView) {
    return await renderDirectoryTree(fullPath, memoryPath);
  }

  // Simple flat listing (existing behavior)
  // Read directory entries
  const items: string[] = [];
  const dirContents = await fs.readdir(fullPath);

  // Sort and format entries
  for (const item of dirContents.sort()) {
    // Skip hidden files
    if (item.startsWith('.')) {
      continue;
    }

    // Get item stats to determine if directory
    const itemPath = path.join(fullPath, item);
    const itemStat = await fs.stat(itemPath);

    // Append / to directories
    items.push(itemStat.isDirectory() ? `${item}/` : item);
  }

  // Check if directory is empty (after filtering hidden files)
  if (items.length === 0) {
    return 'Directory is empty.';
  }

  // Format output
  return `Directory: ${memoryPath}\n` + items.map((item) => `- ${item}`).join('\n');
}

/**
 * Helper: View file contents with optional line range
 */
async function viewFile(fullPath: string, viewRange?: number[]): Promise<string> {
  // Read file content
  const content = await fs.readFile(fullPath, 'utf-8');

  // Use shared formatting function
  return formatFileContent(content, viewRange);
}

/**
 * Create command: Create or overwrite a file
 *
 * @param command - Create command parameters
 * @param context - Operations context with memory root and logger
 * @returns Success message
 */
export async function create(command: CreateCommand, context: OperationsContext): Promise<string> {
  const startTime = Date.now();

  // Validate and convert path
  const fullPath = validatePath(command.path, context.memoryRoot);

  // Execute with write lock (no concurrency check needed - create is for new files only)
  let writtenContent: string = '';
  await withWriteLock(fullPath, false, async () => {
    // Ensure parent directory exists
    const dir = path.dirname(fullPath);
    if (!(await exists(dir))) {
      // Create parent directory
      await fs.mkdir(dir, { recursive: true });
    }

    // Check if file already exists - fail fast unless it's empty
    if (await exists(fullPath)) {
      const existingContent = await fs.readFile(fullPath, 'utf-8');
      if (existingContent !== '') {
        throw new Error(`File already exists at ${command.path}`);
      }
      // Empty file - allow overwrite (common case: created empty, now want content)
    }

    // Write file content (default to empty string if not provided)
    writtenContent = command.file_text ?? '';
    await fs.writeFile(fullPath, writtenContent, 'utf-8');

    // Cache checksum of newly created file
    setCachedChecksum(fullPath, computeChecksum(writtenContent));
  });

  // Log operation
  await context.logger.debug('create', {
    path: command.path,
    file_size: (command.file_text ?? '').length,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  // Return appropriate message based on whether file has content
  const isEmpty = !command.file_text || command.file_text === '';
  return isEmpty ? 'Created empty memory file.' : `File created successfully at ${command.path}`;
}

/**
 * Str_replace command: Replace unique text in a file
 *
 * @param command - Str_replace command parameters
 * @param context - Operations context with memory root and logger
 * @returns Success message
 */
export async function str_replace(
  command: StrReplaceCommand,
  context: OperationsContext,
): Promise<string> {
  const startTime = Date.now();

  // Normalize parameter names: accept both old_str/new_str and old_string/new_string
  // Validate that only one variant of each parameter is provided
  if (command.old_str && command.old_string) {
    throw new Error('Cannot provide both old_str and old_string - use one or the other');
  }
  if (command.new_str && command.new_string) {
    throw new Error('Cannot provide both new_str and new_string - use one or the other');
  }

  const oldStr = command.old_str ?? command.old_string;
  const newStr = command.new_str ?? command.new_string ?? '';

  // Validate that at least one variant of each parameter is provided
  if (!oldStr) {
    throw new Error('Must provide either old_str or old_string');
  }

  // Validate and convert path
  const fullPath = validatePath(command.path, context.memoryRoot);

  // Execute with write lock and concurrency check
  await withWriteLock(fullPath, true, async () => {
    // Check if file exists
    if (!(await exists(fullPath))) {
      throw new Error(`File not found: ${command.path}`);
    }

    // Verify it's a file, not a directory
    const stat = await fs.stat(fullPath);
    if (!stat.isFile()) {
      throw new Error(`Path is not a file: ${command.path}`);
    }

    // Read file content
    const content = await fs.readFile(fullPath, 'utf-8');

    // Count occurrences of old_str
    const count = content.split(oldStr).length - 1;

    if (count === 0) {
      throw new Error(`Text not found in ${command.path}`);
    } else if (count > 1) {
      throw new Error(`Text appears ${count} times in ${command.path}. Must be unique.`);
    }

    // Replace text
    const newContent = content.replace(oldStr, newStr);

    // Write updated content
    await fs.writeFile(fullPath, newContent, 'utf-8');

    // Cache checksum of modified file
    setCachedChecksum(fullPath, computeChecksum(newContent));
  });

  // Log operation
  await context.logger.debug('str_replace', {
    path: command.path,
    old_str_length: oldStr.length,
    new_str_length: newStr.length,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  return `File ${command.path} has been edited`;
}

/**
 * Insert command: Insert text at a specific line
 *
 * @param command - Insert command parameters
 * @param context - Operations context with memory root and logger
 * @returns Success message
 */
export async function insert(command: InsertCommand, context: OperationsContext): Promise<string> {
  const startTime = Date.now();

  // Validate and convert path
  const fullPath = validatePath(command.path, context.memoryRoot);

  // Execute with write lock and concurrency check
  await withWriteLock(fullPath, true, async () => {
    // Check if file exists
    if (!(await exists(fullPath))) {
      throw new Error(`File not found: ${command.path}`);
    }

    // Verify it's a file, not a directory
    const stat = await fs.stat(fullPath);
    if (!stat.isFile()) {
      throw new Error(`Path is not a file: ${command.path}`);
    }

    // Read file content
    const content = await fs.readFile(fullPath, 'utf-8');
    // Handle empty file: split('') gives [''], but we want []
    const lines = content === '' ? [] : content.split('\n');

    // Determine insert position
    let insertLine: number;
    if (command.insert_line === undefined) {
      // Append to end - insert after last line
      insertLine = lines.length + 1;
    } else {
      insertLine = command.insert_line;
      // Validate insert_line (1-based indexing)
      if (insertLine < 1 || insertLine > lines.length + 1) {
        throw new Error(`Invalid insert_line ${insertLine}. Must be 1-${lines.length + 1}`);
      }
    }

    // Insert text at specified line (convert from 1-based to 0-based array index)
    // Remove trailing newline from insert_text to avoid double newlines
    lines.splice(insertLine - 1, 0, command.insert_text.replace(/\n$/, ''));

    // Write updated content
    const newContent = lines.join('\n');
    await fs.writeFile(fullPath, newContent, 'utf-8');

    // Cache checksum of modified file
    setCachedChecksum(fullPath, computeChecksum(newContent));
  });

  // Log operation
  await context.logger.debug('insert', {
    path: command.path,
    insert_line: command.insert_line,
    insert_text_length: command.insert_text.length,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  if (command.insert_line === undefined) {
    return `Text appended to end of ${command.path}`;
  } else {
    return `Text inserted at line ${command.insert_line} in ${command.path}`;
  }
}

/**
 * Clear cached checksums for a directory and all children
 * Prevents memory leaks when deleting directories
 *
 * @param dirPath - Absolute directory path
 */
function clearCachedChecksumsRecursive(dirPath: string): void {
  const normalizedDir = path.resolve(dirPath);

  // Iterate through cache and remove entries under this directory
  for (const [cachedPath] of getAllCachedEntries()) {
    if (cachedPath.startsWith(normalizedDir + path.sep) || cachedPath === normalizedDir) {
      clearCachedChecksum(cachedPath);
    }
  }
}

/**
 * Delete command: Delete a file or directory
 *
 * @param command - Delete command parameters
 * @param context - Operations context with memory root and logger
 * @returns Success message
 */
export async function deleteOp(
  command: DeleteCommand,
  context: OperationsContext,
): Promise<string> {
  const startTime = Date.now();

  // Prevent deletion of /memories root
  if (command.path === '/memories') {
    throw new Error('Cannot delete the /memories directory itself');
  }

  // Normalize parameter names for old_str (forgiving naming)
  const old_str = command.old_str ?? command.old_string;

  // Validation: Cannot mix position-based and content-based deletion
  if (command.delete_line !== undefined && old_str !== undefined) {
    throw new Error(
      'Cannot use both delete_line and old_str - choose position-based OR content-based deletion'
    );
  }

  // Validate and convert path
  const fullPath = validatePath(command.path, context.memoryRoot);

  // Handle text-based deletion
  if (old_str !== undefined) {
    return await deleteMatchingText(command.path, fullPath, old_str, context, startTime);
  }

  // Handle line-specific deletion
  if (command.delete_line !== undefined) {
    const deleteLine = command.delete_line;

    // Execute with write lock and concurrency check for line deletion
    await withWriteLock(fullPath, true, async () => {
      // Check if file exists
      if (!(await exists(fullPath))) {
        throw new Error(`File not found: ${command.path}`);
      }

      // Verify it's a file, not a directory
      const stat = await fs.stat(fullPath);
      if (!stat.isFile()) {
        throw new Error(`Cannot delete line from directory: ${command.path}`);
      }

      // Read file content
      const content = await fs.readFile(fullPath, 'utf-8');
      const lines = content.split('\n');

      // Validate delete_line (1-based indexing)
      if (deleteLine < 1 || deleteLine > lines.length) {
        throw new Error(`Invalid delete_line ${deleteLine}. Must be 1-${lines.length}`);
      }

      // Delete the specified line (convert from 1-based to 0-based array index)
      lines.splice(deleteLine - 1, 1);

      // Write updated content
      const newContent = lines.join('\n');
      await fs.writeFile(fullPath, newContent, 'utf-8');

      // Cache checksum of modified file
      setCachedChecksum(fullPath, computeChecksum(newContent));
    });

    // Log operation
    await context.logger.debug('delete', {
      path: command.path,
      delete_line: deleteLine,
      type: 'line',
      duration_ms: Date.now() - startTime,
      success: true,
    });

    return `Line ${deleteLine} deleted from ${command.path}`;
  }

  // Track what was deleted for return message
  let deletedType: 'file' | 'directory' = 'file';

  // Execute with write lock (no concurrency check needed for file/directory delete)
  await withWriteLock(fullPath, false, async () => {
    // Check if path exists
    if (!(await exists(fullPath))) {
      throw new Error(`Path not found: ${command.path}`);
    }

    // Get file stats to determine type
    const stat = await fs.stat(fullPath);

    if (stat.isFile()) {
      // Delete file
      await fs.unlink(fullPath);
      deletedType = 'file';

      // Clear cache for deleted file
      clearCachedChecksum(fullPath);
    } else if (stat.isDirectory()) {
      // Delete directory recursively
      await fs.rm(fullPath, { recursive: true });
      deletedType = 'directory';

      // Clear cache for deleted directory and all children
      clearCachedChecksumsRecursive(fullPath);
    } else {
      throw new Error(`Path not found: ${command.path}`);
    }
  });

  // Log operation
  await context.logger.debug('delete', {
    path: command.path,
    type: deletedType,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  return deletedType === 'file'
    ? `File deleted: ${command.path}`
    : `Directory deleted: ${command.path}`;
}

/**
 * Helper: Delete all occurrences of matching text from a file
 * Empty lines resulting from deletion are also removed
 */
async function deleteMatchingText(
  memoryPath: string,
  fullPath: string,
  searchText: string,
  context: OperationsContext,
  startTime: number,
): Promise<string> {
  // Count occurrences for return message
  let occurrences = 0;

  // Execute with write lock and concurrency check
  await withWriteLock(fullPath, true, async () => {
    // Check if file exists
    if (!(await exists(fullPath))) {
      throw new Error(`File not found: ${memoryPath}`);
    }

    // Verify it's a file, not a directory
    const stat = await fs.stat(fullPath);
    if (!stat.isFile()) {
      throw new Error(`Cannot delete text from directory: ${memoryPath}`);
    }

    // Read file content
    const content = await fs.readFile(fullPath, 'utf-8');

    // Count occurrences before deletion
    occurrences = (content.match(new RegExp(escapeRegExp(searchText), 'g')) || []).length;

    if (occurrences === 0) {
      throw new Error(`Text not found in file: "${searchText}"`);
    } else if (occurrences > 1) {
      throw new Error(`Text appears ${occurrences} times in ${memoryPath}. Must be unique.`);
    }

    // Delete the unique occurrence of the text
    const afterDeletion = content.replace(searchText, '');

    // Remove empty lines
    const lines = afterDeletion.split('\n');
    const nonEmptyLines = lines.filter((line) => line.trim() !== '');

    // Write cleaned content
    const finalContent = nonEmptyLines.join('\n');
    await fs.writeFile(fullPath, finalContent, 'utf-8');

    // Cache checksum of modified file
    setCachedChecksum(fullPath, computeChecksum(finalContent));
  });

  // Log operation
  await context.logger.debug('delete_text', {
    path: memoryPath,
    searchText,
    occurrences,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  return `Deleted ${occurrences} occurrence(s) of "${searchText}" from ${memoryPath}`;
}

/**
 * Helper: Escape special regex characters for literal string matching
 */
function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Rename command: Rename or move a file/directory
 *
 * @param command - Rename command parameters
 * @param context - Operations context with memory root and logger
 * @returns Success message
 */
export async function rename(command: RenameCommand, context: OperationsContext): Promise<string> {
  const startTime = Date.now();

  // Validate and convert both paths
  const oldFullPath = validatePath(command.old_path, context.memoryRoot);
  const newFullPath = validatePath(command.new_path, context.memoryRoot);

  // Execute with write locks on BOTH source and destination (prevents race conditions)
  await withMultipleWriteLocks([oldFullPath, newFullPath], async () => {
    // Check if source path exists
    if (!(await exists(oldFullPath))) {
      throw new Error(`Source path not found: ${command.old_path}`);
    }

    // Get source file stats to determine if it's a file or directory
    const sourceStat = await fs.stat(oldFullPath);

    // Ensure destination directory exists
    const newDir = path.dirname(newFullPath);
    if (!(await exists(newDir))) {
      await fs.mkdir(newDir, { recursive: true });
    }

    if (sourceStat.isFile()) {
      // For files: Use link + unlink for atomic rename without overwrite
      // fs.link fails if destination exists (EEXIST), preventing overwrites
      // This is safer than fs.rename which silently overwrites on POSIX systems
      try {
        await fs.link(oldFullPath, newFullPath);
        await fs.unlink(oldFullPath);
      } catch (error) {
        // Check if error is due to destination already existing
        // Node.js filesystem errors have a 'code' property
        if (
          typeof error === 'object' &&
          error !== null &&
          'code' in error &&
          error.code === 'EEXIST'
        ) {
          throw new Error(`Destination already exists: ${command.new_path}`);
        }
        // Other errors (EXDEV for cross-filesystem, etc.)
        throw error;
      }

      // Clear cache for old path (new path will be cached on next access)
      clearCachedChecksum(oldFullPath);
    } else if (sourceStat.isDirectory()) {
      // For directories: Use fs.rename (link doesn't work for directories)
      // Check if destination exists first (has TOCTOU race, but unavoidable for directories)
      if (await exists(newFullPath)) {
        throw new Error(`Destination already exists: ${command.new_path}`);
      }
      await fs.rename(oldFullPath, newFullPath);

      // Clear cache for old path
      clearCachedChecksum(oldFullPath);
    } else {
      throw new Error(`Source path not found: ${command.old_path}`);
    }
  });

  // Log operation
  await context.logger.debug('rename', {
    old_path: command.old_path,
    new_path: command.new_path,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  return `Renamed ${command.old_path} to ${command.new_path}`;
}
