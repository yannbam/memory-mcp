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
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import { withReadLock, withWriteLock } from './locking.js';
import { validatePath } from './path-security.js';
import { renderDirectoryTree } from './tree-view.js';
import type { Logger } from '../utils/logger.js';

/**
 * Memory command interfaces matching MCP memory tool specification
 */

export interface ViewCommand {
  path: string;
  view_range?: [number, number];
}

export interface CreateCommand {
  path: string;
  file_text: string;
}

export interface StrReplaceCommand {
  path: string;
  old_str: string;
  new_str: string;
}

export interface InsertCommand {
  path: string;
  insert_line: number;
  insert_text: string;
}

export interface DeleteCommand {
  path: string;
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
 */
async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
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
      // View directory contents
      return await viewDirectory(fullPath, command.path, context);
    } else if (stat.isFile()) {
      // View file contents
      return await viewFile(fullPath, command.view_range);
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

  // Format output
  return `Directory: ${memoryPath}\n` + items.map((item) => `- ${item}`).join('\n');
}

/**
 * Helper: View file contents with optional line range
 */
async function viewFile(fullPath: string, viewRange?: [number, number]): Promise<string> {
  // Read file content
  const content = await fs.readFile(fullPath, 'utf-8');
  const lines = content.split('\n');

  // Determine which lines to display
  let displayLines = lines;
  let startNum = 1;

  if (viewRange && viewRange.length === 2) {
    // Extract line range
    const startLine = Math.max(1, viewRange[0]) - 1;
    const endLine = viewRange[1] === -1 ? lines.length : viewRange[1];
    displayLines = lines.slice(startLine, endLine);
    startNum = startLine + 1;
  }

  // Format with line numbers
  const numberedLines = displayLines.map(
    (line, i) => `${String(i + startNum).padStart(4, ' ')}: ${line}`,
  );

  return numberedLines.join('\n');
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

  // Execute with write lock and concurrency check
  // Always enable concurrency checking - locking module handles non-existent files correctly
  // (mtimeBefore will be null, wasFileModified returns false)
  await withWriteLock(fullPath, true, async () => {
    // Ensure parent directory exists
    const dir = path.dirname(fullPath);
    if (!(await exists(dir))) {
      // Create parent directory
      await fs.mkdir(dir, { recursive: true });
    }

    // Write file content
    await fs.writeFile(fullPath, command.file_text, 'utf-8');
  });

  // Log operation
  await context.logger.debug('create', {
    path: command.path,
    file_size: command.file_text.length,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  return `File created successfully at ${command.path}`;
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
    const count = content.split(command.old_str).length - 1;

    if (count === 0) {
      throw new Error(`Text not found in ${command.path}`);
    } else if (count > 1) {
      throw new Error(`Text appears ${count} times in ${command.path}. Must be unique.`);
    }

    // Replace text
    const newContent = content.replace(command.old_str, command.new_str);

    // Write updated content
    await fs.writeFile(fullPath, newContent, 'utf-8');
  });

  // Log operation
  await context.logger.debug('str_replace', {
    path: command.path,
    old_str_length: command.old_str.length,
    new_str_length: command.new_str.length,
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
    const lines = content.split('\n');

    // Validate insert_line
    if (command.insert_line < 0 || command.insert_line > lines.length) {
      throw new Error(`Invalid insert_line ${command.insert_line}. Must be 0-${lines.length}`);
    }

    // Insert text at specified line
    // Remove trailing newline from insert_text to avoid double newlines
    lines.splice(command.insert_line, 0, command.insert_text.replace(/\n$/, ''));

    // Write updated content
    await fs.writeFile(fullPath, lines.join('\n'), 'utf-8');
  });

  // Log operation
  await context.logger.debug('insert', {
    path: command.path,
    insert_line: command.insert_line,
    insert_text_length: command.insert_text.length,
    duration_ms: Date.now() - startTime,
    success: true,
  });

  return `Text inserted at line ${command.insert_line} in ${command.path}`;
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

  // Validate and convert path
  const fullPath = validatePath(command.path, context.memoryRoot);

  // Track what was deleted for return message
  let deletedType: 'file' | 'directory' = 'file';

  // Execute with write lock (no concurrency check needed for delete)
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
    } else if (stat.isDirectory()) {
      // Delete directory recursively
      await fs.rm(fullPath, { recursive: true });
      deletedType = 'directory';
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

  // Execute with write lock on source path (no concurrency check needed for rename)
  await withWriteLock(oldFullPath, false, async () => {
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
    } else if (sourceStat.isDirectory()) {
      // For directories: Use fs.rename (link doesn't work for directories)
      // Check if destination exists first (has TOCTOU race, but unavoidable for directories)
      if (await exists(newFullPath)) {
        throw new Error(`Destination already exists: ${command.new_path}`);
      }
      await fs.rename(oldFullPath, newFullPath);
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
