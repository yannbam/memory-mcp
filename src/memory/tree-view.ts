/**
 * Tree View Module
 *
 * Provides hierarchical directory visualization for memory tool.
 * Optimized for Claude instances viewing /memories at session start.
 *
 * Features:
 * - Hierarchical tree structure with unlimited depth
 * - Modification times for recency tracking
 * - File sizes for quick assessment
 * - Line counts for document length estimation
 *
 * Simplified from koding-tools-mcp/tools/ls.js for memory-specific use:
 * - All files assumed to be text (Claude writes them)
 * - No symlink handling
 * - No executable markers
 * - No clutter filtering
 * - No truncation limits
 */

import * as fs from 'fs/promises';
import * as path from 'path';

/**
 * Tree node structure representing a file or directory
 */
interface TreeNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  fileSize: number | null;
  lineCount: number | null;
  modificationTime: Date | null;
  children?: TreeNode[];
}

/**
 * Format file size in human-readable format
 *
 * @param bytes - File size in bytes
 * @returns Human-readable size (e.g., "1.5KB", "2.3MB")
 */
export function formatFileSize(bytes: number): string {
  // Handle zero bytes
  if (bytes === 0) return '0B';

  // Define size units
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const k = 1024;

  // Calculate appropriate unit
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  // Format with 1 decimal place
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + units[i];
}

/**
 * Format modification date in [YYYY/MM/DD - HH:MM:SS] format
 *
 * @param modificationTime - Modification time as Date object or timestamp
 * @returns Formatted date string or empty string if invalid
 */
export function formatModificationDate(modificationTime: Date | number): string {
  try {
    // Convert to Date object if needed
    const date = modificationTime instanceof Date ? modificationTime : new Date(modificationTime);

    // Validate date
    if (isNaN(date.getTime())) {
      return '';
    }

    // Extract date components
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');

    // Return formatted string
    return `[${year}/${month}/${day} - ${hours}:${minutes}:${seconds}]`;
  } catch {
    return '';
  }
}

/**
 * Count lines in a text file
 *
 * All files in /memories are assumed to be text files since Claude writes them.
 *
 * @param filePath - Absolute path to the file
 * @returns Number of lines, or null if couldn't read/count
 */
export async function countFileLines(filePath: string): Promise<number | null> {
  try {
    // Read file content as UTF-8 text
    const content = await fs.readFile(filePath, 'utf-8');

    // Handle empty file
    if (content.length === 0) return 0;

    // Count newlines
    const lines = content.split('\n').length;

    // Adjust for trailing newline
    // If file ends with newline, split creates empty string at end
    return content.endsWith('\n') ? lines - 1 : lines;
  } catch {
    // Return null if file cannot be read
    return null;
  }
}

/**
 * Build hierarchical tree structure from directory contents
 *
 * @param dirPath - Absolute path to directory
 * @param basePath - Base path for calculating relative paths
 * @returns Array of root-level tree nodes
 */
async function buildDirectoryTree(dirPath: string, basePath: string): Promise<TreeNode[]> {
  const nodes: TreeNode[] = [];

  try {
    // Read directory contents
    const entries = await fs.readdir(dirPath, { withFileTypes: true });

    // Process each entry
    for (const entry of entries) {
      // Skip hidden files (starting with .)
      if (entry.name.startsWith('.')) {
        continue;
      }

      // Get full path
      const fullPath = path.join(dirPath, entry.name);
      const relativePath = path.relative(basePath, fullPath);

      // Get stats for metadata
      const stats = await fs.stat(fullPath);

      if (entry.isDirectory()) {
        // Create directory node
        const dirNode: TreeNode = {
          name: entry.name,
          path: relativePath,
          type: 'directory',
          fileSize: null,
          lineCount: null,
          modificationTime: stats.mtime,
          children: [],
        };

        // Recursively build children
        dirNode.children = await buildDirectoryTree(fullPath, basePath);

        nodes.push(dirNode);
      } else if (entry.isFile()) {
        // Create file node with size and line count
        const fileNode: TreeNode = {
          name: entry.name,
          path: relativePath,
          type: 'file',
          fileSize: stats.size,
          lineCount: await countFileLines(fullPath),
          modificationTime: stats.mtime,
        };

        nodes.push(fileNode);
      }
    }

    // Sort: directories first, then files, both alphabetically
    nodes.sort((a, b) => {
      // Directories before files
      if (a.type !== b.type) {
        return a.type === 'directory' ? -1 : 1;
      }
      // Alphabetical within each category
      return a.name.localeCompare(b.name);
    });
  } catch {
    // Return empty array on error
  }

  return nodes;
}

/**
 * Render tree structure as formatted string with indentation
 *
 * @param nodes - Array of tree nodes to render
 * @param prefix - Indentation prefix for current level
 * @returns Formatted tree string
 */
function printTreeNodes(nodes: TreeNode[], prefix: string = ''): string {
  let result = '';

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const isLast = i === nodes.length - 1;

    // Determine tree connector symbols
    const connector = isLast ? '└── ' : '├── ';
    const childPrefix = isLast ? '    ' : '│   ';

    // Build display name
    let displayName = node.name;

    // Add directory indicator
    if (node.type === 'directory') {
      displayName += '/';
    }

    // Format file information
    let fileInfo = '';
    let modTimeInfo = '';

    if (node.type === 'file') {
      // Add file size and line count
      if (node.fileSize !== null) {
        const sizeStr = formatFileSize(node.fileSize);
        if (node.lineCount !== null) {
          // Text file with line count
          fileInfo = `\t(${sizeStr} / ${node.lineCount} lines)`;
        } else {
          // File without line count
          fileInfo = `\t(${sizeStr})`;
        }
      }

      // Add modification time with single tab
      if (node.modificationTime) {
        modTimeInfo = `\t${formatModificationDate(node.modificationTime)}`;
      }
    } else {
      // Directory gets double tab before modification time
      if (node.modificationTime) {
        modTimeInfo = `\t\t${formatModificationDate(node.modificationTime)}`;
      }
    }

    // Add current node to result with tree symbols
    result += `${prefix}${connector}${displayName}${fileInfo}${modTimeInfo}\n`;

    // Recursively print children with tree-style indentation
    if (node.children && node.children.length > 0) {
      result += printTreeNodes(node.children, `${prefix}${childPrefix}`);
    }
  }

  return result;
}

/**
 * Render directory tree with hierarchical structure and metadata
 *
 * Main entry point for tree view rendering. Recursively scans directory,
 * collects metadata (modification times, file sizes, line counts), and
 * formats as indented tree structure.
 *
 * @param fullPath - Absolute filesystem path to directory
 * @param memoryPath - Virtual memory path (e.g., "/memories")
 * @returns Formatted tree view string with header and metadata
 */
export async function renderDirectoryTree(fullPath: string, memoryPath: string): Promise<string> {
  // Build tree structure
  const tree = await buildDirectoryTree(fullPath, fullPath);

  // Check if directory is empty (after filtering hidden files)
  if (tree.length === 0) {
    return 'Directory is empty.';
  }

  // Get timezone for header
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;

  // Build header
  let result = `Showing contents of: ${memoryPath}\n`;
  result += `Modification dates shown in [YYYY/MM/DD - HH:MM:SS] format (${timeZone} timezone)\n\n`;

  // Print tree structure
  result += printTreeNodes(tree);

  return result;
}
