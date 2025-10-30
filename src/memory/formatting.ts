/**
 * Shared formatting utilities for file content display
 * Used by operations.ts (view command) and locking.ts (error messages)
 */

/**
 * Format file content with line numbers
 *
 * @param content - File content as string
 * @param viewRange - Optional [start, end] line range (1-based, end can be -1 for EOF)
 * @returns Formatted content with line numbers
 */
export function formatFileContent(content: string, viewRange?: [number, number]): string {
  // Check if file is empty
  if (content === '') {
    return 'Memory file is empty.';
  }

  const lines = content.split('\n');

  // Determine which lines to display
  let displayLines = lines;
  let startNum = 1;

  if (viewRange && viewRange.length === 2) {
    // Validate line range
    const requestedStart = viewRange[0];
    const requestedEnd = viewRange[1];

    // Check if start line is valid (1-based, or -1 for special meaning)
    if (requestedStart < 1) {
      throw new Error(`Invalid line range: start line must be >= 1, got ${requestedStart}`);
    }

    // Check if range is within file bounds
    if (requestedStart > lines.length) {
      throw new Error(
        `Line range out of bounds: requested start line ${requestedStart}, but file only has ${lines.length} lines`,
      );
    }

    // Check if end line is valid (must be >= start, or -1 for EOF)
    if (requestedEnd !== -1 && requestedEnd < requestedStart) {
      throw new Error(
        `Invalid line range: end line ${requestedEnd} is before start line ${requestedStart}`,
      );
    }

    // Extract line range
    const startLine = Math.max(1, requestedStart) - 1;
    const endLine = requestedEnd === -1 ? lines.length : Math.min(requestedEnd, lines.length);
    displayLines = lines.slice(startLine, endLine);
    startNum = startLine + 1;
  }

  // Format with line numbers (4-space padding, colon separator)
  const numberedLines = displayLines.map(
    (line, i) => `${String(i + startNum).padStart(4, ' ')}: ${line}`,
  );

  return numberedLines.join('\n');
}
