/**
 * Memory Operations Tests
 *
 * Comprehensive tests for all 6 memory commands with various inputs and edge cases.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as operations from '../src/memory/operations.js';
import type { Logger } from '../src/utils/logger.js';

// Mock logger for tests
const mockLogger: Logger = {
  debug: async () => {},
  close: async () => {},
};

describe('Memory Operations', () => {
  // Test memory root
  const testRoot = path.join('/tmp', 'memory-ops-test-' + Date.now());
  const memoryRoot = path.join(testRoot, 'memories');

  // Operations context
  const context: operations.OperationsContext = {
    memoryRoot,
    logger: mockLogger,
  };

  // Setup and teardown
  beforeEach(async () => {
    // Create test directory
    await fs.mkdir(memoryRoot, { recursive: true });
  });

  afterEach(async () => {
    // Clean up test directory
    await fs.rm(testRoot, { recursive: true, force: true });
  });

  describe('view command', () => {
    it('should view empty directory', async () => {
      const result = await operations.view({ path: '/memories' }, context);
      expect(result).toBe('Directory: /memories\n');
    });

    it('should view directory with files', async () => {
      // Create test files
      await fs.writeFile(path.join(memoryRoot, 'file1.txt'), 'content');
      await fs.writeFile(path.join(memoryRoot, 'file2.txt'), 'content');
      await fs.mkdir(path.join(memoryRoot, 'subdir'));

      const result = await operations.view({ path: '/memories' }, context);
      expect(result).toContain('Directory: /memories');
      expect(result).toContain('- file1.txt');
      expect(result).toContain('- file2.txt');
      expect(result).toContain('- subdir/');
    });

    it('should view file content with line numbers', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2\nline3\n');

      const result = await operations.view({ path: '/memories/test.txt' }, context);
      expect(result).toContain('   1: line1');
      expect(result).toContain('   2: line2');
      expect(result).toContain('   3: line3');
    });

    it('should view file with line range', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2\nline3\nline4\n');

      const result = await operations.view(
        { path: '/memories/test.txt', view_range: [2, 3] },
        context,
      );
      expect(result).toContain('   2: line2');
      expect(result).toContain('   3: line3');
      expect(result).not.toContain('line1');
      expect(result).not.toContain('line4');
    });

    it('should view file with line range to EOF', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2\nline3\n');

      const result = await operations.view(
        { path: '/memories/test.txt', view_range: [2, -1] },
        context,
      );
      expect(result).toContain('   2: line2');
      expect(result).toContain('   3: line3');
      expect(result).not.toContain('line1');
    });

    it('should skip hidden files in directory listing', async () => {
      // Create test files including hidden
      await fs.writeFile(path.join(memoryRoot, 'visible.txt'), 'content');
      await fs.writeFile(path.join(memoryRoot, '.hidden'), 'secret');

      const result = await operations.view({ path: '/memories' }, context);
      expect(result).toContain('- visible.txt');
      expect(result).not.toContain('.hidden');
    });

    it('should throw error for non-existent path', async () => {
      await expect(operations.view({ path: '/memories/nonexistent.txt' }, context)).rejects.toThrow(
        'Path not found',
      );
    });
  });

  describe('create command', () => {
    it('should create a new file', async () => {
      const content = 'Hello, World!';
      const result = await operations.create(
        { path: '/memories/test.txt', file_text: content },
        context,
      );

      expect(result).toBe('File created successfully at /memories/test.txt');

      // Verify file exists
      const fileContent = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(fileContent).toBe(content);
    });

    it('should create file in subdirectory (creating parent dirs)', async () => {
      const content = 'Nested content';
      await operations.create(
        { path: '/memories/sub/dir/file.txt', file_text: content },
        context,
      );

      // Verify file exists in nested directory
      const fileContent = await fs.readFile(path.join(memoryRoot, 'sub/dir/file.txt'), 'utf-8');
      expect(fileContent).toBe(content);
    });

    it('should overwrite existing file', async () => {
      // Create initial file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'original');

      // Overwrite with new content
      await operations.create({ path: '/memories/test.txt', file_text: 'updated' }, context);

      // Verify content was updated
      const fileContent = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(fileContent).toBe('updated');
    });

    it('should create empty file', async () => {
      await operations.create({ path: '/memories/empty.txt', file_text: '' }, context);

      // Verify empty file exists
      const fileContent = await fs.readFile(path.join(memoryRoot, 'empty.txt'), 'utf-8');
      expect(fileContent).toBe('');
    });
  });

  describe('str_replace command', () => {
    it('should replace unique text in file', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Hello, World!\nGoodbye, World!');

      // Replace text
      await operations.str_replace(
        {
          path: '/memories/test.txt',
          old_str: 'Hello, World!',
          new_str: 'Hi, Universe!',
        },
        context,
      );

      // Verify replacement
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('Hi, Universe!\nGoodbye, World!');
    });

    it('should throw error if text not found', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'content');

      // Attempt to replace non-existent text
      await expect(
        operations.str_replace(
          { path: '/memories/test.txt', old_str: 'missing', new_str: 'replacement' },
          context,
        ),
      ).rejects.toThrow('Text not found');
    });

    it('should throw error if text appears multiple times', async () => {
      // Create test file with duplicate text
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'duplicate\ndup licate\nduplicate');

      // Attempt to replace non-unique text
      await expect(
        operations.str_replace(
          { path: '/memories/test.txt', old_str: 'duplicate', new_str: 'unique' },
          context,
        ),
      ).rejects.toThrow('appears 2 times');
      await expect(
        operations.str_replace(
          { path: '/memories/test.txt', old_str: 'duplicate', new_str: 'unique' },
          context,
        ),
      ).rejects.toThrow('Must be unique');
    });

    it('should throw error for non-existent file', async () => {
      await expect(
        operations.str_replace(
          { path: '/memories/nonexistent.txt', old_str: 'old', new_str: 'new' },
          context,
        ),
      ).rejects.toThrow('File not found');
    });

    it('should throw error when path is a directory', async () => {
      // Create a directory
      await fs.mkdir(path.join(memoryRoot, 'subdir'));

      await expect(
        operations.str_replace({ path: '/memories/subdir', old_str: 'old', new_str: 'new' }, context),
      ).rejects.toThrow('not a file');
    });
  });

  describe('insert command', () => {
    it('should insert text at line 0 (beginning)', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line2\nline3');

      // Insert at beginning
      await operations.insert(
        { path: '/memories/test.txt', insert_line: 0, insert_text: 'line1' },
        context,
      );

      // Verify insertion
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line1\nline2\nline3');
    });

    it('should insert text in middle', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline3');

      // Insert in middle
      await operations.insert(
        { path: '/memories/test.txt', insert_line: 1, insert_text: 'line2' },
        context,
      );

      // Verify insertion
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line1\nline2\nline3');
    });

    it('should insert text at end', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2');

      // Insert at end
      await operations.insert(
        { path: '/memories/test.txt', insert_line: 2, insert_text: 'line3' },
        context,
      );

      // Verify insertion
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line1\nline2\nline3');
    });

    it('should throw error for invalid line number (negative)', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'content');

      await expect(
        operations.insert(
          { path: '/memories/test.txt', insert_line: -1, insert_text: 'text' },
          context,
        ),
      ).rejects.toThrow('Invalid insert_line');
    });

    it('should throw error for invalid line number (too large)', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2');

      await expect(
        operations.insert(
          { path: '/memories/test.txt', insert_line: 10, insert_text: 'text' },
          context,
        ),
      ).rejects.toThrow('Invalid insert_line');
    });

    it('should throw error for non-existent file', async () => {
      await expect(
        operations.insert(
          { path: '/memories/nonexistent.txt', insert_line: 0, insert_text: 'text' },
          context,
        ),
      ).rejects.toThrow('File not found');
    });
  });

  describe('delete command', () => {
    it('should delete a file', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'delete-me.txt'), 'content');

      // Delete file
      const result = await operations.deleteOp({ path: '/memories/delete-me.txt' }, context);
      expect(result).toBe('File deleted: /memories/delete-me.txt');

      // Verify file is gone
      await expect(fs.access(path.join(memoryRoot, 'delete-me.txt'))).rejects.toThrow();
    });

    it('should delete a directory recursively', async () => {
      // Create test directory with contents
      await fs.mkdir(path.join(memoryRoot, 'delete-dir/sub'), { recursive: true });
      await fs.writeFile(path.join(memoryRoot, 'delete-dir/file.txt'), 'content');
      await fs.writeFile(path.join(memoryRoot, 'delete-dir/sub/file2.txt'), 'content');

      // Delete directory
      const result = await operations.deleteOp({ path: '/memories/delete-dir' }, context);
      expect(result).toBe('Directory deleted: /memories/delete-dir');

      // Verify directory is gone
      await expect(fs.access(path.join(memoryRoot, 'delete-dir'))).rejects.toThrow();
    });

    it('should throw error when deleting /memories root', async () => {
      await expect(operations.deleteOp({ path: '/memories' }, context)).rejects.toThrow(
        'Cannot delete the /memories directory itself',
      );
    });

    it('should throw error for non-existent path', async () => {
      await expect(operations.deleteOp({ path: '/memories/nonexistent.txt' }, context)).rejects.toThrow(
        'Path not found',
      );
    });
  });

  describe('rename command', () => {
    it('should rename a file', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'old-name.txt'), 'content');

      // Rename file
      const result = await operations.rename(
        { old_path: '/memories/old-name.txt', new_path: '/memories/new-name.txt' },
        context,
      );
      expect(result).toBe('Renamed /memories/old-name.txt to /memories/new-name.txt');

      // Verify old name is gone
      await expect(fs.access(path.join(memoryRoot, 'old-name.txt'))).rejects.toThrow();

      // Verify new name exists
      const content = await fs.readFile(path.join(memoryRoot, 'new-name.txt'), 'utf-8');
      expect(content).toBe('content');
    });

    it('should move file to subdirectory', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'file.txt'), 'content');
      await fs.mkdir(path.join(memoryRoot, 'subdir'));

      // Move file
      await operations.rename(
        { old_path: '/memories/file.txt', new_path: '/memories/subdir/file.txt' },
        context,
      );

      // Verify file moved
      await expect(fs.access(path.join(memoryRoot, 'file.txt'))).rejects.toThrow();
      const content = await fs.readFile(path.join(memoryRoot, 'subdir/file.txt'), 'utf-8');
      expect(content).toBe('content');
    });

    it('should rename a directory', async () => {
      // Create test directory with contents
      await fs.mkdir(path.join(memoryRoot, 'old-dir'));
      await fs.writeFile(path.join(memoryRoot, 'old-dir/file.txt'), 'content');

      // Rename directory
      await operations.rename(
        { old_path: '/memories/old-dir', new_path: '/memories/new-dir' },
        context,
      );

      // Verify directory renamed
      await expect(fs.access(path.join(memoryRoot, 'old-dir'))).rejects.toThrow();
      const content = await fs.readFile(path.join(memoryRoot, 'new-dir/file.txt'), 'utf-8');
      expect(content).toBe('content');
    });

    it('should create parent directories if needed', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'file.txt'), 'content');

      // Move to non-existent subdirectory
      await operations.rename(
        { old_path: '/memories/file.txt', new_path: '/memories/new/sub/dir/file.txt' },
        context,
      );

      // Verify file moved and parent dirs created
      const content = await fs.readFile(path.join(memoryRoot, 'new/sub/dir/file.txt'), 'utf-8');
      expect(content).toBe('content');
    });

    it('should throw error for non-existent source', async () => {
      await expect(
        operations.rename(
          { old_path: '/memories/nonexistent.txt', new_path: '/memories/new.txt' },
          context,
        ),
      ).rejects.toThrow('Source path not found');
    });

    it('should throw error if destination exists', async () => {
      // Create two test files
      await fs.writeFile(path.join(memoryRoot, 'file1.txt'), 'content1');
      await fs.writeFile(path.join(memoryRoot, 'file2.txt'), 'content2');

      // Attempt to rename to existing file
      await expect(
        operations.rename(
          { old_path: '/memories/file1.txt', new_path: '/memories/file2.txt' },
          context,
        ),
      ).rejects.toThrow('Destination already exists');
    });
  });
});
