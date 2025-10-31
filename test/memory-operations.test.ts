/**
 * Memory Operations Tests
 *
 * Comprehensive tests for all 6 memory commands with various inputs and edge cases.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as operations from '../src/memory/operations.js';
import {
  clearAllCachedChecksums,
  getCachedChecksum,
  computeChecksum,
  setCachedChecksum,
  getChecksumCacheStats,
} from '../src/memory/checksums.js';
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
    // Clear checksum cache to prevent cross-test contamination
    clearAllCachedChecksums();

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
      expect(result).toBe('Directory is empty.');
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

    it('should view empty file with friendly message', async () => {
      // Create empty file
      await fs.writeFile(path.join(memoryRoot, 'empty.txt'), '');

      const result = await operations.view({ path: '/memories/empty.txt' }, context);
      expect(result).toBe('Memory file is empty.');
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

    it('should fail when file already exists', async () => {
      // Create initial file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'original');

      // Attempt to create file that already exists should fail
      await expect(
        operations.create({ path: '/memories/test.txt', file_text: 'updated' }, context)
      ).rejects.toThrow('File already exists at /memories/test.txt');

      // Verify original content was not modified
      const fileContent = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(fileContent).toBe('original');
    });

    it('should create empty file with file_text=""', async () => {
      const result = await operations.create({ path: '/memories/empty.txt', file_text: '' }, context);

      expect(result).toBe('Created empty memory file.');

      // Verify empty file exists
      const fileContent = await fs.readFile(path.join(memoryRoot, 'empty.txt'), 'utf-8');
      expect(fileContent).toBe('');
    });

    it('should create empty file without file_text parameter', async () => {
      const result = await operations.create({ path: '/memories/empty2.txt' }, context);

      expect(result).toBe('Created empty memory file.');

      // Verify empty file exists
      const fileContent = await fs.readFile(path.join(memoryRoot, 'empty2.txt'), 'utf-8');
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

  describe('str_replace command - forgiving parameter naming', () => {
    it('should accept old_string and new_string parameters', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Original text\nSecond line');

      // Replace using old_string/new_string variant
      await operations.str_replace(
        {
          path: '/memories/test.txt',
          old_string: 'Original text',
          new_string: 'Modified text',
        },
        context,
      );

      // Verify replacement
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('Modified text\nSecond line');
    });

    it('should accept mixed old_str and new_string parameters', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Line one\nLine two');

      // Replace using mixed naming
      await operations.str_replace(
        {
          path: '/memories/test.txt',
          old_str: 'Line one',
          new_string: 'First line',
        },
        context,
      );

      // Verify replacement
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('First line\nLine two');
    });

    it('should accept mixed old_string and new_str parameters', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Alpha\nBeta');

      // Replace using mixed naming (reverse combination)
      await operations.str_replace(
        {
          path: '/memories/test.txt',
          old_string: 'Alpha',
          new_str: 'Gamma',
        },
        context,
      );

      // Verify replacement
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('Gamma\nBeta');
    });

    it('should throw error when both old_str and old_string are provided', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Some text');

      // Attempt to use both old_str and old_string
      await expect(
        operations.str_replace(
          {
            path: '/memories/test.txt',
            old_str: 'Some text',
            old_string: 'Some text',
            new_str: 'New text',
          },
          context,
        ),
      ).rejects.toThrow('Cannot provide both old_str and old_string');
    });

    it('should throw error when both new_str and new_string are provided', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Some text');

      // Attempt to use both new_str and new_string
      await expect(
        operations.str_replace(
          {
            path: '/memories/test.txt',
            old_str: 'Some text',
            new_str: 'New text',
            new_string: 'Another text',
          },
          context,
        ),
      ).rejects.toThrow('Cannot provide both new_str and new_string');
    });

    it('should throw error when neither old_str nor old_string is provided', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Some text');

      // Attempt without old parameter
      await expect(
        operations.str_replace(
          {
            path: '/memories/test.txt',
            new_str: 'New text',
          },
          context,
        ),
      ).rejects.toThrow('Must provide either old_str or old_string');
    });

    it('should delete text when new_str is omitted (defaults to empty)', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Keep this old text here');

      // Omit new_str - should default to empty string (deletion)
      const result = await operations.str_replace(
        {
          path: '/memories/test.txt',
          old_str: 'old text ',
        },
        context,
      );

      expect(result).toContain('has been edited');
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('Keep this here');
    });
  });

  describe('insert command', () => {
    it('should insert text at line 1 (beginning)', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line2\nline3');

      // Insert at beginning (line 1)
      await operations.insert(
        { path: '/memories/test.txt', insert_line: 1, insert_text: 'line1' },
        context,
      );

      // Verify insertion
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line1\nline2\nline3');
    });

    it('should insert text in middle', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline3');

      // Insert in middle (line 2)
      await operations.insert(
        { path: '/memories/test.txt', insert_line: 2, insert_text: 'line2' },
        context,
      );

      // Verify insertion
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line1\nline2\nline3');
    });

    it('should insert text at end', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2');

      // Insert at end (line 3)
      await operations.insert(
        { path: '/memories/test.txt', insert_line: 3, insert_text: 'line3' },
        context,
      );

      // Verify insertion
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line1\nline2\nline3');
    });

    it('should throw error for invalid line number (zero)', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'content');

      await expect(
        operations.insert(
          { path: '/memories/test.txt', insert_line: 0, insert_text: 'text' },
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
          { path: '/memories/nonexistent.txt', insert_line: 1, insert_text: 'text' },
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

    it('should delete a line from middle of file', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2\nline3');

      // Delete line 2
      const result = await operations.deleteOp(
        { path: '/memories/test.txt', delete_line: 2 },
        context,
      );
      expect(result).toBe('Line 2 deleted from /memories/test.txt');

      // Verify line deleted
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line1\nline3');
    });

    it('should delete first line', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2\nline3');

      // Delete line 1
      await operations.deleteOp({ path: '/memories/test.txt', delete_line: 1 }, context);

      // Verify first line deleted
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line2\nline3');
    });

    it('should delete last line', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2\nline3');

      // Delete line 3
      await operations.deleteOp({ path: '/memories/test.txt', delete_line: 3 }, context);

      // Verify last line deleted
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line1\nline2');
    });

    it('should delete from single-line file resulting in empty file', async () => {
      // Create single-line file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'only line');

      // Delete the only line
      await operations.deleteOp({ path: '/memories/test.txt', delete_line: 1 }, context);

      // Verify file is now empty
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('');
    });

    it('should throw error for delete_line out of range (too high)', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2');

      await expect(
        operations.deleteOp({ path: '/memories/test.txt', delete_line: 10 }, context),
      ).rejects.toThrow('Invalid delete_line');
    });

    it('should throw error for delete_line less than 1', async () => {
      // Create test file
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'line1\nline2');

      await expect(
        operations.deleteOp({ path: '/memories/test.txt', delete_line: 0 }, context),
      ).rejects.toThrow('Invalid delete_line');
    });

    it('should throw error for delete_line on directory', async () => {
      // Create test directory
      await fs.mkdir(path.join(memoryRoot, 'test-dir'));

      await expect(
        operations.deleteOp({ path: '/memories/test-dir', delete_line: 1 }, context),
      ).rejects.toThrow('Cannot delete line from directory');
    });

    it('should throw error for delete_line on non-existent file', async () => {
      await expect(
        operations.deleteOp({ path: '/memories/nonexistent.txt', delete_line: 1 }, context),
      ).rejects.toThrow('File not found');
    });

    it('should clear cache entries for directory children when deleting directory', async () => {
      // Create directory with files
      await fs.mkdir(path.join(memoryRoot, 'testdir'));
      const child1 = path.join(memoryRoot, 'testdir/child1.txt');
      const child2 = path.join(memoryRoot, 'testdir/child2.txt');
      await fs.writeFile(child1, 'content1', 'utf-8');
      await fs.writeFile(child2, 'content2', 'utf-8');

      // Cache children (simulating previous reads)
      setCachedChecksum(child1, computeChecksum('content1'));
      setCachedChecksum(child2, computeChecksum('content2'));

      const statsBefore = getChecksumCacheStats();
      expect(statsBefore.size).toBeGreaterThanOrEqual(2);

      // Delete parent directory
      await operations.deleteOp({ path: '/memories/testdir' }, context);

      // Child cache entries should be cleared
      expect(getCachedChecksum(child1)).toBeUndefined();
      expect(getCachedChecksum(child2)).toBeUndefined();

      const statsAfter = getChecksumCacheStats();
      expect(statsAfter.size).toBeLessThan(statsBefore.size);
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

  // Parameter combinations tests
  describe('create with optional file_text', () => {
    it('should create empty file when file_text omitted', async () => {
      const result = await operations.create({ path: '/memories/empty.txt' }, context);

      expect(result).toBe('Created empty memory file.');
      const content = await fs.readFile(path.join(memoryRoot, 'empty.txt'), 'utf-8');
      expect(content).toBe('');
    });

    it('should create file with content when file_text provided', async () => {
      const result = await operations.create(
        { path: '/memories/data.txt', file_text: 'content' },
        context,
      );

      expect(result).toBe('File created successfully at /memories/data.txt');
      const content = await fs.readFile(path.join(memoryRoot, 'data.txt'), 'utf-8');
      expect(content).toBe('content');
    });

    it('should create empty file when file_text is empty string', async () => {
      const result = await operations.create(
        { path: '/memories/empty2.txt', file_text: '' },
        context,
      );

      expect(result).toBe('Created empty memory file.');
      const content = await fs.readFile(path.join(memoryRoot, 'empty2.txt'), 'utf-8');
      expect(content).toBe('');
    });
  });

  describe('insert with optional insert_line', () => {
    it('should append to end when insert_line omitted', async () => {
      // Create file with content
      await fs.writeFile(
        path.join(memoryRoot, 'test.txt'),
        'line 1\nline 2\nline 3',
      );

      const result = await operations.insert(
        {
          path: '/memories/test.txt',
          insert_text: 'appended line',
        },
        context,
      );

      expect(result).toContain('appended to end');
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line 1\nline 2\nline 3\nappended line');
    });

    it('should insert at specific line when insert_line provided', async () => {
      // Create file with content
      await fs.writeFile(
        path.join(memoryRoot, 'test.txt'),
        'line 1\nline 2\nline 3',
      );

      const result = await operations.insert(
        {
          path: '/memories/test.txt',
          insert_line: 2,
          insert_text: 'inserted line',
        },
        context,
      );

      expect(result).toContain('inserted at line 2');
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('line 1\ninserted line\nline 2\nline 3');
    });

    it('should append to empty file when insert_line omitted', async () => {
      // Create empty file
      await fs.writeFile(path.join(memoryRoot, 'empty.txt'), '');

      const result = await operations.insert(
        {
          path: '/memories/empty.txt',
          insert_text: 'first line',
        },
        context,
      );

      expect(result).toContain('appended to end');
      const content = await fs.readFile(path.join(memoryRoot, 'empty.txt'), 'utf-8');
      expect(content).toBe('first line');
    });
  });

  describe('delete with old_str (content-based deletion)', () => {
    it('should delete unique text and remove empty line if created', async () => {
      await fs.writeFile(
        path.join(memoryRoot, 'test.txt'),
        'debug code\nThis is line 2\nLine 3',
      );

      const result = await operations.deleteOp(
        { path: '/memories/test.txt', old_str: 'debug code' },
        context,
      );

      expect(result).toContain('1 occurrence(s)');
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('This is line 2\nLine 3');
    });

    it('should delete text from middle of line', async () => {
      await fs.writeFile(
        path.join(memoryRoot, 'test.txt'),
        'Start debug code end\nNormal line\nLine 3',
      );

      const result = await operations.deleteOp(
        { path: '/memories/test.txt', old_str: 'debug code ' },
        context,
      );

      expect(result).toContain('1 occurrence(s)');
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('Start end\nNormal line\nLine 3');
    });

    it('should error when text appears multiple times', async () => {
      await fs.writeFile(
        path.join(memoryRoot, 'test.txt'),
        'debug code\nSome line\ndebug code again',
      );

      await expect(
        operations.deleteOp({ path: '/memories/test.txt', old_str: 'debug code' }, context),
      ).rejects.toThrow('Text appears 2 times');
    });

    it('should error when text not found', async () => {
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Some content');

      await expect(
        operations.deleteOp({ path: '/memories/test.txt', old_str: 'nonexistent' }, context),
      ).rejects.toThrow('Text not found');
    });

    it('should error when using both delete_line and old_str', async () => {
      await fs.writeFile(path.join(memoryRoot, 'test.txt'), 'Some content');

      await expect(
        operations.deleteOp(
          {
            path: '/memories/test.txt',
            delete_line: 1,
            old_str: 'content',
          },
          context,
        ),
      ).rejects.toThrow('Cannot use both delete_line and old_str');
    });

    it('should accept old_string as alternative parameter name', async () => {
      await fs.writeFile(
        path.join(memoryRoot, 'test.txt'),
        'Text to delete\nKeep this',
      );

      const result = await operations.deleteOp(
        { path: '/memories/test.txt', old_string: 'Text to delete' },
        context,
      );

      expect(result).toContain('1 occurrence(s)');
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('Keep this');
    });

    it('should handle special regex characters in search text', async () => {
      await fs.writeFile(
        path.join(memoryRoot, 'test.txt'),
        'Price: $100.00 total',
      );

      const result = await operations.deleteOp(
        { path: '/memories/test.txt', old_str: '$100.00' },
        context,
      );

      expect(result).toContain('1 occurrence(s)');
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('Price:  total');
    });
  });

  describe('str_replace with empty string deletion', () => {
    it('should delete text when new_str is empty', async () => {
      await fs.writeFile(
        path.join(memoryRoot, 'test.txt'),
        'Keep this debug code and this',
      );

      const result = await operations.str_replace(
        {
          path: '/memories/test.txt',
          old_str: 'debug code ',
          new_str: '',
        },
        context,
      );

      expect(result).toContain('has been edited');
      const content = await fs.readFile(path.join(memoryRoot, 'test.txt'), 'utf-8');
      expect(content).toBe('Keep this and this');
    });
  });

  describe('Checksum Caching Behavior', () => {
    it('should cache checksum after view (full file read)', async () => {
      // Create file
      const testFile = path.join(memoryRoot, 'test.txt');
      const content = 'Hello, World!';
      await fs.writeFile(testFile, content);

      // View file (full read)
      await operations.view({ path: '/memories/test.txt' }, context);

      // Checksum should be cached
      const cached = getCachedChecksum(testFile);
      expect(cached).toBe(computeChecksum(content));
    });

    it('should NOT cache checksum after partial view (with view_range)', async () => {
      // Create file
      const testFile = path.join(memoryRoot, 'test.txt');
      await fs.writeFile(testFile, 'line1\nline2\nline3');

      // View with range (partial read)
      await operations.view({ path: '/memories/test.txt', view_range: [1, 2] }, context);

      // Checksum should NOT be cached
      const cached = getCachedChecksum(testFile);
      expect(cached).toBeUndefined();
    });

    it('should cache checksum after create', async () => {
      // Create file
      const content = 'New file content';
      await operations.create(
        { path: '/memories/new.txt', file_text: content },
        context,
      );

      // Checksum should be cached
      const testFile = path.join(memoryRoot, 'new.txt');
      const cached = getCachedChecksum(testFile);
      expect(cached).toBe(computeChecksum(content));
    });

    it('should cache checksum after str_replace', async () => {
      // Create file
      const testFile = path.join(memoryRoot, 'test.txt');
      await fs.writeFile(testFile, 'Hello, World!');

      // str_replace
      await operations.str_replace(
        { path: '/memories/test.txt', old_str: 'World', new_str: 'Universe' },
        context,
      );

      // Checksum should be cached (new content)
      const newContent = 'Hello, Universe!';
      const cached = getCachedChecksum(testFile);
      expect(cached).toBe(computeChecksum(newContent));
    });

    it('should cache checksum after insert', async () => {
      // Create file
      const testFile = path.join(memoryRoot, 'test.txt');
      await fs.writeFile(testFile, 'line1\nline3');

      // Insert
      await operations.insert(
        { path: '/memories/test.txt', insert_line: 2, insert_text: 'line2' },
        context,
      );

      // Checksum should be cached (new content)
      const newContent = 'line1\nline2\nline3';
      const cached = getCachedChecksum(testFile);
      expect(cached).toBe(computeChecksum(newContent));
    });

    it('should clear checksum after file deletion', async () => {
      // Create and cache file
      const testFile = path.join(memoryRoot, 'delete-me.txt');
      await fs.writeFile(testFile, 'content');
      setCachedChecksum(testFile, computeChecksum('content'));

      // Delete file
      await operations.deleteOp({ path: '/memories/delete-me.txt' }, context);

      // Checksum should be cleared
      const cached = getCachedChecksum(testFile);
      expect(cached).toBeUndefined();
    });

    it('should clear checksum after rename (old path)', async () => {
      // Create and cache file
      const oldFile = path.join(memoryRoot, 'old.txt');
      const newFile = path.join(memoryRoot, 'new.txt');
      await fs.writeFile(oldFile, 'content');
      setCachedChecksum(oldFile, computeChecksum('content'));

      // Rename
      await operations.rename(
        { old_path: '/memories/old.txt', new_path: '/memories/new.txt' },
        context,
      );

      // Old path checksum should be cleared
      const cachedOld = getCachedChecksum(oldFile);
      expect(cachedOld).toBeUndefined();

      // New path won't have checksum yet (will be cached on next read)
      const cachedNew = getCachedChecksum(newFile);
      expect(cachedNew).toBeUndefined();
    });

    it('should detect sequential modifications and throw error', async () => {
      // Create file and cache it
      const testFile = path.join(memoryRoot, 'test.txt');
      const originalContent = 'TODO: Buy milk';
      await fs.writeFile(testFile, originalContent);

      // Simulate previous read caching the checksum
      setCachedChecksum(testFile, computeChecksum(originalContent));

      // External process modifies file
      await fs.writeFile(testFile, 'TODO: Buy eggs');

      // Try to str_replace - should detect modification
      await expect(
        operations.str_replace(
          { path: '/memories/test.txt', old_str: 'milk', new_str: 'bread' },
          context,
        ),
      ).rejects.toThrow('File has been modified by another process');
    });

    it('should show current contents in sequential modification error', async () => {
      // Create file and cache it
      const testFile = path.join(memoryRoot, 'notes.txt');
      await fs.writeFile(testFile, 'Original notes');
      setCachedChecksum(testFile, computeChecksum('Original notes'));

      // External modification
      await fs.writeFile(testFile, 'Modified by another process');

      // Try to modify
      try {
        await operations.str_replace(
          { path: '/memories/notes.txt', old_str: 'Original', new_str: 'Updated' },
          context,
        );
        fail('Should have thrown error');
      } catch (err) {
        const error = err as Error;
        expect(error.message).toContain('Modified by another process');
        expect(error.message).toContain('Current contents of');
      }
    });

    it('should not mask concurrent modifications when str_replace fails for other reasons', async () => {
      // Verify concurrent modification detection happens BEFORE text validation
      const testFile = path.join(memoryRoot, 'test.txt');
      const originalContent = 'TODO: Buy milk';
      await fs.writeFile(testFile, originalContent, 'utf-8');

      // Establish cached checksum
      const originalChecksum = computeChecksum(originalContent);
      setCachedChecksum(testFile, originalChecksum);

      // External process modifies file
      const modifiedContent = 'TODO: Buy eggs';
      await fs.writeFile(testFile, modifiedContent, 'utf-8');

      // Try operation - concurrent modification detected BEFORE text matching
      await expect(
        operations.str_replace(
          { path: '/memories/test.txt', old_str: 'bread', new_str: 'cookies' },
          context,
        ),
      ).rejects.toThrow('File has been modified by another process');

      // Verify cache wasn't corrupted by failed operation
      const currentChecksum = getCachedChecksum(testFile);
      expect(currentChecksum).toBe(originalChecksum); // Should still be original

      // After re-reading file, operation should succeed with correct text
      await operations.view({ path: '/memories/test.txt' }, context); // Updates cache
      await expect(
        operations.str_replace(
          { path: '/memories/test.txt', old_str: 'eggs', new_str: 'bread' },
          context,
        ),
      ).resolves.toBeDefined();
    });

    it('should not mask concurrent modifications when insert fails for other reasons', async () => {
      // Verify concurrent modification detection happens BEFORE line validation
      const testFile = path.join(memoryRoot, 'test.txt');
      const originalContent = 'Line 1\nLine 2';
      await fs.writeFile(testFile, originalContent, 'utf-8');

      setCachedChecksum(testFile, computeChecksum(originalContent));

      // External process modifies file
      const modifiedContent = 'Line 1\nLine 2\nLine 3';
      await fs.writeFile(testFile, modifiedContent, 'utf-8');

      // Try insert - concurrent modification detected BEFORE line range validation
      await expect(
        operations.insert(
          { path: '/memories/test.txt', insert_line: 100, insert_text: 'New' },
          context,
        ),
      ).rejects.toThrow('File has been modified by another process');

      // After re-reading, operation succeeds with valid line
      await operations.view({ path: '/memories/test.txt' }, context);
      await expect(
        operations.insert(
          { path: '/memories/test.txt', insert_line: 2, insert_text: 'New' },
          context,
        ),
      ).resolves.toBeDefined();
    });

    it('should not mask concurrent modifications when delete fails for other reasons', async () => {
      // Verify concurrent modification detection happens BEFORE text matching
      const testFile = path.join(memoryRoot, 'test.txt');
      const originalContent = 'TODO: Buy milk';
      await fs.writeFile(testFile, originalContent, 'utf-8');

      setCachedChecksum(testFile, computeChecksum(originalContent));

      // External process modifies file
      const modifiedContent = 'TODO: Buy eggs';
      await fs.writeFile(testFile, modifiedContent, 'utf-8');

      // Try delete - concurrent modification detected BEFORE text matching
      await expect(
        operations.deleteOp({ path: '/memories/test.txt', old_str: 'bread' }, context),
      ).rejects.toThrow('File has been modified by another process');

      // After re-reading, operation succeeds with correct text
      await operations.view({ path: '/memories/test.txt' }, context);
      await expect(
        operations.deleteOp({ path: '/memories/test.txt', old_str: 'eggs' }, context),
      ).resolves.toBeDefined();
    });
  });
});
