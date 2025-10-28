/**
 * Tree View Module Tests
 *
 * Tests for tree view rendering functions including:
 * - File size formatting
 * - Modification date formatting
 * - Line counting
 * - Tree structure rendering
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import {
  formatFileSize,
  formatModificationDate,
  countFileLines,
  renderDirectoryTree,
} from '../src/memory/tree-view.js';

describe('Tree View Module', () => {
  describe('formatFileSize', () => {
    it('should format bytes correctly', () => {
      expect(formatFileSize(0)).toBe('0B');
      expect(formatFileSize(500)).toBe('500B');
      expect(formatFileSize(1023)).toBe('1023B');
    });

    it('should format kilobytes correctly', () => {
      expect(formatFileSize(1024)).toBe('1KB');
      expect(formatFileSize(1536)).toBe('1.5KB');
      expect(formatFileSize(2048)).toBe('2KB');
      expect(formatFileSize(10240)).toBe('10KB');
    });

    it('should format megabytes correctly', () => {
      expect(formatFileSize(1024 * 1024)).toBe('1MB');
      expect(formatFileSize(1024 * 1024 * 1.5)).toBe('1.5MB');
      expect(formatFileSize(1024 * 1024 * 10)).toBe('10MB');
    });

    it('should format gigabytes correctly', () => {
      expect(formatFileSize(1024 * 1024 * 1024)).toBe('1GB');
      expect(formatFileSize(1024 * 1024 * 1024 * 2.3)).toBe('2.3GB');
    });

    it('should format terabytes correctly', () => {
      expect(formatFileSize(1024 * 1024 * 1024 * 1024)).toBe('1TB');
      expect(formatFileSize(1024 * 1024 * 1024 * 1024 * 1.7)).toBe('1.7TB');
    });
  });

  describe('formatModificationDate', () => {
    it('should format Date objects correctly', () => {
      const date = new Date('2025-10-15T14:23:17.000Z');
      const formatted = formatModificationDate(date);

      // Should match [YYYY/MM/DD - HH:MM:SS] format
      expect(formatted).toMatch(/^\[\d{4}\/\d{2}\/\d{2} - \d{2}:\d{2}:\d{2}\]$/);
    });

    it('should format timestamps correctly', () => {
      const timestamp = new Date('2025-10-15T14:23:17.000Z').getTime();
      const formatted = formatModificationDate(timestamp);

      expect(formatted).toMatch(/^\[\d{4}\/\d{2}\/\d{2} - \d{2}:\d{2}:\d{2}\]$/);
    });

    it('should return empty string for invalid dates', () => {
      expect(formatModificationDate(NaN)).toBe('');
      expect(formatModificationDate(new Date('invalid'))).toBe('');
    });

    it('should pad single-digit months, days, hours, minutes, seconds', () => {
      // January 5, 2025 at 03:05:07
      const date = new Date('2025-01-05T03:05:07.000Z');
      const formatted = formatModificationDate(date);

      // Check that padding exists (will vary by timezone)
      expect(formatted).toMatch(/\[\d{4}\/0\d\/0\d - \d{2}:\d{2}:\d{2}\]/);
    });
  });

  describe('countFileLines', () => {
    let testDir: string;

    beforeEach(async () => {
      // Create temporary test directory
      testDir = await fs.mkdtemp(path.join(os.tmpdir(), 'tree-view-test-'));
    });

    afterEach(async () => {
      // Clean up test directory
      await fs.rm(testDir, { recursive: true, force: true });
    });

    it('should count lines in files with Unix line endings', async () => {
      const filePath = path.join(testDir, 'unix.txt');
      await fs.writeFile(filePath, 'line1\nline2\nline3\n', 'utf-8');

      const count = await countFileLines(filePath);
      expect(count).toBe(3);
    });

    it('should count lines in files without trailing newline', async () => {
      const filePath = path.join(testDir, 'no-trailing.txt');
      await fs.writeFile(filePath, 'line1\nline2\nline3', 'utf-8');

      const count = await countFileLines(filePath);
      expect(count).toBe(3);
    });

    it('should count lines in files with Windows line endings', async () => {
      const filePath = path.join(testDir, 'windows.txt');
      await fs.writeFile(filePath, 'line1\r\nline2\r\nline3\r\n', 'utf-8');

      const count = await countFileLines(filePath);
      // Split by \n will give 4 elements for 3 lines with \r\n endings
      expect(count).toBe(3);
    });

    it('should return 0 for empty files', async () => {
      const filePath = path.join(testDir, 'empty.txt');
      await fs.writeFile(filePath, '', 'utf-8');

      const count = await countFileLines(filePath);
      expect(count).toBe(0);
    });

    it('should return 1 for single-line files', async () => {
      const filePath = path.join(testDir, 'single.txt');
      await fs.writeFile(filePath, 'single line', 'utf-8');

      const count = await countFileLines(filePath);
      expect(count).toBe(1);
    });

    it('should return null for non-existent files', async () => {
      const filePath = path.join(testDir, 'nonexistent.txt');

      const count = await countFileLines(filePath);
      expect(count).toBeNull();
    });
  });

  describe('renderDirectoryTree', () => {
    let testDir: string;

    beforeEach(async () => {
      // Create temporary test directory with structure
      testDir = await fs.mkdtemp(path.join(os.tmpdir(), 'tree-view-test-'));

      // Create test structure:
      // testDir/
      //   file1.txt
      //   file2.txt
      //   subdir/
      //     file3.txt
      //     nested/
      //       file4.txt
      await fs.writeFile(path.join(testDir, 'file1.txt'), 'content1\ncontent2\n', 'utf-8');
      await fs.writeFile(path.join(testDir, 'file2.txt'), 'line1', 'utf-8');
      await fs.mkdir(path.join(testDir, 'subdir'));
      await fs.writeFile(path.join(testDir, 'subdir', 'file3.txt'), 'a\nb\nc\n', 'utf-8');
      await fs.mkdir(path.join(testDir, 'subdir', 'nested'));
      await fs.writeFile(path.join(testDir, 'subdir', 'nested', 'file4.txt'), 'test', 'utf-8');
    });

    afterEach(async () => {
      // Clean up test directory
      await fs.rm(testDir, { recursive: true, force: true });
    });

    it('should render directory tree with hierarchy', async () => {
      const result = await renderDirectoryTree(testDir, '/memories');

      // Check header
      expect(result).toContain('Showing contents of: /memories');
      expect(result).toContain('Modification dates shown in');

      // Check hierarchical structure with tree symbols
      expect(result).toContain('file1.txt');
      expect(result).toContain('file2.txt');
      expect(result).toContain('subdir/');
      expect(result).toContain('file3.txt');
      expect(result).toContain('nested/');
      expect(result).toContain('file4.txt');

      // Check tree symbols are present
      expect(result).toMatch(/[├└]── /);
      expect(result).toMatch(/│   /);
    });

    it('should include file sizes', async () => {
      const result = await renderDirectoryTree(testDir, '/memories');

      // file1.txt has "content1\ncontent2\n" (18 bytes)
      expect(result).toMatch(/file1\.txt.*18B/);

      // file2.txt has "line1" (5 bytes)
      expect(result).toMatch(/file2\.txt.*5B/);
    });

    it('should include line counts', async () => {
      const result = await renderDirectoryTree(testDir, '/memories');

      // file1.txt has 2 lines
      expect(result).toMatch(/file1\.txt.*2 lines/);

      // file2.txt has 1 line
      expect(result).toMatch(/file2\.txt.*1 lines/);

      // file3.txt has 3 lines
      expect(result).toMatch(/file3\.txt.*3 lines/);
    });

    it('should include modification times', async () => {
      const result = await renderDirectoryTree(testDir, '/memories');

      // Should have timestamps in [YYYY/MM/DD - HH:MM:SS] format
      expect(result).toMatch(/\[\d{4}\/\d{2}\/\d{2} - \d{2}:\d{2}:\d{2}\]/);
    });

    it('should skip hidden files', async () => {
      // Create hidden file
      await fs.writeFile(path.join(testDir, '.hidden'), 'secret', 'utf-8');

      const result = await renderDirectoryTree(testDir, '/memories');

      // Hidden file should not appear
      expect(result).not.toContain('.hidden');
    });

    it('should handle deep nesting without limits', async () => {
      // Create deeply nested structure
      let currentDir = testDir;
      for (let i = 0; i < 10; i++) {
        currentDir = path.join(currentDir, `level${i}`);
        await fs.mkdir(currentDir);
        await fs.writeFile(path.join(currentDir, `file${i}.txt`), `level ${i}`, 'utf-8');
      }

      const result = await renderDirectoryTree(testDir, '/memories');

      // Should contain deeply nested items
      expect(result).toContain('level0/');
      expect(result).toContain('level9/');
      expect(result).toContain('file9.txt');
    });

    it('should handle empty directories', async () => {
      // Create empty subdirectory
      await fs.mkdir(path.join(testDir, 'empty'));

      const result = await renderDirectoryTree(testDir, '/memories');

      // Empty directory should appear
      expect(result).toContain('empty/');
    });

    it('should show friendly message for completely empty directory', async () => {
      // Create a new empty directory (not the testDir with files)
      const emptyDir = await fs.mkdtemp(path.join(os.tmpdir(), 'empty-tree-test-'));

      try {
        const result = await renderDirectoryTree(emptyDir, '/memories');

        // Should return friendly message instead of header
        expect(result).toBe('Directory is empty.');
      } finally {
        await fs.rm(emptyDir, { recursive: true, force: true });
      }
    });

    it('should sort directories before files', async () => {
      const result = await renderDirectoryTree(testDir, '/memories');

      // Get lines and find positions
      const lines = result.split('\n');
      const file1Index = lines.findIndex((l) => l.includes('file1.txt'));
      const subdirIndex = lines.findIndex((l) => l.includes('subdir/'));

      // subdir/ should come before file1.txt (directories first)
      expect(subdirIndex).toBeLessThan(file1Index);
    });

    it('should sort items alphabetically within type', async () => {
      const result = await renderDirectoryTree(testDir, '/memories');

      // Get lines
      const lines = result.split('\n');
      const file1Index = lines.findIndex((l) => l.includes('file1.txt'));
      const file2Index = lines.findIndex((l) => l.includes('file2.txt'));

      // file1.txt should come before file2.txt (alphabetical)
      expect(file1Index).toBeLessThan(file2Index);
    });
  });
});
