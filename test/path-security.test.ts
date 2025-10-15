/**
 * Path Security Tests
 *
 * Tests for directory traversal protection and path validation.
 * CRITICAL: These tests ensure attackers cannot escape the memory directory.
 */

import { describe, it, expect } from '@jest/globals';
import { validatePath, toMemoryPath } from '../src/memory/path-security.js';
import * as path from 'path';

describe('Path Security - validatePath', () => {
  const memoryRoot = '/tmp/test-memory';

  describe('Valid paths', () => {
    it('should accept /memories root path', () => {
      const result = validatePath('/memories', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot));
    });

    it('should accept /memories/file.txt', () => {
      const result = validatePath('/memories/file.txt', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot, 'file.txt'));
    });

    it('should accept /memories/subdir/file.txt', () => {
      const result = validatePath('/memories/subdir/file.txt', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot, 'subdir/file.txt'));
    });

    it('should accept /memories/deep/nested/path/file.txt', () => {
      const result = validatePath('/memories/deep/nested/path/file.txt', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot, 'deep/nested/path/file.txt'));
    });
  });

  describe('Invalid prefix', () => {
    it('should reject path not starting with /memories', () => {
      expect(() => validatePath('/other/path.txt', memoryRoot)).toThrow(
        'Path must start with /memories',
      );
    });

    it('should reject empty path', () => {
      expect(() => validatePath('', memoryRoot)).toThrow('Path must start with /memories');
    });

    it('should reject /memory (singular)', () => {
      expect(() => validatePath('/memory/file.txt', memoryRoot)).toThrow(
        'Path must start with /memories',
      );
    });

    it('should reject path starting with memories but no slash', () => {
      expect(() => validatePath('memories/file.txt', memoryRoot)).toThrow(
        'Path must start with /memories',
      );
    });
  });

  describe('Directory traversal attacks', () => {
    it('should reject ../ traversal', () => {
      expect(() => validatePath('/memories/../etc/passwd', memoryRoot)).toThrow(
        'would escape /memories directory',
      );
    });

    it('should reject multiple ../ traversals', () => {
      expect(() => validatePath('/memories/../../etc/passwd', memoryRoot)).toThrow(
        'would escape /memories directory',
      );
    });

    it('should reject ../ in middle of path', () => {
      expect(() => validatePath('/memories/sub/../../../etc/passwd', memoryRoot)).toThrow(
        'would escape /memories directory',
      );
    });

    it('should handle Windows-style backslashes as literal characters on Unix', () => {
      // On Unix systems, backslashes are valid filename characters, not path separators
      // This path would create a file literally named "..\..\etc\passwd"
      // On Windows, this would need additional validation, but MCP servers run on Unix
      const result = validatePath('/memories/..\\..\\etc\\passwd', memoryRoot);
      // The backslashes are treated as part of the filename
      expect(result).toBe(path.resolve(memoryRoot, '..\\..\\etc\\passwd'));
    });

    it('should reject URL-encoded traversal %2e%2e%2f', () => {
      // URL decode happens before this function in a real scenario,
      // but test the decoded form
      const decoded = decodeURIComponent('/memories/%2e%2e%2f%2e%2e%2fetc/passwd');
      expect(() => validatePath(decoded, memoryRoot)).toThrow(
        'would escape /memories directory',
      );
    });

    it('should reject double-encoded traversal', () => {
      const doubleDecoded = decodeURIComponent(
        decodeURIComponent('/memories/%252e%252e%252f%252e%252e%252fetc/passwd'),
      );
      expect(() => validatePath(doubleDecoded, memoryRoot)).toThrow(
        'would escape /memories directory',
      );
    });

    it('should reject absolute path escape attempt', () => {
      expect(() => validatePath('/memories/../../../../../etc/passwd', memoryRoot)).toThrow(
        'would escape /memories directory',
      );
    });

    it('should reject path with . and ..', () => {
      expect(() => validatePath('/memories/./../../etc/passwd', memoryRoot)).toThrow(
        'would escape /memories directory',
      );
    });
  });

  describe('Edge cases', () => {
    it('should allow ./ (current directory reference)', () => {
      const result = validatePath('/memories/./file.txt', memoryRoot);
      // After path.resolve, ./ is normalized away
      expect(result).toBe(path.resolve(memoryRoot, 'file.txt'));
    });

    it('should allow subdirectory named "backup" (not "..backup")', () => {
      const result = validatePath('/memories/backup/file.txt', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot, 'backup/file.txt'));
    });

    it('should allow file with dots in name', () => {
      const result = validatePath('/memories/file.tar.gz', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot, 'file.tar.gz'));
    });

    it('should allow hidden files (starting with dot)', () => {
      const result = validatePath('/memories/.hidden', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot, '.hidden'));
    });

    it('should handle trailing slash', () => {
      const result = validatePath('/memories/subdir/', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot, 'subdir'));
    });

    it('should handle multiple slashes', () => {
      const result = validatePath('/memories//file.txt', memoryRoot);
      expect(result).toBe(path.resolve(memoryRoot, 'file.txt'));
    });
  });
});

describe('Path Security - toMemoryPath', () => {
  const memoryRoot = '/tmp/test-memory';

  describe('Valid conversions', () => {
    it('should convert memory root to /memories', () => {
      const result = toMemoryPath(memoryRoot, memoryRoot);
      expect(result).toBe('/memories');
    });

    it('should convert file path to memory path', () => {
      const fsPath = path.join(memoryRoot, 'file.txt');
      const result = toMemoryPath(fsPath, memoryRoot);
      expect(result).toBe('/memories/file.txt');
    });

    it('should convert nested path to memory path', () => {
      const fsPath = path.join(memoryRoot, 'subdir/file.txt');
      const result = toMemoryPath(fsPath, memoryRoot);
      expect(result).toBe('/memories/subdir/file.txt');
    });
  });

  describe('Invalid conversions', () => {
    it('should reject path outside memory root', () => {
      expect(() => toMemoryPath('/etc/passwd', memoryRoot)).toThrow(
        'is not within memory root',
      );
    });

    it('should reject path in different root', () => {
      expect(() => toMemoryPath('/var/log/test.log', memoryRoot)).toThrow(
        'is not within memory root',
      );
    });
  });
});
