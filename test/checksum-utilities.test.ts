/**
 * Checksum Utilities Tests
 *
 * Tests for the checksum computation and caching system used for
 * concurrency detection across sequential operations.
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import {
  computeChecksum,
  getCachedChecksum,
  setCachedChecksum,
  clearCachedChecksum,
  clearAllCachedChecksums,
  getChecksumCacheStats,
} from '../src/memory/checksums.js';

describe('Checksum Utilities', () => {
  // Clear cache before each test to prevent contamination
  beforeEach(() => {
    clearAllCachedChecksums();
  });

  describe('computeChecksum', () => {
    it('should compute consistent SHA-256 checksums', () => {
      // Compute checksum for same content multiple times
      const content = 'Hello, World!';
      const checksum1 = computeChecksum(content);
      const checksum2 = computeChecksum(content);

      // Should be identical
      expect(checksum1).toBe(checksum2);
    });

    it('should produce different checksums for different content', () => {
      // Compute checksums for different content
      const checksum1 = computeChecksum('Hello, World!');
      const checksum2 = computeChecksum('Goodbye, World!');
      const checksum3 = computeChecksum('Hello, World! '); // Note trailing space

      // All should be different
      expect(checksum1).not.toBe(checksum2);
      expect(checksum1).not.toBe(checksum3);
      expect(checksum2).not.toBe(checksum3);
    });

    it('should return 64-character hex digest', () => {
      // Compute checksum
      const checksum = computeChecksum('test content');

      // Verify format: 64 hex characters
      expect(checksum).toHaveLength(64);
      expect(checksum).toMatch(/^[a-f0-9]{64}$/);
    });

    it('should handle empty string', () => {
      // Empty string should produce valid checksum
      const checksum = computeChecksum('');

      expect(checksum).toHaveLength(64);
      expect(checksum).toMatch(/^[a-f0-9]{64}$/);
    });

    it('should handle large content', () => {
      // Generate large content (50 KB)
      const largeContent = 'x'.repeat(50 * 1024);

      // Should compute without error
      const checksum = computeChecksum(largeContent);

      expect(checksum).toHaveLength(64);
    });

    it('should handle unicode content', () => {
      // Unicode content
      const unicodeContent = 'Hello 世界 🌍 مرحبا';

      // Should compute without error
      const checksum = computeChecksum(unicodeContent);

      expect(checksum).toHaveLength(64);
    });
  });

  describe('Cache Operations', () => {
    it('should store and retrieve checksums', () => {
      // Store checksum
      const filePath = '/tmp/test/file.txt';
      const checksum = computeChecksum('content');

      setCachedChecksum(filePath, checksum);

      // Retrieve checksum
      const retrieved = getCachedChecksum(filePath);

      expect(retrieved).toBe(checksum);
    });

    it('should return undefined for non-cached paths', () => {
      // Try to get checksum for path that was never cached
      const retrieved = getCachedChecksum('/tmp/never-cached.txt');

      expect(retrieved).toBeUndefined();
    });

    it('should normalize paths before caching', () => {
      // Store checksum with non-normalized path
      const checksum = computeChecksum('content');
      setCachedChecksum('/tmp/./test/../test/file.txt', checksum);

      // Retrieve with normalized path
      const retrieved = getCachedChecksum('/tmp/test/file.txt');

      expect(retrieved).toBe(checksum);
    });

    it('should overwrite existing cached checksums', () => {
      // Store initial checksum
      const filePath = '/tmp/test/file.txt';
      const checksum1 = computeChecksum('content 1');
      const checksum2 = computeChecksum('content 2');

      setCachedChecksum(filePath, checksum1);

      // Overwrite with new checksum
      setCachedChecksum(filePath, checksum2);

      // Should retrieve the new one
      const retrieved = getCachedChecksum(filePath);

      expect(retrieved).toBe(checksum2);
      expect(retrieved).not.toBe(checksum1);
    });

    it('should clear specific cached checksums', () => {
      // Cache two checksums
      const path1 = '/tmp/test/file1.txt';
      const path2 = '/tmp/test/file2.txt';

      setCachedChecksum(path1, computeChecksum('content 1'));
      setCachedChecksum(path2, computeChecksum('content 2'));

      // Clear one
      clearCachedChecksum(path1);

      // path1 should be gone, path2 should remain
      expect(getCachedChecksum(path1)).toBeUndefined();
      expect(getCachedChecksum(path2)).toBeDefined();
    });

    it('should handle clearing non-existent cache entries', () => {
      // Clearing non-existent entry should not throw
      expect(() => {
        clearCachedChecksum('/tmp/never-existed.txt');
      }).not.toThrow();
    });

    it('should clear all cached checksums', () => {
      // Cache multiple checksums
      setCachedChecksum('/tmp/file1.txt', computeChecksum('content 1'));
      setCachedChecksum('/tmp/file2.txt', computeChecksum('content 2'));
      setCachedChecksum('/tmp/file3.txt', computeChecksum('content 3'));

      // Verify they're cached
      expect(getCachedChecksum('/tmp/file1.txt')).toBeDefined();
      expect(getCachedChecksum('/tmp/file2.txt')).toBeDefined();
      expect(getCachedChecksum('/tmp/file3.txt')).toBeDefined();

      // Clear all
      clearAllCachedChecksums();

      // All should be gone
      expect(getCachedChecksum('/tmp/file1.txt')).toBeUndefined();
      expect(getCachedChecksum('/tmp/file2.txt')).toBeUndefined();
      expect(getCachedChecksum('/tmp/file3.txt')).toBeUndefined();
    });
  });

  describe('Cache Statistics', () => {
    it('should report cache size', () => {
      // Empty cache
      let stats = getChecksumCacheStats();
      expect(stats.size).toBe(0);

      // Add entries
      setCachedChecksum('/tmp/file1.txt', computeChecksum('content 1'));
      setCachedChecksum('/tmp/file2.txt', computeChecksum('content 2'));

      stats = getChecksumCacheStats();
      expect(stats.size).toBe(2);

      // Clear one
      clearCachedChecksum('/tmp/file1.txt');

      stats = getChecksumCacheStats();
      expect(stats.size).toBe(1);

      // Clear all
      clearAllCachedChecksums();

      stats = getChecksumCacheStats();
      expect(stats.size).toBe(0);
    });

    it('should estimate memory usage', () => {
      // Empty cache
      let stats = getChecksumCacheStats();
      expect(stats.memoryEstimate).toBe(0);

      // Add entries
      setCachedChecksum('/tmp/file.txt', computeChecksum('content'));

      stats = getChecksumCacheStats();
      // Should estimate ~82 bytes per entry
      expect(stats.memoryEstimate).toBeGreaterThan(0);
    });
  });

  describe('Real-World Scenarios', () => {
    it('should detect content changes via checksum mismatch', () => {
      // Simulate file being read and cached
      const filePath = '/tmp/test/notes.txt';
      const originalContent = 'TODO: Buy milk';
      const originalChecksum = computeChecksum(originalContent);

      setCachedChecksum(filePath, originalChecksum);

      // Simulate external modification
      const modifiedContent = 'TODO: Buy eggs';
      const currentChecksum = computeChecksum(modifiedContent);

      // Retrieve cached checksum
      const cached = getCachedChecksum(filePath);

      // Should detect mismatch
      expect(cached).toBe(originalChecksum);
      expect(currentChecksum).not.toBe(cached);
    });

    it('should handle rename scenario (clear old, cache new)', () => {
      // File is read and cached
      const oldPath = '/tmp/draft.txt';
      const newPath = '/tmp/final.txt';
      const content = 'Document content';
      const checksum = computeChecksum(content);

      setCachedChecksum(oldPath, checksum);

      // Simulate rename operation
      clearCachedChecksum(oldPath);

      // Old path should be gone
      expect(getCachedChecksum(oldPath)).toBeUndefined();

      // New path gets cached on next access
      setCachedChecksum(newPath, checksum);

      expect(getCachedChecksum(newPath)).toBe(checksum);
    });

    it('should handle delete scenario (clear cache)', () => {
      // File is cached
      const filePath = '/tmp/temp-file.txt';
      const checksum = computeChecksum('temporary content');

      setCachedChecksum(filePath, checksum);

      // Simulate file deletion
      clearCachedChecksum(filePath);

      // Cache should be empty for that path
      expect(getCachedChecksum(filePath)).toBeUndefined();
    });
  });
});
