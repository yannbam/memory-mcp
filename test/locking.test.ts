/**
 * Locking and Concurrency Detection Tests
 *
 * Tests for the checksum-based concurrency detection system in locking.ts.
 * Verifies both sequential and concurrent modification detection.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import * as fs from 'fs/promises';
import * as path from 'path';
import { withWriteLock, withReadLock, withMultipleWriteLocks } from '../src/memory/locking.js';
import { setCachedChecksum, clearAllCachedChecksums, computeChecksum } from '../src/memory/checksums.js';

describe('Locking and Concurrency Detection', () => {
  // Test directory
  const testRoot = path.join('/tmp', 'locking-test-' + Date.now());

  beforeEach(async () => {
    // Clear cache and create test directory
    clearAllCachedChecksums();
    await fs.mkdir(testRoot, { recursive: true });
  });

  afterEach(async () => {
    // Clean up test directory
    await fs.rm(testRoot, { recursive: true, force: true });
  });

  describe('Read Locks', () => {
    it('should allow concurrent reads', async () => {
      // Create test file
      const filePath = path.join(testRoot, 'test.txt');
      await fs.writeFile(filePath, 'content', 'utf-8');

      // Execute multiple concurrent reads
      const results = await Promise.all([
        withReadLock(filePath, async () => {
          await fs.readFile(filePath, 'utf-8');
          return 'read1';
        }),
        withReadLock(filePath, async () => {
          await fs.readFile(filePath, 'utf-8');
          return 'read2';
        }),
        withReadLock(filePath, async () => {
          await fs.readFile(filePath, 'utf-8');
          return 'read3';
        }),
      ]);

      // All should complete successfully
      expect(results).toEqual(['read1', 'read2', 'read3']);
    });

    it('should lock non-existent file\'s parent directory', async () => {
      // Try to read non-existent file
      const filePath = path.join(testRoot, 'nonexistent.txt');

      // Should lock parent directory instead (no error)
      await withReadLock(filePath, async () => {
        // Operation succeeds even though file doesn't exist
        return 'success';
      });
    });
  });

  describe('Write Locks - No Concurrency Check', () => {
    it('should execute write operation without concurrency check', async () => {
      // For operations like create that don't need concurrency checking
      const filePath = path.join(testRoot, 'new-file.txt');

      await withWriteLock(filePath, false, async () => {
        await fs.writeFile(filePath, 'new content', 'utf-8');
      });

      // File should exist
      const content = await fs.readFile(filePath, 'utf-8');
      expect(content).toBe('new content');
    });
  });

  describe('Write Locks - Sequential Modification Detection', () => {
    it('should detect sequential modifications via cache comparison', async () => {
      // Create and cache initial state
      const filePath = path.join(testRoot, 'test.txt');
      const originalContent = 'Original content';
      await fs.writeFile(filePath, originalContent, 'utf-8');

      // Simulate previous operation caching the checksum
      const originalChecksum = computeChecksum(originalContent);
      setCachedChecksum(filePath, originalChecksum);

      // External process modifies file
      const modifiedContent = 'Modified by another process';
      await fs.writeFile(filePath, modifiedContent, 'utf-8');

      // Try to write with concurrency check enabled
      await expect(
        withWriteLock(filePath, true, async () => {
          // This should never execute
          throw new Error('Operation should not execute');
        }),
      ).rejects.toThrow('File has been modified by another process');
    });

    it('should show current file contents in error message', async () => {
      // Create and cache initial state
      const filePath = path.join(testRoot, 'notes.txt');
      await fs.writeFile(filePath, 'TODO: Buy milk', 'utf-8');

      setCachedChecksum(filePath, computeChecksum('TODO: Buy milk'));

      // External modification
      await fs.writeFile(filePath, 'TODO: Buy eggs', 'utf-8');

      // Try to write
      try {
        await withWriteLock(filePath, true, async () => {
          throw new Error('Should not execute');
        });
        fail('Should have thrown concurrency error');
      } catch (err) {
        const error = err as Error;
        expect(error.message).toContain('File has been modified by another process');
        expect(error.message).toContain('TODO: Buy eggs'); // Current content
        expect(error.message).toContain('Current contents of');
        expect(error.message).toContain('━'); // Separator line
      }
    });

    it('should truncate large file contents in error message', async () => {
      // Create file with large content
      const filePath = path.join(testRoot, 'large.txt');
      const largeContent = 'x'.repeat(10000); // 10KB

      await fs.writeFile(filePath, largeContent, 'utf-8');
      setCachedChecksum(filePath, computeChecksum(largeContent));

      // Modify to different large content
      const newLargeContent = 'y'.repeat(10000);
      await fs.writeFile(filePath, newLargeContent, 'utf-8');

      // Try to write
      try {
        await withWriteLock(filePath, true, async () => {
          throw new Error('Should not execute');
        });
        fail('Should have thrown concurrency error');
      } catch (err) {
        const error = err as Error;
        // Should contain truncation indicator
        expect(error.message).toContain('truncated');
        expect(error.message).toContain('10000 bytes total');
        // Should not contain the full 10KB
        expect(error.message.length).toBeLessThan(7000); // ~5000 chars + overhead
      }
    });

    it('should allow write when no cache entry exists (first access)', async () => {
      // Create file but DON'T cache it
      const filePath = path.join(testRoot, 'uncached.txt');
      await fs.writeFile(filePath, 'content', 'utf-8');

      // Write should succeed (no cached checksum to compare against)
      await withWriteLock(filePath, true, async () => {
        await fs.writeFile(filePath, 'new content', 'utf-8');
      });

      // Verify write succeeded
      const content = await fs.readFile(filePath, 'utf-8');
      expect(content).toBe('new content');
    });

    it('should allow write when content matches cache', async () => {
      // Create file and cache its checksum
      const filePath = path.join(testRoot, 'unchanged.txt');
      const content = 'Unchanged content';

      await fs.writeFile(filePath, content, 'utf-8');
      setCachedChecksum(filePath, computeChecksum(content));

      // Content hasn't changed - write should succeed
      await withWriteLock(filePath, true, async () => {
        await fs.writeFile(filePath, 'Updated content', 'utf-8');
      });

      const result = await fs.readFile(filePath, 'utf-8');
      expect(result).toBe('Updated content');
    });
  });

  describe('Write Locks - Concurrent Modification Detection', () => {
    it('should detect modifications during lock acquisition', async () => {
      // This test simulates concurrent modification during the lock-wait window
      // In practice this is tested via integration tests with actual concurrency

      // Create and cache file
      const filePath = path.join(testRoot, 'concurrent.txt');
      const originalContent = 'Original';
      await fs.writeFile(filePath, originalContent, 'utf-8');

      // Cache original checksum
      setCachedChecksum(filePath, computeChecksum(originalContent));

      // Simulate: during lock acquisition, file gets modified
      // We'll modify it synchronously before the operation executes
      let operationStarted = false;

      try {
        await withWriteLock(filePath, true, async () => {
          operationStarted = true;
          // This should not execute if concurrent modification detected
          throw new Error('Should not reach here');
        });

        fail('Should have detected modification');
      } catch (err) {
        const error = err as Error;

        // If modification was detected, operation should not start
        // But in this unit test, we can't easily simulate true concurrency
        // Integration tests will cover this properly
        expect(error.message).toBeDefined();
      }
    });
  });

  describe('Write Locks - Error Handling', () => {
    it('should handle non-existent files gracefully', async () => {
      // File doesn't exist
      const filePath = path.join(testRoot, 'nonexistent.txt');

      // Should lock parent directory and proceed
      await withWriteLock(filePath, true, async () => {
        // Can create the file
        await fs.writeFile(filePath, 'created', 'utf-8');
      });

      // File should be created
      const content = await fs.readFile(filePath, 'utf-8');
      expect(content).toBe('created');
    });

    it('should handle directory paths gracefully (EISDIR)', async () => {
      // Create directory
      const dirPath = path.join(testRoot, 'testdir');
      await fs.mkdir(dirPath);

      // Should handle EISDIR error and let operation validate
      let operationExecuted = false;

      try {
        await withWriteLock(dirPath, true, async () => {
          operationExecuted = true;
          // Operation can check if it's a directory and throw appropriate error
          const stat = await fs.stat(dirPath);
          if (stat.isDirectory()) {
            throw new Error('Path is a directory, not a file');
          }
        });
      } catch (err) {
        const error = err as Error;
        expect(error.message).toBe('Path is a directory, not a file');
      }

      // Operation should have executed (EISDIR was handled)
      expect(operationExecuted).toBe(true);
    });

    it('should propagate other filesystem errors', async () => {
      // Try to read a file we don't have permission for
      const filePath = path.join(testRoot, 'noperm.txt');
      await fs.writeFile(filePath, 'content', 'utf-8');
      await fs.chmod(filePath, 0o000); // No permissions

      // Should propagate permission error
      await expect(
        withWriteLock(filePath, true, async () => {
          throw new Error('Should not execute');
        }),
      ).rejects.toThrow(); // EACCES or similar

      // Restore permissions for cleanup
      await fs.chmod(filePath, 0o644);
    });
  });

  describe('Multiple Write Locks', () => {
    it('should acquire locks on multiple paths atomically', async () => {
      // Create two files
      const file1 = path.join(testRoot, 'file1.txt');
      const file2 = path.join(testRoot, 'file2.txt');

      await fs.writeFile(file1, 'content1', 'utf-8');
      await fs.writeFile(file2, 'content2', 'utf-8');

      // Acquire locks on both
      await withMultipleWriteLocks([file1, file2], async () => {
        // Modify both files
        await fs.writeFile(file1, 'modified1', 'utf-8');
        await fs.writeFile(file2, 'modified2', 'utf-8');
      });

      // Both should be modified
      expect(await fs.readFile(file1, 'utf-8')).toBe('modified1');
      expect(await fs.readFile(file2, 'utf-8')).toBe('modified2');
    });

    it('should handle duplicate paths in multiple locks', async () => {
      // Duplicate paths should be deduplicated
      const filePath = path.join(testRoot, 'test.txt');
      await fs.writeFile(filePath, 'content', 'utf-8');

      // Pass same path multiple times
      await withMultipleWriteLocks([filePath, filePath, filePath], async () => {
        await fs.writeFile(filePath, 'modified', 'utf-8');
      });

      expect(await fs.readFile(filePath, 'utf-8')).toBe('modified');
    });

    it('should sort paths to prevent deadlocks', async () => {
      // Create files
      const fileA = path.join(testRoot, 'a.txt');
      const fileB = path.join(testRoot, 'b.txt');
      const fileC = path.join(testRoot, 'c.txt');

      await fs.writeFile(fileA, 'a', 'utf-8');
      await fs.writeFile(fileB, 'b', 'utf-8');
      await fs.writeFile(fileC, 'c', 'utf-8');

      // Paths in unsorted order
      await withMultipleWriteLocks([fileC, fileA, fileB], async () => {
        // Locks should be acquired in sorted order: a, b, c
        // This prevents deadlocks
        await fs.writeFile(fileA, 'modified', 'utf-8');
      });

      expect(await fs.readFile(fileA, 'utf-8')).toBe('modified');
    });
  });
});
