/**
 * Command Executor Tests
 *
 * Tests the command executor's parameter normalization, validation,
 * and dispatch logic. These tests verify behavior AFTER schema validation
 * but BEFORE operations execution.
 *
 * Follow e/test principles: test that executor fails when it should fail.
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { executeMemoryCommand } from '../src/memory/command-executor.js';
import type { OperationsContext } from '../src/memory/operations.js';
import type { Logger } from '../src/utils/logger.js';
import * as fs from 'fs/promises';
import * as path from 'path';

// Mock logger
const mockLogger: Logger = {
  debug: async () => {},
  close: async () => {},
};

describe('Command Executor - insert_line Normalization', () => {
  let testDir: string;
  let context: OperationsContext;

  beforeEach(async () => {
    // Create test directory
    testDir = path.join('/tmp', `test-executor-${Date.now()}`);
    await fs.mkdir(testDir, { recursive: true });

    context = {
      memoryRoot: testDir,
      logger: mockLogger,
      treeView: false,
    };

    // Create test file for insert operations
    await fs.writeFile(path.join(testDir, 'test.txt'), 'line1\nline2\nline3\n');
  });

  afterEach(async () => {
    // Clean up
    await fs.rm(testDir, { recursive: true, force: true });
  });

  it('should accept insert_line as number', async () => {
    // Test numeric insert_line
    const result = await executeMemoryCommand(
      {
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: 2,
        insert_text: 'inserted',
      },
      context,
    );

    // Verify insertion succeeded
    expect(result).toContain('Text inserted at line');
    const content = await fs.readFile(path.join(testDir, 'test.txt'), 'utf-8');
    expect(content).toContain('inserted');
  });

  it('should accept insert_line as numeric string', async () => {
    // Test string insert_line (Claude Code compatibility)
    const result = await executeMemoryCommand(
      {
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: '2',
        insert_text: 'inserted',
      },
      context,
    );

    expect(result).toContain('Text inserted at line');
    const content = await fs.readFile(path.join(testDir, 'test.txt'), 'utf-8');
    expect(content).toContain('inserted');
  });

  it('should reject insert_line as non-numeric string', async () => {
    // Test adversarial: invalid string should produce CLEAR error
    await expect(
      executeMemoryCommand(
        {
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: 'abc',
          insert_text: 'inserted',
        },
        context,
      ),
    ).rejects.toThrow(/Invalid insert_line.*integer.*abc/i);
  });

  it('should accept insert_line as float string (parseInt truncates)', async () => {
    // Test edge case: parseInt('2.5', 10) returns 2 (stops at decimal)
    // This is JavaScript behavior - executor accepts it, operations validates line bounds
    const result = await executeMemoryCommand(
      {
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: '2.5',
        insert_text: 'inserted',
      },
      context,
    );

    expect(result).toContain('Text inserted at line');
  });

  it('should reject insert_line as negative number (operations validates)', async () => {
    // Test: executor accepts negative integers, operations rejects them
    // parseInt('-1', 10) returns -1, Number.isInteger(-1) is true
    // Operations layer validates line number bounds
    await expect(
      executeMemoryCommand(
        {
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: '-1',
          insert_text: 'inserted',
        },
        context,
      ),
    ).rejects.toThrow(/Invalid insert_line.*-1/);
  });

  it('should accept insert_line with leading/trailing whitespace', async () => {
    // Test edge case: whitespace should be trimmed by parseInt
    const result = await executeMemoryCommand(
      {
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: ' 2 ',
        insert_text: 'inserted',
      },
      context,
    );

    expect(result).toContain('Text inserted at line');
  });

  it('should accept large numbers (operations validates bounds)', async () => {
    // Test: executor accepts any integer, operations validates line bounds
    // Very large numbers pass executor validation but fail in operations
    await expect(
      executeMemoryCommand(
        {
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: 999999,
          insert_text: 'inserted',
        },
        context,
      ),
    ).rejects.toThrow(/Invalid insert_line 999999.*Must be 0-/);
  });

  it('should accept insert_line as zero', async () => {
    // Test edge case: zero line number (operations validates if invalid)
    const result = await executeMemoryCommand(
      {
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: 0,
        insert_text: 'inserted',
      },
      context,
    );

    // Executor accepts it, operations validates line numbers
    expect(result).toBeDefined();
  });

  it('should accept insert_line with trailing non-digits (parseInt behavior)', async () => {
    // Test edge case: parseInt('2a', 10) returns 2 (parses until non-digit)
    // This is JavaScript behavior - executor accepts it
    const result = await executeMemoryCommand(
      {
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: '2a',
        insert_text: 'inserted',
      },
      context,
    );

    expect(result).toContain('Text inserted at line');
  });

  it('should reject insert_line as empty string', async () => {
    // Test adversarial: empty string should fail
    await expect(
      executeMemoryCommand(
        {
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: '',
          insert_text: 'inserted',
        },
        context,
      ),
    ).rejects.toThrow(/Invalid insert_line.*integer/i);
  });
});

describe('Command Executor - str_replace Naming Flexibility', () => {
  let testDir: string;
  let context: OperationsContext;

  beforeEach(async () => {
    testDir = path.join('/tmp', `test-executor-${Date.now()}`);
    await fs.mkdir(testDir, { recursive: true });

    context = {
      memoryRoot: testDir,
      logger: mockLogger,
      treeView: false,
    };

    // Create test file
    await fs.writeFile(path.join(testDir, 'test.txt'), 'foo bar baz');
  });

  afterEach(async () => {
    await fs.rm(testDir, { recursive: true, force: true });
  });

  it('should accept snake_case naming (old_str/new_str)', async () => {
    // Test primary naming convention
    const result = await executeMemoryCommand(
      {
        command: 'str_replace',
        path: '/memories/test.txt',
        old_str: 'bar',
        new_str: 'qux',
      },
      context,
    );

    expect(result).toContain('has been edited');
    const content = await fs.readFile(path.join(testDir, 'test.txt'), 'utf-8');
    expect(content).toBe('foo qux baz');
  });

  it('should accept old_string/new_string naming', async () => {
    // Test alternative naming convention
    const result = await executeMemoryCommand(
      {
        command: 'str_replace',
        path: '/memories/test.txt',
        old_string: 'bar',
        new_string: 'qux',
      },
      context,
    );

    expect(result).toContain('has been edited');
    const content = await fs.readFile(path.join(testDir, 'test.txt'), 'utf-8');
    expect(content).toBe('foo qux baz');
  });

  it('should accept mixed naming: old_str + new_string', async () => {
    // Test mixing is allowed (Phase 1 clarification)
    const result = await executeMemoryCommand(
      {
        command: 'str_replace',
        path: '/memories/test.txt',
        old_str: 'bar',
        new_string: 'qux',
      },
      context,
    );

    expect(result).toContain('has been edited');
    const content = await fs.readFile(path.join(testDir, 'test.txt'), 'utf-8');
    expect(content).toBe('foo qux baz');
  });

  it('should accept mixed naming: old_string + new_str', async () => {
    // Test opposite mixing combination
    const result = await executeMemoryCommand(
      {
        command: 'str_replace',
        path: '/memories/test.txt',
        old_string: 'bar',
        new_str: 'qux',
      },
      context,
    );

    expect(result).toContain('has been edited');
    const content = await fs.readFile(path.join(testDir, 'test.txt'), 'utf-8');
    expect(content).toBe('foo qux baz');
  });

  it('should fail fast when BOTH old_str AND old_string provided', async () => {
    // Test adversarial: ambiguous input should produce CLEAR error
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          old_str: 'bar',
          old_string: 'bar',
          new_str: 'qux',
        },
        context,
      ),
    ).rejects.toThrow(/Cannot provide both old_str and old_string/i);
  });

  it('should fail fast when BOTH new_str AND new_string provided', async () => {
    // Test adversarial: ambiguous output should produce CLEAR error
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          old_str: 'bar',
          new_str: 'qux',
          new_string: 'qux',
        },
        context,
      ),
    ).rejects.toThrow(/Cannot provide both new_str and new_string/i);
  });

  it('should reject when no old field provided', async () => {
    // Test adversarial: missing required field
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          new_str: 'qux',
        } as any,
        context,
      ),
    ).rejects.toThrow(/Missing required field: old_str/i);
  });

  it('should reject when no new field provided', async () => {
    // Test adversarial: missing required field
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          old_str: 'bar',
        } as any,
        context,
      ),
    ).rejects.toThrow(/Missing required field: new_str/i);
  });

  it('should prefer old_str over old_string when both undefined', async () => {
    // Test nullish coalescing behavior
    // When neither provided, error should mention old_str (first in chain)
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          new_str: 'qux',
        } as any,
        context,
      ),
    ).rejects.toThrow(/old_str/);
  });
});

describe('Command Executor - Command Dispatch', () => {
  let testDir: string;
  let context: OperationsContext;

  beforeEach(async () => {
    testDir = path.join('/tmp', `test-executor-${Date.now()}`);
    await fs.mkdir(testDir, { recursive: true });
    context = {
      memoryRoot: testDir,
      logger: mockLogger,
      treeView: false,
    };
  });

  afterEach(async () => {
    await fs.rm(testDir, { recursive: true, force: true });
  });

  it('should dispatch view command correctly', async () => {
    // Test view dispatches to operations.view
    const result = await executeMemoryCommand(
      {
        command: 'view',
        path: '/memories',
      },
      context,
    );

    expect(result).toContain('Directory: /memories');
  });

  it('should dispatch create command correctly', async () => {
    // Test create dispatches to operations.create
    const result = await executeMemoryCommand(
      {
        command: 'create',
        path: '/memories/new.txt',
        file_text: 'Hello, World!',
      },
      context,
    );

    expect(result).toContain('File created successfully');
    const exists = await fs
      .access(path.join(testDir, 'new.txt'))
      .then(() => true)
      .catch(() => false);
    expect(exists).toBe(true);
  });

  it('should dispatch delete command correctly', async () => {
    // Test delete dispatches to operations.deleteOp
    await fs.writeFile(path.join(testDir, 'delete-me.txt'), 'content');

    const result = await executeMemoryCommand(
      {
        command: 'delete',
        path: '/memories/delete-me.txt',
      },
      context,
    );

    expect(result).toContain('File deleted:');
    const exists = await fs
      .access(path.join(testDir, 'delete-me.txt'))
      .then(() => true)
      .catch(() => false);
    expect(exists).toBe(false);
  });

  it('should dispatch rename command correctly', async () => {
    // Test rename dispatches to operations.rename
    await fs.writeFile(path.join(testDir, 'old.txt'), 'content');

    const result = await executeMemoryCommand(
      {
        command: 'rename',
        old_path: '/memories/old.txt',
        new_path: '/memories/new.txt',
      },
      context,
    );

    expect(result).toContain('Renamed');
    const oldExists = await fs
      .access(path.join(testDir, 'old.txt'))
      .then(() => true)
      .catch(() => false);
    const newExists = await fs
      .access(path.join(testDir, 'new.txt'))
      .then(() => true)
      .catch(() => false);
    expect(oldExists).toBe(false);
    expect(newExists).toBe(true);
  });

  it('should pass context to all operations', async () => {
    // Test context (memoryRoot, logger) is passed correctly
    // Create a file and verify it uses the correct memoryRoot
    await executeMemoryCommand(
      {
        command: 'create',
        path: '/memories/context-test.txt',
        file_text: 'testing context',
      },
      context,
    );

    const content = await fs.readFile(path.join(testDir, 'context-test.txt'), 'utf-8');
    expect(content).toBe('testing context');
  });

  it('should propagate operation return values', async () => {
    // Test that operation results are returned unchanged
    const result = await executeMemoryCommand(
      {
        command: 'view',
        path: '/memories',
      },
      context,
    );

    // Result should be the exact string from operations.view
    expect(typeof result).toBe('string');
    expect(result).toContain('Directory:');
  });
});

describe('Command Executor - Error Handling', () => {
  let testDir: string;
  let context: OperationsContext;

  beforeEach(async () => {
    testDir = path.join('/tmp', `test-executor-${Date.now()}`);
    await fs.mkdir(testDir, { recursive: true });
    context = {
      memoryRoot: testDir,
      logger: mockLogger,
      treeView: false,
    };
  });

  afterEach(async () => {
    await fs.rm(testDir, { recursive: true, force: true });
  });

  it('should provide clear error for invalid insert_line', async () => {
    // Test error message includes the actual invalid value
    await expect(
      executeMemoryCommand(
        {
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: 'not-a-number',
          insert_text: 'text',
        },
        context,
      ),
    ).rejects.toThrow(/Invalid insert_line.*integer.*not-a-number/);
  });

  it('should propagate file not found errors from operations', async () => {
    // Test operations errors bubble up
    await expect(
      executeMemoryCommand(
        {
          command: 'view',
          path: '/memories/does-not-exist.txt',
        },
        context,
      ),
    ).rejects.toThrow(/Path not found/i);
  });

  it('should propagate path validation errors from operations', async () => {
    // Test path security validation errors bubble up
    await expect(
      executeMemoryCommand(
        {
          command: 'view',
          path: '/etc/passwd',
        },
        context,
      ),
    ).rejects.toThrow(/must start with \/memories/i);
  });

  it('should provide clear error when old_str missing', async () => {
    // Test missing field error is helpful
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          new_str: 'replacement',
        } as any,
        context,
      ),
    ).rejects.toThrow(/Missing required field: old_str \(or old_string\)/);
  });

  it('should provide clear error when new_str missing', async () => {
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          old_str: 'original',
        } as any,
        context,
      ),
    ).rejects.toThrow(/Missing required field: new_str \(or new_string\)/);
  });

  it('should include command context in ambiguous parameter errors', async () => {
    // Test error mentions both field names for clarity
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          old_str: 'foo',
          old_string: 'bar',
          new_str: 'baz',
        },
        context,
      ),
    ).rejects.toThrow(/both old_str and old_string/i);
  });

  it('should handle errors in underlying operations gracefully', async () => {
    // Test that operations errors are not wrapped/hidden
    await fs.writeFile(path.join(testDir, 'test.txt'), 'content');
    await fs.chmod(path.join(testDir, 'test.txt'), 0o444); // Read-only

    // This should fail when operations tries to write
    await expect(
      executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          old_str: 'content',
          new_str: 'new content',
        },
        context,
      ),
    ).rejects.toThrow();
  });
});
