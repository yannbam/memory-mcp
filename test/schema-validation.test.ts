/**
 * Schema Validation Tests
 *
 * Tests that MemoryCommandSchema correctly validates all command variants
 * and rejects invalid input. These tests verify the discriminated union
 * schema behavior before commands reach the executor.
 *
 * Follow e/test principles: adversarial testing to catch real problems.
 */

import { describe, it, expect } from '@jest/globals';
import { MemoryCommandSchema } from '../src/memory/schemas.js';

describe('Schema Validation - Valid Commands', () => {
  describe('view command', () => {
    it('should accept view command without view_range', () => {
      // Test basic view command
      const result = MemoryCommandSchema.parse({
        command: 'view',
        path: '/memories',
      });

      expect(result.command).toBe('view');
      expect(result.path).toBe('/memories');
      expect(result).not.toHaveProperty('view_range');
    });

    it('should accept view command with view_range', () => {
      // Test view with line range
      const result = MemoryCommandSchema.parse({
        command: 'view',
        path: '/memories/test.txt',
        view_range: [1, 10],
      });

      expect(result.command).toBe('view');
      expect(result.path).toBe('/memories/test.txt');
      expect(result.view_range).toEqual([1, 10]);
    });

    it('should accept view_range with negative end (EOF marker)', () => {
      // Test EOF marker pattern
      const result = MemoryCommandSchema.parse({
        command: 'view',
        path: '/memories/test.txt',
        view_range: [5, -1],
      });

      expect(result.view_range).toEqual([5, -1]);
    });
  });

  describe('create command', () => {
    it('should accept create command with file_text', () => {
      // Test file creation
      const result = MemoryCommandSchema.parse({
        command: 'create',
        path: '/memories/new.txt',
        file_text: 'Hello, World!',
      });

      expect(result.command).toBe('create');
      expect(result.path).toBe('/memories/new.txt');
      expect(result.file_text).toBe('Hello, World!');
    });

    it('should accept create command with empty file_text', () => {
      // Test edge case: empty file creation
      const result = MemoryCommandSchema.parse({
        command: 'create',
        path: '/memories/empty.txt',
        file_text: '',
      });

      expect(result.file_text).toBe('');
    });
  });

  describe('str_replace command', () => {
    it('should accept str_replace with old_str/new_str (snake_case)', () => {
      // Test primary naming convention
      const result = MemoryCommandSchema.parse({
        command: 'str_replace',
        path: '/memories/test.txt',
        old_str: 'foo',
        new_str: 'bar',
      });

      expect(result.command).toBe('str_replace');
      expect(result).toHaveProperty('old_str', 'foo');
      expect(result).toHaveProperty('new_str', 'bar');
    });

    it('should accept str_replace with old_string/new_string', () => {
      // Test alternative naming convention
      const result = MemoryCommandSchema.parse({
        command: 'str_replace',
        path: '/memories/test.txt',
        old_string: 'foo',
        new_string: 'bar',
      });

      expect(result.command).toBe('str_replace');
      expect(result).toHaveProperty('old_string', 'foo');
      expect(result).toHaveProperty('new_string', 'bar');
    });

    it('should accept mixed naming: old_str + new_string', () => {
      // Test mixing is ALLOWED (Phase 1 clarification)
      const result = MemoryCommandSchema.parse({
        command: 'str_replace',
        path: '/memories/test.txt',
        old_str: 'foo',
        new_string: 'bar',
      });

      expect(result).toHaveProperty('old_str', 'foo');
      expect(result).toHaveProperty('new_string', 'bar');
    });

    it('should accept mixed naming: old_string + new_str', () => {
      // Test opposite mixing combination
      const result = MemoryCommandSchema.parse({
        command: 'str_replace',
        path: '/memories/test.txt',
        old_string: 'foo',
        new_str: 'bar',
      });

      expect(result).toHaveProperty('old_string', 'foo');
      expect(result).toHaveProperty('new_str', 'bar');
    });

    it('should accept all four fields simultaneously', () => {
      // Test schema accepts all optional fields
      // Executor will fail-fast if both variants provided (Phase 1 fix)
      const result = MemoryCommandSchema.parse({
        command: 'str_replace',
        path: '/memories/test.txt',
        old_str: 'foo',
        old_string: 'foo',
        new_str: 'bar',
        new_string: 'bar',
      });

      // Schema accepts it (validation happens in executor)
      expect(result).toHaveProperty('old_str');
      expect(result).toHaveProperty('old_string');
      expect(result).toHaveProperty('new_str');
      expect(result).toHaveProperty('new_string');
    });
  });

  describe('insert command', () => {
    it('should accept insert with insert_line as number', () => {
      // Test numeric insert_line
      const result = MemoryCommandSchema.parse({
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: 5,
        insert_text: 'inserted content',
      });

      expect(result.command).toBe('insert');
      expect(result.insert_line).toBe(5);
      expect(typeof result.insert_line).toBe('number');
    });

    it('should accept insert with insert_line as string', () => {
      // Test string insert_line (Claude Code compatibility)
      const result = MemoryCommandSchema.parse({
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: '5',
        insert_text: 'inserted content',
      });

      expect(result.command).toBe('insert');
      expect(result.insert_line).toBe('5');
      expect(typeof result.insert_line).toBe('string');
    });

    it('should accept insert with empty insert_text', () => {
      // Test edge case: inserting empty line
      const result = MemoryCommandSchema.parse({
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: 1,
        insert_text: '',
      });

      expect(result.insert_text).toBe('');
    });
  });

  describe('delete command', () => {
    it('should accept delete command', () => {
      // Test file/directory deletion
      const result = MemoryCommandSchema.parse({
        command: 'delete',
        path: '/memories/old.txt',
      });

      expect(result.command).toBe('delete');
      expect(result.path).toBe('/memories/old.txt');
    });
  });

  describe('rename command', () => {
    it('should accept rename command', () => {
      // Test file/directory rename
      const result = MemoryCommandSchema.parse({
        command: 'rename',
        old_path: '/memories/old.txt',
        new_path: '/memories/new.txt',
      });

      expect(result.command).toBe('rename');
      expect(result.old_path).toBe('/memories/old.txt');
      expect(result.new_path).toBe('/memories/new.txt');
    });
  });
});

describe('Schema Validation - Invalid Commands', () => {
  describe('invalid command names', () => {
    it('should reject unknown command name', () => {
      // Test discriminator validation
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'invalid',
          path: '/memories',
        }),
      ).toThrow();
    });

    it('should reject missing command field', () => {
      // Test required discriminator
      expect(() =>
        MemoryCommandSchema.parse({
          path: '/memories',
        }),
      ).toThrow();
    });

    it('should reject command as wrong type', () => {
      // Test discriminator type validation
      expect(() =>
        MemoryCommandSchema.parse({
          command: 123,
          path: '/memories',
        }),
      ).toThrow();
    });
  });

  describe('missing required fields', () => {
    it('should reject view without path', () => {
      // Test required field validation
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
        }),
      ).toThrow(/path/i);
    });

    it('should reject create without path', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'create',
          file_text: 'content',
        }),
      ).toThrow(/path/i);
    });

    it('should reject create without file_text', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'create',
          path: '/memories/test.txt',
        }),
      ).toThrow(/file_text/i);
    });

    it('should reject str_replace without path', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'str_replace',
          old_str: 'foo',
          new_str: 'bar',
        }),
      ).toThrow(/path/i);
    });

    it('should reject insert without path', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'insert',
          insert_line: 1,
          insert_text: 'text',
        }),
      ).toThrow(/path/i);
    });

    it('should reject insert without insert_line', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'insert',
          path: '/memories/test.txt',
          insert_text: 'text',
        }),
      ).toThrow(/insert_line/i);
    });

    it('should reject insert without insert_text', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: 1,
        }),
      ).toThrow(/insert_text/i);
    });

    it('should reject rename without old_path', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'rename',
          new_path: '/memories/new.txt',
        }),
      ).toThrow(/old_path/i);
    });

    it('should reject rename without new_path', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'rename',
          old_path: '/memories/old.txt',
        }),
      ).toThrow(/new_path/i);
    });
  });

  describe('wrong field types', () => {
    it('should reject path as number', () => {
      // Test type validation catches wrong types
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
          path: 123,
        }),
      ).toThrow();
    });

    it('should reject path as array', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
          path: ['/memories'],
        }),
      ).toThrow();
    });

    it('should reject file_text as number', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'create',
          path: '/memories/test.txt',
          file_text: 123,
        }),
      ).toThrow();
    });

    it('should reject insert_text as boolean', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: 1,
          insert_text: true,
        }),
      ).toThrow();
    });

    it('should reject insert_line as boolean', () => {
      // insert_line accepts number | string, should reject other types
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: true,
          insert_text: 'text',
        }),
      ).toThrow();
    });

    it('should reject insert_line as array', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: [1],
          insert_text: 'text',
        }),
      ).toThrow();
    });
  });

  describe('invalid view_range', () => {
    it('should reject view_range as single number', () => {
      // view_range must be tuple [number, number]
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
          path: '/memories/test.txt',
          view_range: 5,
        }),
      ).toThrow();
    });

    it('should reject view_range with one element', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
          path: '/memories/test.txt',
          view_range: [5],
        }),
      ).toThrow();
    });

    it('should reject view_range with three elements', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
          path: '/memories/test.txt',
          view_range: [1, 5, 10],
        }),
      ).toThrow();
    });

    it('should reject view_range with non-numeric elements', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
          path: '/memories/test.txt',
          view_range: ['1', '10'],
        }),
      ).toThrow();
    });

    it('should reject view_range with mixed types', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
          path: '/memories/test.txt',
          view_range: [1, '10'],
        }),
      ).toThrow();
    });
  });
});
