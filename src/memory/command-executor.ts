/**
 * Memory Command Executor
 *
 * Type-safe command dispatch using discriminated union.
 * Provides automatic type narrowing based on command field.
 */

import { type MemoryCommand, assertNever } from './schemas.js';
import * as operations from './operations.js';
import type { OperationsContext } from './operations.js';

/**
 * Execute a memory command with type-safe dispatch
 *
 * @param args - Validated memory command (discriminated union)
 * @param context - Operations context with memory root, logger, tree view
 * @returns Command result as string
 */
export async function executeMemoryCommand(
  args: MemoryCommand,
  context: OperationsContext,
): Promise<string> {
  // TypeScript provides exhaustive checking here
  // Each case has automatic type narrowing
  switch (args.command) {
    case 'view':
      // TypeScript knows: args has { command: 'view', path: string, view_range?: [number, number] }
      return await operations.view(args, context);

    case 'create':
      // TypeScript knows: args has { command: 'create', path: string, file_text: string }
      return await operations.create(args, context);

    case 'str_replace': {
      // TypeScript knows: args has { command: 'str_replace', path: string, old_str?: string, old_string?: string, new_str?: string, new_string?: string }
      // Normalize field names: accept both old_str/new_str and old_string/new_string (or mixed)
      // Fail fast if both variants provided (likely user error)
      if (args.old_str && args.old_string) {
        throw new Error('Cannot provide both old_str and old_string. Use one or the other.');
      }
      if (args.new_str && args.new_string) {
        throw new Error('Cannot provide both new_str and new_string. Use one or the other.');
      }

      const oldStr = args.old_str ?? args.old_string;
      const newStr = args.new_str ?? args.new_string;

      if (!oldStr) {
        throw new Error('Missing required field: old_str (or old_string)');
      }
      if (!newStr) {
        throw new Error('Missing required field: new_str (or new_string)');
      }

      return await operations.str_replace(
        {
          path: args.path,
          old_str: oldStr,
          new_str: newStr,
        },
        context,
      );
    }

    case 'insert': {
      // TypeScript knows: args has { command: 'insert', path: string, insert_line: number | string, insert_text: string }
      // Normalize insert_line to number (handles Claude Code serialization issue)
      const insertLine = typeof args.insert_line === 'string' ? parseInt(args.insert_line, 10) : args.insert_line;

      // Validate the conversion
      if (isNaN(insertLine) || !Number.isInteger(insertLine)) {
        throw new Error(`Invalid insert_line: must be an integer, got "${args.insert_line}"`);
      }

      return await operations.insert(
        { ...args, insert_line: insertLine },
        context,
      );
    }

    case 'delete':
      // TypeScript knows: args has { command: 'delete', path: string }
      return await operations.deleteOp(args, context);

    case 'rename':
      // TypeScript knows: args has { command: 'rename', old_path: string, new_path: string }
      return await operations.rename(args, context);

    default:
      // This ensures exhaustive checking - won't compile if we miss a case
      assertNever(args);
  }
}
