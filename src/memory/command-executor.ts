/**
 * Memory Command Executor
 *
 * Type-safe command dispatch using discriminated union.
 * Provides automatic type narrowing based on command field.
 */

import { MemoryCommandSchema, type MemoryCommand, assertNever } from './schemas.js';
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

    case 'str_replace':
      // TypeScript knows: args has { command: 'str_replace', path: string, old_str: string, new_str: string }
      return await operations.str_replace(args, context);

    case 'insert':
      // TypeScript knows: args has { command: 'insert', path: string, insert_line: number, insert_text: string }
      return await operations.insert(args, context);

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
