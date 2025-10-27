/**
 * Memory Command Schemas
 *
 * Defines discriminated union schemas for all 6 memory commands.
 * Each command has its own schema with exact required fields.
 * Zod automatically provides type narrowing based on the 'command' discriminator.
 */

import { z } from 'zod';

// Define each command as a separate schema with required fields
const ViewCommand = z.object({
  command: z.literal('view'),
  path: z.string().describe('Path to view (file or directory)'),
  view_range: z
    .tuple([z.number(), z.number()])
    .optional()
    .describe('Optional [start, end] line range for file viewing'),
});

const CreateCommand = z.object({
  command: z.literal('create'),
  path: z.string().describe('Path where file will be created'),
  file_text: z.string().describe('Content to write to the file'),
});

const StrReplaceCommand = z.object({
  command: z.literal('str_replace'),
  path: z.string().describe('Path to file to modify'),
  old_str: z.string().describe('Exact text to find (must be unique in file)'),
  new_str: z.string().describe('Text to replace with'),
});

const InsertCommand = z.object({
  command: z.literal('insert'),
  path: z.string().describe('Path to file to modify'),
  insert_line: z.number().int().describe('Line number where text will be inserted (1-based)'),
  insert_text: z.string().describe('Text to insert'),
});

const DeleteCommand = z.object({
  command: z.literal('delete'),
  path: z.string().describe('Path to file or directory to delete'),
});

const RenameCommand = z.object({
  command: z.literal('rename'),
  old_path: z.string().describe('Current path of file or directory'),
  new_path: z.string().describe('New path for file or directory'),
});

// Discriminated union - Zod will use 'command' field to narrow the type
export const MemoryCommandSchema = z.discriminatedUnion('command', [
  ViewCommand,
  CreateCommand,
  StrReplaceCommand,
  InsertCommand,
  DeleteCommand,
  RenameCommand,
]);

// TypeScript type for the union
export type MemoryCommand = z.infer<typeof MemoryCommandSchema>;

// Type guard for exhaustive switch checking
export function assertNever(x: never): never {
  throw new Error(`Unexpected command: ${JSON.stringify(x)}`);
}
