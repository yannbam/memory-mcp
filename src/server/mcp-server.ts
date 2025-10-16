/**
 * MCP Server Setup
 *
 * Creates and configures the MCP server with memory tool registrations.
 * Defines Zod schemas for all 6 memory commands.
 * Registers tools with the MCP server.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import * as operations from '../memory/operations.js';
import type { Logger } from '../utils/logger.js';

/**
 * Unified memory command schema using discriminated union
 *
 * This matches the official Anthropic Memory tool specification where
 * a single "memory" tool dispatches based on the "command" field.
 * Each command variant has only its relevant parameters.
 */
const MemoryCommandSchema = z.discriminatedUnion('command', [
  z.object({
    command: z.literal('view'),
    path: z.string().describe('Memory path starting with /memories'),
    view_range: z
      .tuple([z.number(), z.number()])
      .optional()
      .describe('Optional line range [start, end]. Use -1 for end to read until EOF'),
  }),
  z.object({
    command: z.literal('create'),
    path: z.string().describe('Memory path starting with /memories'),
    file_text: z.string().describe('File content to write'),
  }),
  z.object({
    command: z.literal('str_replace'),
    path: z.string().describe('Memory path starting with /memories'),
    old_str: z.string().describe('Text to find (must be unique in file)'),
    new_str: z.string().describe('Replacement text'),
  }),
  z.object({
    command: z.literal('insert'),
    path: z.string().describe('Memory path starting with /memories'),
    insert_line: z.number().describe('Line number where text should be inserted (0-based)'),
    insert_text: z.string().describe('Text to insert'),
  }),
  z.object({
    command: z.literal('delete'),
    path: z.string().describe('Memory path starting with /memories'),
  }),
  z.object({
    command: z.literal('rename'),
    old_path: z.string().describe('Current memory path'),
    new_path: z.string().describe('New memory path'),
  }),
]);

/**
 * TypeScript type inferred from the discriminated union schema
 * Provides type safety for command dispatch and operations
 */
export type MemoryCommand = z.infer<typeof MemoryCommandSchema>;

/**
 * Create and configure MCP server with memory tools
 *
 * @param memoryRoot - Absolute filesystem path to memory root directory
 * @param logger - Debug logger instance
 * @param treeView - Enable tree view for directory listings
 * @returns Configured McpServer instance
 */
export function createMemoryServer(memoryRoot: string, logger: Logger, treeView: boolean): McpServer {
  // Create MCP server instance
  const server = new McpServer({
    name: 'memory-mcp',
    version: '0.1.0',
  });

  // Create operations context
  const context: operations.OperationsContext = {
    memoryRoot,
    logger,
    treeView,
  };

  // Register unified memory tool with discriminated union schema
  // Note: MCP SDK's inputSchema expects ZodRawShape, but we use a discriminated union
  // for proper type safety. We validate using MemoryCommandSchema in the handler.
  server.registerTool(
    'memory',
    {
      title: 'Memory Operations',
      description:
        'Perform memory operations with command parameter: ' +
        'view (show directory/file contents), create (create/overwrite file), ' +
        'str_replace (replace unique text in file), insert (insert text at line), ' +
        'delete (remove file/directory), rename (move/rename file/directory). ' +
        'The command field determines which parameters are required.',
      inputSchema: {
        command: z.enum(['view', 'create', 'str_replace', 'insert', 'delete', 'rename']),
        path: z.string().optional(),
        view_range: z.tuple([z.number(), z.number()]).optional(),
        file_text: z.string().optional(),
        old_str: z.string().optional(),
        new_str: z.string().optional(),
        insert_line: z.number().optional(),
        insert_text: z.string().optional(),
        old_path: z.string().optional(),
        new_path: z.string().optional(),
      },
    },
    async (params) => {
      try {
        // Validate input using discriminated union schema for proper type safety
        // This ensures only relevant parameters are provided for each command
        const command = MemoryCommandSchema.parse(params);

        // Dispatch to appropriate operation based on command field
        let result: string;
        switch (command.command) {
          case 'view':
            result = await operations.view(command, context);
            break;
          case 'create':
            result = await operations.create(command, context);
            break;
          case 'str_replace':
            result = await operations.str_replace(command, context);
            break;
          case 'insert':
            result = await operations.insert(command, context);
            break;
          case 'delete':
            result = await operations.deleteOp(command, context);
            break;
          case 'rename':
            result = await operations.rename(command, context);
            break;
          default: {
            // TypeScript exhaustiveness check ensures all cases are handled
            const exhaustiveCheck: never = command;
            throw new Error(`Unknown command: ${JSON.stringify(exhaustiveCheck)}`);
          }
        }

        // Return MCP tool response for success
        return {
          content: [{ type: 'text', text: result }],
        };
      } catch (error) {
        // Return MCP tool response for error
        // Set isError flag so clients can detect failures
        const errorMessage = error instanceof Error ? error.message : String(error);
        return {
          content: [{ type: 'text', text: errorMessage }],
          isError: true,
        };
      }
    },
  );

  return server;
}
