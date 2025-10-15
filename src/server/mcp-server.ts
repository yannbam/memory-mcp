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
 * Zod schemas for memory tool commands
 */

const ViewCommandSchema = z.object({
  path: z.string().describe('Memory path starting with /memories'),
  view_range: z
    .tuple([z.number(), z.number()])
    .optional()
    .describe('Optional line range [start, end]. Use -1 for end to read until EOF'),
});

const CreateCommandSchema = z.object({
  path: z.string().describe('Memory path starting with /memories'),
  file_text: z.string().describe('File content to write'),
});

const StrReplaceCommandSchema = z.object({
  path: z.string().describe('Memory path starting with /memories'),
  old_str: z.string().describe('Text to find (must be unique in file)'),
  new_str: z.string().describe('Replacement text'),
});

const InsertCommandSchema = z.object({
  path: z.string().describe('Memory path starting with /memories'),
  insert_line: z.number().describe('Line number where text should be inserted (0-based)'),
  insert_text: z.string().describe('Text to insert'),
});

const DeleteCommandSchema = z.object({
  path: z.string().describe('Memory path starting with /memories'),
});

const RenameCommandSchema = z.object({
  old_path: z.string().describe('Current memory path'),
  new_path: z.string().describe('New memory path'),
});

/**
 * Create and configure MCP server with memory tools
 *
 * @param memoryRoot - Absolute filesystem path to memory root directory
 * @param logger - Debug logger instance
 * @returns Configured McpServer instance
 */
export function createMemoryServer(memoryRoot: string, logger: Logger): McpServer {
  // Create MCP server instance
  const server = new McpServer({
    name: 'memory-mcp',
    version: '0.1.0',
  });

  // Create operations context
  const context: operations.OperationsContext = {
    memoryRoot,
    logger,
  };

  // Register view tool
  server.registerTool(
    'memory_view',
    {
      title: 'View Memory',
      description: 'Show directory contents or file contents with optional line ranges',
      inputSchema: {
        path: z.string(),
        view_range: z.tuple([z.number(), z.number()]).optional(),
      },
    },
    async (params) => {
      // Validate input
      const command = ViewCommandSchema.parse(params);

      // Execute operation
      const result = await operations.view(command, context);

      // Return MCP tool response
      return {
        content: [{ type: 'text', text: result }],
      };
    },
  );

  // Register create tool
  server.registerTool(
    'memory_create',
    {
      title: 'Create Memory File',
      description: 'Create or overwrite a file in memory',
      inputSchema: {
        path: z.string(),
        file_text: z.string(),
      },
    },
    async (params) => {
      // Validate input
      const command = CreateCommandSchema.parse(params);

      // Execute operation
      const result = await operations.create(command, context);

      // Return MCP tool response
      return {
        content: [{ type: 'text', text: result }],
      };
    },
  );

  // Register str_replace tool
  server.registerTool(
    'memory_str_replace',
    {
      title: 'Replace Text in Memory File',
      description: 'Replace unique text in a memory file',
      inputSchema: {
        path: z.string(),
        old_str: z.string(),
        new_str: z.string(),
      },
    },
    async (params) => {
      // Validate input
      const command = StrReplaceCommandSchema.parse(params);

      // Execute operation
      const result = await operations.str_replace(command, context);

      // Return MCP tool response
      return {
        content: [{ type: 'text', text: result }],
      };
    },
  );

  // Register insert tool
  server.registerTool(
    'memory_insert',
    {
      title: 'Insert Text in Memory File',
      description: 'Insert text at a specific line in a memory file',
      inputSchema: {
        path: z.string(),
        insert_line: z.number(),
        insert_text: z.string(),
      },
    },
    async (params) => {
      // Validate input
      const command = InsertCommandSchema.parse(params);

      // Execute operation
      const result = await operations.insert(command, context);

      // Return MCP tool response
      return {
        content: [{ type: 'text', text: result }],
      };
    },
  );

  // Register delete tool
  server.registerTool(
    'memory_delete',
    {
      title: 'Delete Memory File/Directory',
      description: 'Delete a file or directory from memory',
      inputSchema: {
        path: z.string(),
      },
    },
    async (params) => {
      // Validate input
      const command = DeleteCommandSchema.parse(params);

      // Execute operation
      const result = await operations.deleteOp(command, context);

      // Return MCP tool response
      return {
        content: [{ type: 'text', text: result }],
      };
    },
  );

  // Register rename tool
  server.registerTool(
    'memory_rename',
    {
      title: 'Rename/Move Memory File/Directory',
      description: 'Rename or move a file or directory in memory',
      inputSchema: {
        old_path: z.string(),
        new_path: z.string(),
      },
    },
    async (params) => {
      // Validate input
      const command = RenameCommandSchema.parse(params);

      // Execute operation
      const result = await operations.rename(command, context);

      // Return MCP tool response
      return {
        content: [{ type: 'text', text: result }],
      };
    },
  );

  return server;
}
