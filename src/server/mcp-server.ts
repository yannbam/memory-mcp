/**
 * MCP Server Setup with Low-Level API
 *
 * Uses low-level Server API instead of McpServer.registerTool() to enable
 * proper discriminated union schema with oneOf JSON Schema structure.
 *
 * This approach allows passing exact required fields per command variant,
 * providing better guidance to Claude Code about which parameters are needed.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  type ListToolsResult,
  type CallToolResult,
  ErrorCode,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { MemoryCommandSchema, type MemoryCommand } from '../memory/schemas.js';
import { executeMemoryCommand } from '../memory/command-executor.js';
import type { OperationsContext } from '../memory/operations.js';
import type { Logger } from '../utils/logger.js';

/**
 * Create and configure MCP server with memory tools using low-level API
 *
 * @param memoryRoot - Absolute filesystem path to memory root directory
 * @param logger - Debug logger instance
 * @param treeView - Enable tree view for directory listings
 * @returns Configured Server instance
 */
export function createMemoryServer(memoryRoot: string, logger: Logger, treeView: boolean): Server {
  // Create base Server instance
  const server = new Server(
    {
      name: 'memory-mcp',
      version: '0.1.0',
    },
    {
      capabilities: {
        tools: {}, // Enable tools capability
      },
    },
  );

  // Create operations context
  const context: OperationsContext = {
    memoryRoot,
    logger,
    treeView,
  };

  // Convert Zod discriminated union to JSON Schema
  // Use $refStrategy: "none" to inline the union instead of using $ref
  // This allows us to safely add type: "object" at the top level
  const baseSchema = zodToJsonSchema(MemoryCommandSchema, {
    name: 'MemoryCommand',
    strictUnions: true,
    $refStrategy: 'none', // Inline the schema to avoid $ref at top level
  });

  // MCP protocol requires type: "object" at top level
  // Add it to satisfy protocol validation while keeping anyOf structure
  const memoryToolInputSchema = {
    type: 'object',
    ...baseSchema,
  };

  // Register tools/list handler
  server.setRequestHandler(ListToolsRequestSchema, (): ListToolsResult => {
    return {
      tools: [
        {
          name: 'memory',
          description:
            'File-based memory storage tool' +
            'Perform memory operations with command parameter: ' +
            'view (show directory/file contents), create (create/overwrite file), ' +
            'str_replace (replace unique text in file), insert (insert text at line), ' +
            'delete (remove file/directory), rename (move/rename file/directory). ' +
            'The command field determines which parameters are required.' +
            'Path must start with /memories/' +
            'Claude MUST call memory(command: "view", path: "/memories") at the VERY BEGINNING of EVERY conversation' +
            'Claude MUST use the memory tool *proactively* and *regularly* to read and write important facts and insights into persistent memory shared across all conversations!',
          // Cast to any to bypass SDK's overly strict type constraint
          // The actual MCP protocol supports any valid JSON Schema, including oneOf
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any
          inputSchema: memoryToolInputSchema as any, // ← Top-level oneOf with discriminated union!
        },
      ],
    };
  });

  // Register tools/call handler
  server.setRequestHandler(CallToolRequestSchema, async (request): Promise<CallToolResult> => {
    const toolName = request.params.name;

    if (toolName !== 'memory') {
      throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${toolName}`);
    }

    // Parse and validate with discriminated union schema
    // This provides automatic type narrowing based on 'command' field
    let args: MemoryCommand;
    try {
      args = MemoryCommandSchema.parse(request.params.arguments);
    } catch (error) {
      throw new McpError(
        ErrorCode.InvalidParams,
        `Invalid arguments for memory tool: ${error instanceof Error ? error.message : String(error)}`,
      );
    }

    // Execute command with full type safety
    try {
      const result = await executeMemoryCommand(args, context);

      return {
        content: [
          {
            type: 'text',
            text: result,
          },
        ],
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);

      // Log error for server-side debugging and monitoring
      await context.logger.debug('command-execution-error', {
        command: args.command,
        path: 'path' in args ? args.path : undefined,
        error: errorMessage,
        stack: error instanceof Error ? error.stack : undefined,
      });

      return {
        content: [
          {
            type: 'text',
            text: `Error: ${errorMessage}`,
          },
        ],
        isError: true,
      };
    }
  });

  return server;
}
