# Discriminated Union Implementation Guide

## Overview

This document provides a complete, self-contained guide for implementing a **top-level discriminated union** in the memory MCP tool's input schema using the **low-level Server API** (Approach 2 from GPT-5 research).

## Why This Approach

After comprehensive research (GPT-5 consultation, MCP SDK exploration, Context7 documentation review), we found that the MCP TypeScript SDK's `registerTool` method cannot accept a `z.discriminatedUnion()` directly because:

1. **Type Constraint**: `registerTool` expects `ZodRawShape` (plain object of Zod types), not a complete Zod schema
2. **Internal Wrapping**: The SDK automatically wraps input in `z.object(inputSchema)`, making top-level unions impossible
3. **Pending PR**: There's an open PR (#816) to support `ZodType<object>`, but it's not merged as of 2025-10-27

**Approach 2 (Low-Level Server API)** is the most technically correct solution that:
- ✅ Works today with SDK v1.0.4
- ✅ Produces proper `oneOf` JSON Schema with discriminated union
- ✅ Provides full TypeScript type narrowing
- ✅ Is used by production MCP servers
- ✅ Standards-compliant and future-proof

## Current Implementation Overview

**Current file**: `src/server/mcp-server.ts` (lines ~112-240)

Currently uses `McpServer.registerTool()` with a flat schema where all parameters are optional:

```typescript
// Current approach (simplified)
const memoryInputSchema = {
  command: z.enum(['view', 'create', 'str_replace', 'insert', 'delete', 'rename']),
  path: z.string().optional(),
  view_range: z.array(z.number()).length(2).optional(),
  file_text: z.string().optional(),
  // ... all 10+ parameters optional
};

server.registerTool('memory', { inputSchema: memoryInputSchema }, async (input) => {
  // Manual validation per command
});
```

**Problem**: JSON Schema shows all fields as optional, providing poor guidance to clients (Claude Code).

## New Implementation: Discriminated Union with Low-Level Server

### Step 1: Define the Discriminated Union Schema

Create a new file `src/memory/schemas.ts`:

```typescript
import { z } from 'zod';

// Define each command as a separate schema with required fields
const ViewCommand = z.object({
  command: z.literal('view'),
  path: z.string().describe('Path to view (file or directory)'),
  view_range: z.tuple([z.number(), z.number()])
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
```

### Step 2: Modify Server Setup to Use Low-Level API

Update `src/server/mcp-server.ts` to use the low-level `Server` instead of `McpServer` for tool registration:

```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  type ListToolsResult,
  type CallToolResult,
  ErrorCode,
  McpError
} from '@modelcontextprotocol/sdk/types.js';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { MemoryCommandSchema, type MemoryCommand } from '../memory/schemas.js';
import { executeMemoryCommand } from '../memory/command-executor.js';

export function createMemoryToolServer(config: ServerConfig): Server {
  const server = new Server(
    {
      name: 'memory-mcp',
      version: '0.1.0',
    },
    {
      capabilities: {
        tools: {},  // Enable tools capability
      },
    }
  );

  // Convert Zod discriminated union to JSON Schema
  // This produces proper oneOf with const discriminants
  const memoryToolInputSchema = zodToJsonSchema(MemoryCommandSchema, {
    name: 'MemoryCommand',
    strictUnions: true,
  });

  // Register tools/list handler
  server.setRequestHandler(ListToolsRequestSchema, async (): Promise<ListToolsResult> => {
    return {
      tools: [
        {
          name: 'memory',
          description: 'Manage persistent memory files with view, create, edit, delete, and rename operations. ' +
                      'All paths must start with /memories. Supports file operations, directory listings, and text manipulation.',
          inputSchema: memoryToolInputSchema,  // ← Top-level oneOf with discriminated union!
        },
      ],
    };
  });

  // Register tools/call handler
  server.setRequestHandler(CallToolRequestSchema, async (request): Promise<CallToolResult> => {
    const toolName = request.params.name;

    if (toolName !== 'memory') {
      throw new McpError(
        ErrorCode.MethodNotFound,
        `Unknown tool: ${toolName}`
      );
    }

    // Parse and validate with discriminated union schema
    // This provides automatic type narrowing based on 'command' field
    let args: MemoryCommand;
    try {
      args = MemoryCommandSchema.parse(request.params.arguments);
    } catch (error) {
      throw new McpError(
        ErrorCode.InvalidParams,
        `Invalid arguments for memory tool: ${error instanceof Error ? error.message : String(error)}`
      );
    }

    // Execute command with full type safety
    try {
      return await executeMemoryCommand(args, config);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
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
```

### Step 3: Create Command Executor with Type-Safe Dispatch

Create `src/memory/command-executor.ts`:

```typescript
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import { MemoryCommandSchema, type MemoryCommand, assertNever } from './schemas.js';
import { MemoryOperations } from './operations.js';
import type { ServerConfig } from '../types.js';

export async function executeMemoryCommand(
  args: MemoryCommand,
  config: ServerConfig
): Promise<CallToolResult> {
  const operations = new MemoryOperations(config.memoryRoot, config.treeView);

  // TypeScript provides exhaustive checking here
  switch (args.command) {
    case 'view':
      // TypeScript knows: args has { command, path, view_range? }
      return await operations.view(args.path, args.view_range);

    case 'create':
      // TypeScript knows: args has { command, path, file_text }
      return await operations.create(args.path, args.file_text);

    case 'str_replace':
      // TypeScript knows: args has { command, path, old_str, new_str }
      return await operations.strReplace(args.path, args.old_str, args.new_str);

    case 'insert':
      // TypeScript knows: args has { command, path, insert_line, insert_text }
      return await operations.insert(args.path, args.insert_line, args.insert_text);

    case 'delete':
      // TypeScript knows: args has { command, path }
      return await operations.delete(args.path);

    case 'rename':
      // TypeScript knows: args has { command, old_path, new_path }
      return await operations.rename(args.old_path, args.new_path);

    default:
      // This ensures exhaustive checking - won't compile if we miss a case
      assertNever(args);
  }
}
```

### Step 4: Update MemoryOperations Method Signatures

The operations methods should match the discriminated union fields. Update `src/memory/operations.ts`:

```typescript
export class MemoryOperations {
  // ... existing constructor and private methods ...

  /**
   * View command - show file contents or directory listing
   */
  async view(path: string, viewRange?: [number, number]): Promise<CallToolResult> {
    // Implementation (already exists, just update signature)
  }

  /**
   * Create command - create new file with content
   */
  async create(path: string, fileText: string): Promise<CallToolResult> {
    // Implementation (already exists, just update signature)
  }

  /**
   * String replace command - replace text in file
   */
  async strReplace(path: string, oldStr: string, newStr: string): Promise<CallToolResult> {
    // Implementation (already exists, just update signature)
  }

  /**
   * Insert command - insert text at line number
   */
  async insert(path: string, insertLine: number, insertText: string): Promise<CallToolResult> {
    // Implementation (already exists, just update signature)
  }

  /**
   * Delete command - remove file or directory
   */
  async delete(path: string): Promise<CallToolResult> {
    // Implementation (already exists, just update signature)
  }

  /**
   * Rename command - move/rename file or directory
   */
  async rename(oldPath: string, newPath: string): Promise<CallToolResult> {
    // Implementation (already exists, just update signature)
  }
}
```

### Step 5: Update Entry Point

Update `src/index.ts` to use the new server:

```typescript
import { createMemoryToolServer } from './server/mcp-server.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

// Parse CLI args (existing code)
const config = parseCliArgs();

// Create server with discriminated union support
const server = createMemoryToolServer(config);

// Connect transport
const transport = new StdioServerTransport();
await server.connect(transport);

// Log ready message
if (config.debugLog) {
  await logDebug('Memory MCP server started with discriminated union schema');
}
```

## Resulting JSON Schema

The `zodToJsonSchema` conversion will produce:

```json
{
  "oneOf": [
    {
      "type": "object",
      "properties": {
        "command": { "type": "string", "const": "view" },
        "path": { "type": "string", "description": "Path to view (file or directory)" },
        "view_range": {
          "type": "array",
          "items": [{ "type": "number" }, { "type": "number" }],
          "minItems": 2,
          "maxItems": 2,
          "description": "Optional [start, end] line range for file viewing"
        }
      },
      "required": ["command", "path"],
      "additionalProperties": false
    },
    {
      "type": "object",
      "properties": {
        "command": { "type": "string", "const": "create" },
        "path": { "type": "string", "description": "Path where file will be created" },
        "file_text": { "type": "string", "description": "Content to write to the file" }
      },
      "required": ["command", "path", "file_text"],
      "additionalProperties": false
    },
    {
      "type": "object",
      "properties": {
        "command": { "type": "string", "const": "rename" },
        "old_path": { "type": "string", "description": "Current path of file or directory" },
        "new_path": { "type": "string", "description": "New path for file or directory" }
      },
      "required": ["command", "old_path", "new_path"],
      "additionalProperties": false
    }
    // ... other variants
  ]
}
```

This is a **proper discriminated union** with:
- `oneOf` for the union
- `const` for the discriminator values
- Correct `required` fields per variant
- No extra optional fields

## Testing with Claude Code

### 1. Build and Install

```bash
npm run build
npm link

# Or test directly
node dist/index.js
```

### 2. Configure in Claude Code

Update your `.mcp.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/home/jan/projects/memory-mcp/dist/index.js"],
      "transport": "stdio"
    }
  }
}
```

### 3. Restart Claude Code

Reconnect to the MCP server to pick up the new schema.

### 4. Test Each Command Variant

Ask Claude Code to:

```
Use the memory tool to create a test file at /memories/test.txt with content "Hello, World!"
```

Claude should call:
```json
{
  "command": "create",
  "path": "/memories/test.txt",
  "file_text": "Hello, World!"
}
```

Try invalid calls to verify validation:
```json
{
  "command": "create",
  "path": "/memories/test.txt"
  // Missing file_text - should be rejected by Zod validation
}
```

### 5. Verify JSON Schema

Use MCP Inspector or debug logging to inspect what schema is advertised:

```bash
# Enable debug mode
DEBUG=true node dist/index.js
```

Check the `tools/list` response to confirm `oneOf` structure.

## Benefits of This Implementation

### 1. Client Experience
- Claude Code sees **exact required fields** per command
- Better parameter suggestions and validation
- Clearer error messages

### 2. Type Safety
- **Compile-time exhaustive checking** in switch statement
- **Automatic type narrowing** based on command
- **No type assertions needed**

### 3. Maintainability
- Each command schema is **self-contained and documented**
- Adding new commands is **straightforward** (add to union)
- **Clear separation** of schema definition and execution

### 4. Standards Compliance
- Uses **standard JSON Schema `oneOf`**
- Follows **discriminated union patterns**
- **Future-proof** - works with upcoming SDK improvements

## Migration Checklist

- [ ] Create `src/memory/schemas.ts` with discriminated union
- [ ] Create `src/memory/command-executor.ts` with type-safe dispatch
- [ ] Update `src/server/mcp-server.ts` to use low-level Server API
- [ ] Update `src/memory/operations.ts` method signatures
- [ ] Update `src/index.ts` entry point
- [ ] Update tests to use new schemas
- [ ] Build and test with Claude Code
- [ ] Update README.md with new architecture
- [ ] Update CLAUDE.md handoff section

## Rollback Plan

If this approach doesn't work with Claude Code:

1. **Keep the branch** - don't delete the work
2. **Document findings** in this file (append test results)
3. **Fall back to Approach 4** (superset + runtime validation):
   - Simpler change to existing code
   - Still gets type safety in handler
   - Just loses client-side schema benefits

## References

- **GPT-5 Research**: Session 1761595236443-75uvjxnu
- **MCP SDK PR #816**: https://github.com/modelcontextprotocol/typescript-sdk/issues/816
- **SDK Issue #588**: https://github.com/modelcontextprotocol/typescript-sdk/issues/588
- **Production Example**: Glama MCP servers using low-level API
- **Zod Discriminated Unions**: https://zod.dev/?id=discriminated-unions

---

**Created**: 2025-10-27
**Branch**: `feature/discriminated-union-schema`
**Status**: ✅ **IMPLEMENTATION COMPLETE AND TESTED**

## Implementation Results (2025-10-27)

### Success ✅

**The discriminated union implementation is fully working!**

### Final Solution

The key fix was using `$refStrategy: "none"` to inline the schema:

```typescript
const baseSchema = zodToJsonSchema(MemoryCommandSchema, {
  name: 'MemoryCommand',
  strictUnions: true,
  $refStrategy: 'none',  // ← Critical: inline the union
});

const inputSchema = {
  type: 'object',  // ← MCP protocol requirement
  ...baseSchema,    // ← Spreads anyOf with discriminated union
};
```

### Issues Encountered and Fixed

1. **Initial Problem**: MCP protocol validation error
   - Error: `"expected": "object"` at `inputSchema.type`
   - Cause: zodToJsonSchema produced `$ref` at top level without `type: "object"`
   - Solution: Use `$refStrategy: "none"` + add `type: "object"`

2. **Claude Code Parameter Serialization**
   - Issue: Numeric `insert_line` parameter received as string
   - Fix: Accept `union([number, string])` in schema + normalize in handler
   - This is a pragmatic compatibility fix for MCP client quirks

### Test Results

All 6 commands tested and working:
- ✅ view (directories and files with optional line ranges)
- ✅ create (file creation with content)
- ✅ str_replace (text replacement)
- ✅ insert (text insertion - accepts number or string for line number)
- ✅ rename (file/directory renaming)
- ✅ delete (file/directory deletion)

### Schema Structure (Final)

```json
{
  "type": "object",
  "anyOf": [
    {
      "type": "object",
      "properties": {
        "command": { "type": "string", "const": "view" },
        "path": { "type": "string", "description": "..." },
        "view_range": { "type": "array", ... }
      },
      "required": ["command", "path"],
      "additionalProperties": false
    },
    // ... 5 more variants with exact required fields
  ]
}
```

### Key Learnings

1. **MCP protocol DOES support discriminated unions** - just needs `type: "object"` at root
2. **Use `$refStrategy: "none"`** to avoid `$ref` sibling issues
3. **Production MCP servers use this pattern** (confirmed by GPT-5)
4. **anyOf works fine** for discriminated unions (oneOf is more strict but anyOf is sufficient)
5. **Be pragmatic about client compatibility** - accept string OR number for numeric params if needed

### Commits

- `1b085e2`: Initial discriminated union implementation
- `92880a2`: Compatibility fix for insert_line parameter

### References Used

- GPT-5 consultation: Session 1761595236443-75uvjxnu (provided critical `$refStrategy: "none"` insight)
- MCP-Debug for protocol inspection
- mcp-inspector for validation testing
