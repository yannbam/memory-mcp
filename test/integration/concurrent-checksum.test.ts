/**
 * Concurrent Access Integration Test
 *
 * Tests checksum-based concurrency detection across separate MCP server processes.
 * This validates the real-world scenario: two Claude Code sessions (separate stdio
 * servers) detecting modifications that occurred sequentially (not just concurrently).
 *
 * Scenario:
 * 1. Server A: reads file (caches checksum)
 * 2. Server B: modifies file
 * 3. Server A: attempts to modify file → ERROR (detects modification via checksum)
 */

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import * as fs from 'fs/promises';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { spawn, ChildProcess } from 'child_process';

// ES module __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Type for MCP JSON-RPC messages
interface MCPRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: unknown;
}

interface MCPResponse {
  jsonrpc: '2.0';
  id: number;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

/**
 * Helper: Send JSON-RPC request to stdio MCP server
 */
async function sendMCPRequest(
  server: ChildProcess,
  request: MCPRequest,
): Promise<MCPResponse> {
  return new Promise((resolve, reject) => {
    const requestJson = JSON.stringify(request) + '\n';

    // Set up response listener
    const onData = (data: Buffer) => {
      const lines = data.toString().split('\n').filter((l) => l.trim());
      for (const line of lines) {
        try {
          const response = JSON.parse(line) as MCPResponse;
          if (response.id === request.id) {
            server.stdout!.off('data', onData);
            resolve(response);
          }
        } catch (err) {
          // Ignore parse errors (debug output, etc.)
        }
      }
    };

    server.stdout!.on('data', onData);

    // Send request
    server.stdin!.write(requestJson);

    // Timeout after 10 seconds
    setTimeout(() => {
      server.stdout!.off('data', onData);
      reject(new Error('Request timeout'));
    }, 10000);
  });
}

/**
 * Helper: Start MCP server process with stdio transport
 */
async function startMCPServer(memoryRoot: string): Promise<ChildProcess> {
  // Build path to server
  const serverPath = path.join(__dirname, '../../dist/index.js');

  // Spawn server process
  const server = spawn('node', [serverPath, '--memory-root-path', memoryRoot], {
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  // Wait for server to be ready (send initialize request)
  const initResponse = await sendMCPRequest(server, {
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: {
        name: 'concurrent-test',
        version: '1.0.0',
      },
    },
  });

  if (initResponse.error) {
    server.kill();
    throw new Error(`Server init failed: ${initResponse.error.message}`);
  }

  return server;
}

/**
 * Helper: Call memory tool via MCP
 */
async function callMemoryTool(
  server: ChildProcess,
  command: string,
  params: Record<string, unknown>,
  requestId: number,
): Promise<MCPResponse> {
  return sendMCPRequest(server, {
    jsonrpc: '2.0',
    id: requestId,
    method: 'tools/call',
    params: {
      name: 'memory',
      arguments: {
        command,
        ...params,
      },
    },
  });
}

describe('Concurrent Access Integration Test', () => {
  const testRoot = path.join('/tmp', 'concurrent-test-' + Date.now());
  const memoryPath = path.join(testRoot, 'memories');

  beforeAll(async () => {
    // Create test directory
    await fs.mkdir(memoryPath, { recursive: true });

    // Build server if needed
    try {
      await fs.access(path.join(__dirname, '../../dist/index.js'));
    } catch {
      throw new Error(
        'Server not built. Run `npm run build` before integration tests.',
      );
    }
  });

  afterAll(async () => {
    // Clean up test directory
    await fs.rm(testRoot, { recursive: true, force: true });

    // Give processes time to exit
    await new Promise((resolve) => setTimeout(resolve, 100));
  });

  it('should detect sequential modifications across separate server processes', async () => {
    let serverA: ChildProcess | null = null;
    let serverB: ChildProcess | null = null;

    try {
      // Start two separate MCP server processes
      serverA = await startMCPServer(testRoot);
      serverB = await startMCPServer(testRoot);

      // Create test file directly on filesystem
      const testFilePath = path.join(memoryPath, 'notes.txt');
      await fs.writeFile(testFilePath, 'TODO: Buy milk', 'utf-8');

      // Server A: Read file (caches checksum)
      const viewResponse = await callMemoryTool(
        serverA,
        'view',
        { path: '/memories/notes.txt' },
        100,
      );

      expect(viewResponse.error).toBeUndefined();
      expect((viewResponse.result as any).content[0].text).toContain('TODO: Buy milk');

      // Server B: Modify file (writes new checksum in its own cache)
      const modifyResponse = await callMemoryTool(
        serverB,
        'str_replace',
        {
          path: '/memories/notes.txt',
          old_str: 'milk',
          new_str: 'eggs',
        },
        200,
      );

      expect(modifyResponse.error).toBeUndefined();

      // Verify file was actually modified on disk
      const diskContent = await fs.readFile(testFilePath, 'utf-8');
      expect(diskContent).toBe('TODO: Buy eggs');

      // Server A: Try to modify based on stale understanding
      // This should FAIL - Server A's cache has old checksum, disk has new content
      const staleModifyResponse = await callMemoryTool(
        serverA,
        'str_replace',
        {
          path: '/memories/notes.txt',
          old_str: 'eggs', // Text that EXISTS in current file (after B's modification)
          new_str: 'bread',
        },
        300,
      );

      // Should detect modification and reject
      // MCP tools return errors in result.content with isError: true
      expect(staleModifyResponse.result).toBeDefined();
      const result = staleModifyResponse.result as any;

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain('File has been modified by another process');
      expect(result.content[0].text).toContain('TODO: Buy eggs'); // Current content
      expect(result.content[0].text).toContain('Current contents of'); // Error format
    } finally {
      // Clean up servers
      if (serverA) serverA.kill();
      if (serverB) serverB.kill();
    }
  }, 30000); // 30 second timeout

  it('should allow operations when content matches cached checksum', async () => {
    let server: ChildProcess | null = null;

    try {
      server = await startMCPServer(testRoot);

      // Create file
      const testFilePath = path.join(memoryPath, 'stable.txt');
      await fs.writeFile(testFilePath, 'Stable content', 'utf-8');

      // Read file (caches checksum)
      await callMemoryTool(server, 'view', { path: '/memories/stable.txt' }, 100);

      // Modify file - content matches, should succeed
      const modifyResponse = await callMemoryTool(
        server,
        'str_replace',
        {
          path: '/memories/stable.txt',
          old_str: 'Stable',
          new_str: 'Updated',
        },
        200,
      );

      expect(modifyResponse.error).toBeUndefined();

      // Verify modification succeeded
      const content = await fs.readFile(testFilePath, 'utf-8');
      expect(content).toBe('Updated content');
    } finally {
      if (server) server.kill();
    }
  }, 30000);

  it('should handle file creation without concurrency check', async () => {
    let server: ChildProcess | null = null;

    try {
      server = await startMCPServer(testRoot);

      // Create new file (no concurrency check needed)
      const createResponse = await callMemoryTool(
        server,
        'create',
        {
          path: '/memories/new-file.txt',
          file_text: 'New content',
        },
        100,
      );

      expect(createResponse.error).toBeUndefined();

      // Verify file exists
      const testFilePath = path.join(memoryPath, 'new-file.txt');
      const content = await fs.readFile(testFilePath, 'utf-8');
      expect(content).toBe('New content');
    } finally {
      if (server) server.kill();
    }
  }, 30000);
});
