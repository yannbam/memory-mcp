/**
 * Transport Layer
 *
 * Initializes and configures stdio and streamable HTTP transports.
 */

import express from 'express';
import cors from 'cors';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

/**
 * Initialize stdio transport and connect to server
 *
 * @param server - MCP server instance
 */
export async function initStdioTransport(server: McpServer): Promise<void> {
  // Create stdio transport
  const transport = new StdioServerTransport();

  // Connect server to transport
  await server.connect(transport);

  // Log to stderr (stdout is used for MCP protocol)
  console.error('Memory MCP Server running on stdio transport');
}

/**
 * Initialize HTTP transport with Express and connect to server
 *
 * @param server - MCP server instance
 * @param port - Port number to listen on
 */
export async function initHttpTransport(server: McpServer, port: number): Promise<void> {
  // Create Express app
  const app = express();

  // Enable JSON parsing
  app.use(express.json());

  // Enable CORS for browser-based clients
  app.use(
    cors({
      origin: '*', // Allow all origins
      exposedHeaders: ['Mcp-Session-Id'], // Required for session management
      allowedHeaders: ['Content-Type', 'mcp-session-id'],
    }),
  );

  // Handle MCP requests (stateless mode)
  app.post('/mcp', async (req, res) => {
    // Create new transport for each request to prevent ID collisions
    // In stateless mode, different clients may use same JSON-RPC request IDs
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined, // Stateless mode
      enableJsonResponse: true,
    });

    // Clean up transport when connection closes
    res.on('close', () => {
      transport.close();
    });

    try {
      // Connect server to transport
      await server.connect(transport);

      // Handle the request
      await transport.handleRequest(req, res, req.body);
    } catch (error) {
      console.error('Error handling MCP request:', error);

      // Send error response if headers not sent
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: '2.0',
          error: {
            code: -32603,
            message: 'Internal server error',
          },
          id: null,
        });
      }
    }
  });

  // Start HTTP server
  await new Promise<void>((resolve, reject) => {
    const httpServer = app.listen(port, () => {
      console.error(`Memory MCP Server running on http://localhost:${port}/mcp`);
      resolve();
    });

    httpServer.on('error', (error) => {
      console.error('HTTP server error:', error);
      reject(error);
    });
  });
}
