/**
 * Transport Layer
 *
 * Initializes and configures stdio and streamable HTTP transports.
 */

import express from 'express';
import cors from 'cors';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { Server } from '@modelcontextprotocol/sdk/server/index.js';

/**
 * Initialize stdio transport and connect to server
 *
 * @param server - MCP server instance (base Server class)
 */
export async function initStdioTransport(server: Server): Promise<void> {
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
 * @param server - MCP server instance (base Server class)
 * @param port - Port number to listen on
 */
export async function initHttpTransport(server: Server, port: number): Promise<void> {
  // Create Express app
  const app = express();

  // Enable JSON parsing
  app.use(express.json());

  // Enable CORS for browser-based clients
  app.use(
    cors({
      origin: '*', // Allow all origins
      exposedHeaders: ['Mcp-Session-Id'], // Exposed for future stateful mode support (currently stateless)
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
      transport.close().catch((error) => {
        // Log but don't throw - connection already closing
        console.error('Error closing transport:', error);
      });
    });

    try {
      // Connect server to transport
      await server.connect(transport);

      // Handle the request
      await transport.handleRequest(req, res, req.body);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error('Error handling MCP request:', errorMessage);

      // Send error response if headers not sent
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: '2.0',
          error: {
            code: -32603,
            message: 'Internal server error',
            data: {
              detail: errorMessage,
              // Include stack trace in debug mode only
              stack: process.env.DEBUG ? (error instanceof Error ? error.stack : undefined) : undefined,
            },
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
