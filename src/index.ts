#!/usr/bin/env node

/**
 * Memory MCP Server - Main Entry Point
 *
 * MCP server implementation of Claude's native memory tool.
 * Provides persistent storage across conversations through filesystem operations.
 *
 * Supports:
 * - stdio and streamable HTTP transports
 * - Concurrent access from multiple Claude instances
 * - Path security with directory traversal protection
 * - Optional debug logging
 */

import * as path from 'path';
import * as fs from 'fs/promises';
import { createMemoryServer } from './server/mcp-server.js';
import { initStdioTransport, initHttpTransport } from './server/transports.js';
import { createLogger } from './utils/logger.js';

/**
 * CLI configuration
 */
interface CliConfig {
  memoryRootPath: string;
  transport: 'stdio' | 'http';
  port: number;
  debug: boolean;
}

/**
 * Parse command-line arguments
 *
 * @param args - Process arguments (typically process.argv.slice(2))
 * @returns Parsed configuration
 */
function parseArgs(args: string[]): CliConfig {
  // Default configuration
  const config: CliConfig = {
    memoryRootPath: './.memory',
    transport: 'stdio',
    port: 3000,
    debug: false,
  };

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case '--memory-root-path':
      case '-m':
        // Get next argument as path
        if (i + 1 >= args.length) {
          console.error('Error: --memory-root-path requires a path argument');
          process.exit(1);
        }
        config.memoryRootPath = args[++i];
        break;

      case '--transport':
      case '-t':
        // Get next argument as transport type
        if (i + 1 >= args.length) {
          console.error('Error: --transport requires a type argument (stdio or http)');
          process.exit(1);
        }
        const transport = args[++i];
        if (transport !== 'stdio' && transport !== 'http') {
          console.error('Error: --transport must be either "stdio" or "http"');
          process.exit(1);
        }
        config.transport = transport;
        break;

      case '--port':
      case '-p':
        // Get next argument as port number
        if (i + 1 >= args.length) {
          console.error('Error: --port requires a port number');
          process.exit(1);
        }
        const portStr = args[++i];
        const port = parseInt(portStr, 10);
        if (isNaN(port) || port < 1 || port > 65535) {
          console.error('Error: --port must be a valid port number (1-65535)');
          process.exit(1);
        }
        config.port = port;
        break;

      case '--debug':
      case '-d':
        // Enable debug logging
        config.debug = true;
        break;

      case '--version':
      case '-v':
        // Show version and exit
        console.log('memory-mcp version 0.1.0');
        process.exit(0);
        break;

      case '--help':
      case '-h':
        // Show help and exit
        showHelp();
        process.exit(0);
        break;

      default:
        console.error(`Error: Unknown argument: ${arg}`);
        console.error('Use --help to see available options');
        process.exit(1);
    }
  }

  return config;
}

/**
 * Display help message
 */
function showHelp(): void {
  console.log(`
Memory MCP Server

MCP server implementation of Claude's native memory tool.
Provides persistent storage across conversations through filesystem operations.

USAGE:
  memory-mcp [options]

OPTIONS:
  --memory-root-path PATH, -m PATH   Memory storage root path (default: ./.memory)
  --transport TYPE, -t TYPE          Transport type: stdio | http (default: stdio)
  --port PORT, -p PORT               HTTP server port (default: 3000, http transport only)
  --debug, -d                        Enable debug logging to /tmp/memory-mcp/<instance-id>.log
  --version, -v                      Show version
  --help, -h                         Show this help message

EXAMPLES:
  # stdio transport (default)
  memory-mcp

  # stdio with custom root path
  memory-mcp --memory-root-path /my/data/.memory

  # HTTP transport on port 3000
  memory-mcp --transport http --port 3000

  # With debug logging
  memory-mcp --debug --transport http

For more information, visit: https://github.com/janbam/memory-mcp
  `);
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  // Parse command-line arguments
  const config = parseArgs(process.argv.slice(2));

  try {
    // Resolve memory root path to absolute path
    const memoryRoot = path.resolve(config.memoryRootPath);

    // Ensure memory root directory exists
    await fs.mkdir(path.join(memoryRoot, 'memories'), { recursive: true });

    // Initialize logger
    const logger = await createLogger(config.debug);

    // Log startup configuration
    if (config.debug) {
      await logger.debug('startup', {
        memoryRoot,
        transport: config.transport,
        port: config.port,
      });
    }

    // Create MCP server
    const server = createMemoryServer(path.join(memoryRoot, 'memories'), logger);

    // Initialize transport
    if (config.transport === 'stdio') {
      await initStdioTransport(server);
    } else {
      await initHttpTransport(server, config.port);
    }

    // Handle graceful shutdown
    process.on('SIGINT', async () => {
      console.error('\nShutting down...');
      await logger.close();
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      console.error('\nShutting down...');
      await logger.close();
      process.exit(0);
    });
  } catch (error) {
    console.error('Fatal error:', error);
    process.exit(1);
  }
}

// Run main function
main();
