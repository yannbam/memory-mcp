#!/usr/bin/env node

/**
 * Test script to verify concurrent read operations don't block each other
 * with the new @esfx/async-readerwriterlock implementation
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { spawn } from 'child_process';

// Number of concurrent clients to test
const NUM_CLIENTS = 5;
const TARGET_FILE = '/memories/test-nonunique.txt';

/**
 * Create and initialize an MCP client
 */
async function createClient(clientId) {
  // Spawn the memory-mcp server process
  const serverProcess = spawn('node', ['dist/index.js'], {
    stdio: ['pipe', 'pipe', 'inherit'],
    cwd: process.cwd()
  });

  // Create transport and client
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
    stderr: 'inherit'
  });

  const client = new Client({
    name: `concurrent-test-client-${clientId}`,
    version: '1.0.0'
  }, {
    capabilities: {}
  });

  // Connect
  await client.connect(transport);

  return { client, serverProcess };
}

/**
 * Perform a read operation and measure time
 */
async function performRead(client, clientId) {
  const startTime = Date.now();

  try {
    const result = await client.callTool({
      name: 'memory',
      arguments: {
        command: 'view',
        path: TARGET_FILE
      }
    });

    const duration = Date.now() - startTime;

    // Check if result contains error (MCP tools don't throw exceptions on errors)
    if (result.isError || (result.content && result.content.length > 0 && result.content[0].type === 'text' && result.content[0].text.includes('Error:'))) {
      const errorMsg = result.isError ? 'Tool returned error' : result.content[0].text;
      console.error(`[Client ${clientId}] Read failed after ${duration}ms:`, errorMsg);
      return { clientId, duration, success: false, error: errorMsg };
    }

    console.log(`[Client ${clientId}] Read completed in ${duration}ms`);
    return { clientId, duration, success: true };
  } catch (error) {
    const duration = Date.now() - startTime;
    console.error(`[Client ${clientId}] Read failed after ${duration}ms:`, error.message);

    return { clientId, duration, success: false, error: error.message };
  }
}

/**
 * Main test function
 */
async function runConcurrentReadTest() {
  console.log(`\n🧪 Testing concurrent reads with ${NUM_CLIENTS} clients\n`);
  console.log(`Target file: ${TARGET_FILE}\n`);

  const clients = [];

  try {
    // Create all clients
    console.log('Creating clients...');
    for (let i = 0; i < NUM_CLIENTS; i++) {
      const { client, serverProcess } = await createClient(i);
      clients.push({ client, serverProcess, id: i });
      console.log(`  ✓ Client ${i} ready`);
    }

    console.log(`\nStarting concurrent reads at ${new Date().toISOString()}...\n`);

    // Start all read operations concurrently
    const startTime = Date.now();
    const readPromises = clients.map(({ client, id }) =>
      performRead(client, id)
    );

    // Wait for all to complete
    const results = await Promise.all(readPromises);
    const totalTime = Date.now() - startTime;

    // Analyze results
    console.log(`\n📊 Results:\n`);
    console.log(`Total wall-clock time: ${totalTime}ms`);

    const successfulReads = results.filter(r => r.success);
    const avgDuration = successfulReads.reduce((sum, r) => sum + r.duration, 0) / successfulReads.length;
    const maxDuration = Math.max(...successfulReads.map(r => r.duration));
    const minDuration = Math.min(...successfulReads.map(r => r.duration));

    console.log(`Successful reads: ${successfulReads.length}/${NUM_CLIENTS}`);
    console.log(`Average duration: ${avgDuration.toFixed(2)}ms`);
    console.log(`Min duration: ${minDuration}ms`);
    console.log(`Max duration: ${maxDuration}ms`);

    // Analysis
    console.log(`\n💡 Analysis:\n`);
    if (totalTime < avgDuration * NUM_CLIENTS * 0.5) {
      console.log(`✅ PASS: Reads appear to be truly concurrent!`);
      console.log(`   Wall-clock time (${totalTime}ms) << sum of individual reads (${(avgDuration * NUM_CLIENTS).toFixed(0)}ms)`);
      console.log(`   This indicates reader-writer locks are allowing concurrent reads.`);
    } else {
      console.log(`⚠️  WARNING: Reads may be blocking each other`);
      console.log(`   Wall-clock time (${totalTime}ms) is close to sum of reads (${(avgDuration * NUM_CLIENTS).toFixed(0)}ms)`);
      console.log(`   This suggests operations are serialized, not concurrent.`);
    }

  } catch (error) {
    console.error('Test failed:', error);
  } finally {
    // Cleanup
    console.log(`\nCleaning up...`);
    for (const { client, serverProcess } of clients) {
      try {
        await client.close();
        serverProcess.kill();
      } catch (e) {
        // Ignore cleanup errors
      }
    }
    console.log('Done.\n');
  }
}

// Run the test
runConcurrentReadTest().catch(console.error);
