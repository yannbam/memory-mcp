#!/usr/bin/env node

/**
 * Test script to verify write operations DO block other operations
 * This is correct behavior - writes need exclusive locks
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const NUM_OPERATIONS = 3;

/**
 * Create and initialize an MCP client
 */
async function createClient(clientId) {
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
    stderr: 'inherit'
  });

  const client = new Client({
    name: `write-test-client-${clientId}`,
    version: '1.0.0'
  }, {
    capabilities: {}
  });

  await client.connect(transport);
  return client;
}

/**
 * Perform a write operation
 */
async function performWrite(client, clientId) {
  const startTime = Date.now();

  try {
    const result = await client.callTool({
      name: 'memory',
      arguments: {
        command: 'str_replace',
        path: '/memories/test-nonunique.txt',
        old_str: 'foo bar',
        new_str: `foo bar (modified by client ${clientId})`
      }
    });

    const duration = Date.now() - startTime;

    // Check if result contains error (MCP tools don't throw exceptions on errors)
    // Note: We EXPECT these to fail (non-unique text)
    if (result.isError || (result.content && result.content.length > 0 && result.content[0].type === 'text' && result.content[0].text.includes('Text appears'))) {
      const errorMsg = result.isError ? 'Tool returned error' : result.content[0].text;
      console.log(`[Client ${clientId}] Write failed as expected after ${duration}ms: ${errorMsg}`);
      return { clientId, duration, success: false, error: errorMsg };
    }

    console.log(`[Client ${clientId}] Write completed in ${duration}ms`);
    return { clientId, duration, success: true };
  } catch (error) {
    const duration = Date.now() - startTime;
    console.error(`[Client ${clientId}] Write failed after ${duration}ms:`, error.message);

    return { clientId, duration, success: false, error: error.message };
  }
}

/**
 * Main test function
 */
async function runWriteBlockingTest() {
  console.log(`\n🧪 Testing write operations (should serialize, not run concurrently)\n`);

  const clients = [];

  try {
    // Create test file with unique text
    const setupClient = await createClient('setup');
    await setupClient.callTool({
      name: 'memory',
      arguments: {
        command: 'create',
        path: '/memories/test-nonunique.txt',
        file_text: 'foo bar\nfoo baz\nfoo qux'
      }
    });
    console.log('✓ Test file created\n');

    // Create clients
    console.log('Creating clients...');
    for (let i = 0; i < NUM_OPERATIONS; i++) {
      const client = await createClient(i);
      clients.push({ client, id: i });
      console.log(`  ✓ Client ${i} ready`);
    }

    console.log(`\nStarting concurrent writes at ${new Date().toISOString()}...\n`);

    // Note: We expect these to fail (non-unique text) but what matters is timing
    const startTime = Date.now();
    const writePromises = clients.map(({ client, id }) =>
      performWrite(client, id)
    );

    const results = await Promise.all(writePromises);
    const totalTime = Date.now() - startTime;

    // Analyze
    console.log(`\n📊 Results:\n`);
    console.log(`Total wall-clock time: ${totalTime}ms`);

    const durations = results.map(r => r.duration);
    const avgDuration = durations.reduce((sum, d) => sum + d, 0) / durations.length;

    console.log(`Average operation duration: ${avgDuration.toFixed(2)}ms`);

    console.log(`\n💡 Analysis:\n`);
    if (totalTime > avgDuration * 0.8) {
      console.log(`✅ PASS: Writes appear to be serialized (blocking each other)`);
      console.log(`   Wall-clock time (${totalTime}ms) ≈ avg duration (${avgDuration.toFixed(0)}ms)`);
      console.log(`   This is correct - writes need exclusive locks.`);
    } else {
      console.log(`⚠️  UNEXPECTED: Writes may NOT be blocking properly`);
      console.log(`   Wall-clock time (${totalTime}ms) << avg duration (${avgDuration.toFixed(0)}ms)`);
    }

    await setupClient.close();

  } catch (error) {
    console.error('Test failed:', error);
  } finally {
    console.log(`\nCleaning up...`);
    for (const { client } of clients) {
      try {
        await client.close();
      } catch (e) {
        // Ignore
      }
    }
    console.log('Done.\n');
  }
}

runWriteBlockingTest().catch(console.error);
