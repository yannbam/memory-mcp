#!/usr/bin/env node

/**
 * Stress test: 50+ concurrent clients performing mixed operations
 * Tests system behavior under load with RW locks
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

// Configuration
const NUM_READERS = 40;
const NUM_WRITERS = 10;
const TOTAL_CLIENTS = NUM_READERS + NUM_WRITERS;

/**
 * Create and initialize an MCP client
 */
async function createClient(clientId, role) {
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
    stderr: 'ignore'  // Reduce noise during stress test
  });

  const client = new Client({
    name: `stress-test-${role}-${clientId}`,
    version: '1.0.0'
  }, {
    capabilities: {}
  });

  await client.connect(transport);
  return client;
}

/**
 * Reader operation - reads a file
 */
async function performRead(client, clientId) {
  const startTime = Date.now();

  try {
    const result = await client.callTool({
      name: 'memory',
      arguments: {
        command: 'view',
        path: '/memories/stress-test.txt'
      }
    });

    const duration = Date.now() - startTime;

    // Check if result contains error
    if (result.isError || (result.content && result.content.length > 0 && result.content[0].type === 'text' && result.content[0].text.includes('Error:'))) {
      const errorMsg = result.isError ? 'Tool returned error' : result.content[0].text;
      return { clientId, role: 'reader', duration, success: false, error: errorMsg };
    }

    return { clientId, role: 'reader', duration, success: true };
  } catch (error) {
    const duration = Date.now() - startTime;
    return { clientId, role: 'reader', duration, success: false, error: error.message };
  }
}

/**
 * Writer operation - appends to a file
 */
async function performWrite(client, clientId) {
  const startTime = Date.now();

  try {
    // Create a unique file for this writer to avoid conflicts
    const result = await client.callTool({
      name: 'memory',
      arguments: {
        command: 'create',
        path: `/memories/stress-test-write-${clientId}.txt`,
        file_text: `Write from client ${clientId} at ${new Date().toISOString()}`
      }
    });

    const duration = Date.now() - startTime;

    // Check if result contains error
    if (result.isError || (result.content && result.content.length > 0 && result.content[0].type === 'text' && result.content[0].text.includes('Error:'))) {
      const errorMsg = result.isError ? 'Tool returned error' : result.content[0].text;
      return { clientId, role: 'writer', duration, success: false, error: errorMsg };
    }

    return { clientId, role: 'writer', duration, success: true };
  } catch (error) {
    const duration = Date.now() - startTime;
    return { clientId, role: 'writer', duration, success: false, error: error.message };
  }
}

/**
 * Progress bar display
 */
function showProgress(current, total, label) {
  const percent = Math.floor((current / total) * 100);
  const barLength = 40;
  const filled = Math.floor((current / total) * barLength);
  const bar = '█'.repeat(filled) + '░'.repeat(barLength - filled);
  process.stdout.write(`\r${label}: [${bar}] ${percent}% (${current}/${total})`);
}

/**
 * Main stress test
 */
async function runStressTest() {
  console.log(`\n🔥 STRESS TEST: ${TOTAL_CLIENTS} Concurrent Clients\n`);
  console.log(`Configuration:`);
  console.log(`  - ${NUM_READERS} concurrent readers`);
  console.log(`  - ${NUM_WRITERS} concurrent writers`);
  console.log(`  - Target file: /memories/stress-test.txt\n`);

  const clients = [];
  const setupClient = await createClient('setup', 'setup');

  try {
    // Create initial test file
    console.log('Setting up test file...');
    await setupClient.callTool({
      name: 'memory',
      arguments: {
        command: 'create',
        path: '/memories/stress-test.txt',
        file_text: 'Initial content for stress test\n'
      }
    });
    console.log('✓ Test file created\n');

    // Create all clients with progress
    console.log('Creating clients...');
    for (let i = 0; i < TOTAL_CLIENTS; i++) {
      const role = i < NUM_READERS ? 'reader' : 'writer';
      const client = await createClient(i, role);
      clients.push({ client, id: i, role });
      showProgress(i + 1, TOTAL_CLIENTS, 'Client creation');
    }
    console.log('\n');

    console.log(`Starting operations at ${new Date().toISOString()}...\n`);

    // Launch all operations concurrently
    const startTime = Date.now();
    const operations = clients.map(({ client, id, role }) =>
      role === 'reader' ? performRead(client, id) : performWrite(client, id)
    );

    // Wait for all to complete
    const results = await Promise.all(operations);
    const totalTime = Date.now() - startTime;

    // Analyze results
    console.log(`\n📊 RESULTS:\n`);
    console.log(`Total wall-clock time: ${totalTime}ms`);

    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);

    const readers = results.filter(r => r.role === 'reader');
    const writers = results.filter(r => r.role === 'writer');

    const successfulReaders = readers.filter(r => r.success);
    const successfulWriters = writers.filter(r => r.success);

    console.log(`\nSuccess rate:`);
    console.log(`  Total: ${successful.length}/${TOTAL_CLIENTS} (${(successful.length/TOTAL_CLIENTS*100).toFixed(1)}%)`);
    console.log(`  Readers: ${successfulReaders.length}/${NUM_READERS} (${(successfulReaders.length/NUM_READERS*100).toFixed(1)}%)`);
    console.log(`  Writers: ${successfulWriters.length}/${NUM_WRITERS} (${(successfulWriters.length/NUM_WRITERS*100).toFixed(1)}%)`);

    if (successful.length > 0) {
      const avgDuration = successful.reduce((sum, r) => sum + r.duration, 0) / successful.length;
      const maxDuration = Math.max(...successful.map(r => r.duration));
      const minDuration = Math.min(...successful.map(r => r.duration));

      console.log(`\nTiming:`);
      console.log(`  Average: ${avgDuration.toFixed(2)}ms`);
      console.log(`  Min: ${minDuration}ms`);
      console.log(`  Max: ${maxDuration}ms`);

      if (successfulReaders.length > 0) {
        const avgReaderTime = successfulReaders.reduce((sum, r) => sum + r.duration, 0) / successfulReaders.length;
        console.log(`  Avg reader: ${avgReaderTime.toFixed(2)}ms`);
      }

      if (successfulWriters.length > 0) {
        const avgWriterTime = successfulWriters.reduce((sum, r) => sum + r.duration, 0) / successfulWriters.length;
        console.log(`  Avg writer: ${avgWriterTime.toFixed(2)}ms`);
      }
    }

    // Show failures if any
    if (failed.length > 0) {
      console.log(`\n⚠️  Failures (${failed.length}):`);
      const errorCounts = {};
      failed.forEach(f => {
        const errMsg = f.error || 'Unknown error';
        errorCounts[errMsg] = (errorCounts[errMsg] || 0) + 1;
      });
      Object.entries(errorCounts).forEach(([error, count]) => {
        console.log(`  - ${error}: ${count}x`);
      });
    }

    // Analysis
    console.log(`\n💡 ANALYSIS:\n`);

    const theoreticalReaderTime = successfulReaders.reduce((sum, r) => sum + r.duration, 0);
    const theoreticalWriterTime = successfulWriters.reduce((sum, r) => sum + r.duration, 0);

    console.log(`Theoretical time if all operations were serialized:`);
    console.log(`  Readers: ${theoreticalReaderTime}ms`);
    console.log(`  Writers: ${theoreticalWriterTime}ms`);
    console.log(`  Total: ${theoreticalReaderTime + theoreticalWriterTime}ms`);

    console.log(`\nActual wall-clock time: ${totalTime}ms`);

    const speedup = ((theoreticalReaderTime + theoreticalWriterTime) / totalTime).toFixed(2);
    console.log(`Speedup factor: ${speedup}x`);

    if (speedup > 5) {
      console.log(`\n✅ EXCELLENT: System handles concurrent operations very well!`);
      console.log(`   RW locks are providing significant concurrency benefits.`);
    } else if (speedup > 2) {
      console.log(`\n✅ GOOD: Decent concurrency behavior under load.`);
    } else {
      console.log(`\n⚠️  WARNING: Limited concurrency benefit observed.`);
      console.log(`   Operations may be blocking more than expected.`);
    }

    // Verify file integrity - check that all write files were created
    console.log(`\nVerifying file integrity...`);
    const dirView = await setupClient.callTool({
      name: 'memory',
      arguments: {
        command: 'view',
        path: '/memories'
      }
    });

    const fileList = dirView.content[0].text;
    const writeFilesCreated = successfulWriters.filter(w =>
      fileList.includes(`stress-test-write-${w.clientId}.txt`)
    ).length;

    console.log(`Write files created: ${writeFilesCreated}/${successfulWriters.length}`);

    if (writeFilesCreated === successfulWriters.length) {
      console.log(`✅ File integrity verified: All ${successfulWriters.length} writes persisted`);
    } else {
      console.log(`⚠️  Expected ${successfulWriters.length} write files, found ${writeFilesCreated}`);
    }

  } catch (error) {
    console.error('\n❌ Stress test failed:', error);
    throw error;
  } finally {
    console.log(`\nCleaning up...`);
    await setupClient.close();

    let cleaned = 0;
    for (const { client } of clients) {
      try {
        await client.close();
        cleaned++;
        showProgress(cleaned, clients.length, 'Cleanup');
      } catch (e) {
        // Ignore cleanup errors
      }
    }
    console.log('\n\n✅ Stress test complete.\n');
  }
}

// Run the test
console.log('Starting stress test...');
runStressTest().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
