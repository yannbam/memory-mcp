#!/usr/bin/env node

/**
 * CRITICAL: Test that all memory commands properly return isError flag
 * when operations fail. This ensures MCP clients can detect failures.
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

async function createClient() {
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
    stderr: 'ignore'
  });

  const client = new Client({
    name: 'error-detection-test',
    version: '1.0.0'
  }, {
    capabilities: {}
  });

  await client.connect(transport);
  return client;
}

async function testCommand(client, testName, commandArgs, shouldFail = true, expectSchemaError = false) {
  try {
    const result = await client.callTool({
      name: 'memory',
      arguments: commandArgs
    });

    const hasFailed = result.isError === true;

    if (shouldFail && !hasFailed) {
      console.error(`❌ FAIL: ${testName}`);
      console.error(`   Expected isError=true, got isError=${result.isError}`);
      console.error(`   Result:`, JSON.stringify(result, null, 2));
      return false;
    } else if (!shouldFail && hasFailed) {
      console.error(`❌ FAIL: ${testName}`);
      console.error(`   Expected success, got isError=true`);
      console.error(`   Error:`, result.content[0].text);
      return false;
    } else {
      console.log(`✅ PASS: ${testName}`);
      return true;
    }
  } catch (error) {
    // Schema validation errors throw MCP protocol exceptions (error -32602)
    // These are expected for schema validation tests
    if (expectSchemaError && shouldFail && error.message.includes('-32602')) {
      console.log(`✅ PASS: ${testName} (schema validation error as expected)`);
      return true;
    }

    console.error(`❌ ERROR: ${testName} - Unexpected exception:`, error.message);
    return false;
  }
}

async function runErrorDetectionTests() {
  console.log('\n🧪 ERROR DETECTION TEST SUITE\n');
  console.log('Testing that all commands properly return isError flag on failures\n');

  const client = await createClient();
  const results = [];

  // Setup: Create test file
  await client.callTool({
    name: 'memory',
    arguments: {
      command: 'create',
      path: '/memories/error-test.txt',
      file_text: 'line one\nline two\nline three'
    }
  });

  try {
    console.log('--- VIEW Command ---');
    results.push(await testCommand(client, 'view non-existent file', {
      command: 'view',
      path: '/memories/does-not-exist.txt'
    }, true));

    results.push(await testCommand(client, 'view invalid line range', {
      command: 'view',
      path: '/memories/error-test.txt',
      view_range: [100, 200]
    }, true));

    results.push(await testCommand(client, 'view path traversal', {
      command: 'view',
      path: '/memories/../../../etc/passwd'
    }, true));

    console.log('\n--- CREATE Command ---');
    // Note: Create auto-creates parent dirs (like mkdir -p) - more user-friendly
    // than reference implementation which creates dir then throws error
    results.push(await testCommand(client, 'create auto-creates parent directory', {
      command: 'create',
      path: '/memories/auto-created-parent/file.txt',
      file_text: 'content'
    }, false)); // Should SUCCEED

    console.log('\n--- STR_REPLACE Command ---');
    results.push(await testCommand(client, 'str_replace text not found', {
      command: 'str_replace',
      path: '/memories/error-test.txt',
      old_str: 'does not exist',
      new_str: 'replacement'
    }, true));

    // Create file with non-unique text
    await client.callTool({
      name: 'memory',
      arguments: {
        command: 'create',
        path: '/memories/nonunique.txt',
        file_text: 'foo\nfoo\nfoo'
      }
    });

    results.push(await testCommand(client, 'str_replace non-unique text', {
      command: 'str_replace',
      path: '/memories/nonunique.txt',
      old_str: 'foo',
      new_str: 'bar'
    }, true));

    results.push(await testCommand(client, 'str_replace on non-existent file', {
      command: 'str_replace',
      path: '/memories/does-not-exist.txt',
      old_str: 'old',
      new_str: 'new'
    }, true));

    console.log('\n--- INSERT Command ---');
    results.push(await testCommand(client, 'insert invalid line number', {
      command: 'insert',
      path: '/memories/error-test.txt',
      insert_line: 999,
      insert_text: 'new line'
    }, true));

    results.push(await testCommand(client, 'insert on non-existent file', {
      command: 'insert',
      path: '/memories/does-not-exist.txt',
      insert_line: 0,
      insert_text: 'text'
    }, true));

    console.log('\n--- DELETE Command ---');
    results.push(await testCommand(client, 'delete non-existent file', {
      command: 'delete',
      path: '/memories/does-not-exist.txt'
    }, true));

    console.log('\n--- RENAME Command ---');
    results.push(await testCommand(client, 'rename non-existent file', {
      command: 'rename',
      old_path: '/memories/does-not-exist.txt',
      new_path: '/memories/new-name.txt'
    }, true));

    // Create destination file
    await client.callTool({
      name: 'memory',
      arguments: {
        command: 'create',
        path: '/memories/rename-dest.txt',
        file_text: 'dest'
      }
    });

    results.push(await testCommand(client, 'rename to existing destination', {
      command: 'rename',
      old_path: '/memories/error-test.txt',
      new_path: '/memories/rename-dest.txt'
    }, true));

    console.log('\n--- PATH SECURITY ---');
    results.push(await testCommand(client, 'path traversal in create', {
      command: 'create',
      path: '/memories/../../escape.txt',
      file_text: 'malicious'
    }, true));

    results.push(await testCommand(client, 'path traversal in delete', {
      command: 'delete',
      path: '/memories/../../../etc/passwd'
    }, true));

    console.log('\n--- SCHEMA VALIDATION ERRORS ---');
    // Schema validation errors throw MCP protocol exceptions (-32602)
    // Pass expectSchemaError=true to treat these as expected failures
    results.push(await testCommand(client, 'invalid command name', {
      command: 'invalid_command',
      path: '/memories/test.txt'
    }, true, true));

    results.push(await testCommand(client, 'missing path field in view', {
      command: 'view'
    }, true, true));

    results.push(await testCommand(client, 'missing file_text in create', {
      command: 'create',
      path: '/memories/test.txt'
    }, true, true));

    results.push(await testCommand(client, 'missing insert_line in insert', {
      command: 'insert',
      path: '/memories/error-test.txt',
      insert_text: 'text'
    }, true, true));

    results.push(await testCommand(client, 'missing insert_text in insert', {
      command: 'insert',
      path: '/memories/error-test.txt',
      insert_line: 1
    }, true, true));

    results.push(await testCommand(client, 'missing old_path in rename', {
      command: 'rename',
      new_path: '/memories/new.txt'
    }, true, true));

    results.push(await testCommand(client, 'missing new_path in rename', {
      command: 'rename',
      old_path: '/memories/old.txt'
    }, true, true));

    results.push(await testCommand(client, 'path as number instead of string', {
      command: 'view',
      path: 12345
    }, true, true));

    results.push(await testCommand(client, 'file_text as number instead of string', {
      command: 'create',
      path: '/memories/test.txt',
      file_text: 123
    }, true, true));

    results.push(await testCommand(client, 'invalid view_range format (single number)', {
      command: 'view',
      path: '/memories/error-test.txt',
      view_range: 5
    }, true, true));

    results.push(await testCommand(client, 'invalid view_range format (wrong length)', {
      command: 'view',
      path: '/memories/error-test.txt',
      view_range: [1, 2, 3]
    }, true, true));

    results.push(await testCommand(client, 'invalid view_range format (non-numeric)', {
      command: 'view',
      path: '/memories/error-test.txt',
      view_range: ['1', '10']
    }, true, true));

    // This one returns isError (not schema error) because string passes schema but fails executor
    results.push(await testCommand(client, 'non-numeric insert_line string', {
      command: 'insert',
      path: '/memories/error-test.txt',
      insert_line: 'not-a-number',
      insert_text: 'text'
    }, true, false));

    // Summary
    console.log('\n' + '='.repeat(60));
    const passed = results.filter(r => r).length;
    const total = results.length;
    const percentage = ((passed / total) * 100).toFixed(1);

    console.log(`\n📊 RESULTS: ${passed}/${total} tests passed (${percentage}%)\n`);

    if (passed === total) {
      console.log('✅ SUCCESS: All commands properly return isError flag on failures\n');
      process.exit(0);
    } else {
      console.error('❌ FAILURE: Some commands do not properly return isError flag\n');
      process.exit(1);
    }

  } finally {
    await client.close();
  }
}

console.log('Memory MCP Server - Error Detection Test Suite');
runErrorDetectionTests().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
