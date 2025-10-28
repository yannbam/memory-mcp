# PR Review Action Plan
## Discriminated Union Implementation - Path to Main

**Branch**: `feature/discriminated-union-schema`
**Target**: `main`
**Current Status**: Functionally complete, needs type safety & test coverage fixes

---

## Phase 1: Critical Fixes (REQUIRED FOR MERGE)

**Estimated Time**: 4-6 hours
**Goal**: Clean lint, eliminate type safety escapes, fix error handling gaps

### 1.1 Fix str_replace Schema - Use Union Instead of Passthrough

**Problem**:
- `.passthrough()` accepts ANY extra fields (security risk)
- All four parameters optional (validates anything)
- Requires `as any` casts in executor (type safety escape)

**Solution**: Use discriminated union of two schemas

**File**: `src/memory/schemas.ts`

**Current Code** (lines 27-38):
```typescript
// UX: Accept both old_str/new_str and old_string/new_string parameter naming
// Use passthrough to allow both field names, normalize in command executor
const StrReplaceCommand = z
  .object({
    command: z.literal('str_replace'),
    path: z.string().describe('Path to file to modify'),
    old_str: z.string().optional().describe('Exact text to find (must be unique in file)'),
    old_string: z.string().optional(),
    new_str: z.string().optional().describe('Text to replace with'),
    new_string: z.string().optional(),
  })
  .passthrough();
```

**New Code**:
```typescript
// UX: Accept both old_str/new_str and old_string/new_string parameter naming
// TODO: Next session - find proper solution that prevents mixing conventions
// Current .passthrough() allows invalid combinations like {old_str, new_string}
// See task details for GPT-5 consultation plan
const StrReplaceCommand = z
  .object({
    command: z.literal('str_replace'),
    path: z.string().describe('Path to file to modify'),
    old_str: z.string().optional().describe('Exact text to find (must be unique in file)'),
    old_string: z.string().optional(),
    new_str: z.string().optional().describe('Text to replace with'),
    new_string: z.string().optional(),
  })
  .passthrough();
```

**Status**: ⚠️ DEFERRED - Initial solution was flawed
- Flattening union allows mixing conventions (e.g., {old_str, new_string})
- Need GPT-5 consultation for proper Zod schema approach
- See updated task details in pr-review-fixes plan

---

### 1.2 Fix Command Executor - Remove Type Safety Escapes

**Problem**: Three `as any` casts bypass TypeScript (11 linting errors)

**File**: `src/memory/command-executor.ts`

**Current Code** (lines 33-54):
```typescript
case 'str_replace': {
  // Normalize field names: accept both old_str/new_str and old_string/new_string
  const argsAny = args as any;  // ❌ Type safety escape #1
  const oldStr = args.old_str || argsAny.old_string;
  const newStr = args.new_str || argsAny.new_string;

  if (!oldStr) {
    throw new Error('Missing required field: old_str (or old_string)');
  }
  if (!newStr) {
    throw new Error('Missing required field: new_str (or new_string)');
  }

  return await operations.str_replace(
    {
      path: args.path,
      old_str: oldStr,
      new_str: newStr,
    } as any,  // ❌ Type safety escape #2
    context,
  );
}
```

**New Code** (with union schema from 1.1):
```typescript
case 'str_replace': {
  // Normalize field names to snake_case for operations layer
  // Union schema ensures exactly one set is present (no validation needed)
  const oldStr = 'old_str' in args ? args.old_str : args.old_string;
  const newStr = 'new_str' in args ? args.new_str : args.new_string;

  return await operations.str_replace(
    {
      path: args.path,
      old_str: oldStr,
      new_str: newStr,
    },
    context,
  );
}
```

**Benefits**:
- ✅ No type casts = full type safety restored
- ✅ TypeScript knows exactly which fields are present
- ✅ Fixes 11 linting errors
- ✅ Simpler code (no validation, schema handles it)

---

### 1.3 Remove Unused Import

**Problem**: `MemoryCommandSchema` imported but never used (1 linting error)

**File**: `src/memory/command-executor.ts:8`

**Change**:
```typescript
// Before:
import { MemoryCommandSchema, type MemoryCommand, assertNever } from './schemas.js';

// After:
import { type MemoryCommand, assertNever } from './schemas.js';
```

---

### 1.4 Fix Unnecessary Async Function

**Problem**: ListToolsRequestSchema handler marked async but has no await (1 linting error)

**File**: `src/server/mcp-server.ts:72`

**Current Code**:
```typescript
server.setRequestHandler(ListToolsRequestSchema, async (): Promise<ListToolsResult> => {
  return {
    tools: [
      {
        name: 'memory',
        description: /* ... */,
        inputSchema: memoryToolInputSchema as JSONSchema,
      },
    ],
  };
});
```

**New Code**:
```typescript
server.setRequestHandler(ListToolsRequestSchema, (): ListToolsResult => {
  return {
    tools: [
      {
        name: 'memory',
        description: /* ... */,
        inputSchema: memoryToolInputSchema as JSONSchema,
      },
    ],
  };
});
```

---

### 1.5 Add Error Logging to Command Execution

**Problem**: Errors returned to client but never logged server-side (debugging black hole)

**File**: `src/server/mcp-server.ts:116-138`

**Current Code**:
```typescript
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
```

**New Code**:
```typescript
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
```

**Benefits**:
- ✅ Server operators can see what errors users encounter
- ✅ Enables monitoring and alerting
- ✅ Debug logs include stack traces
- ✅ Helps identify error patterns

---

### 1.6 Fix Transport Cleanup Error Handling

**Problem**: `void transport.close()` ignores errors, risking unhandled promise rejection

**File**: `src/server/transports.ts:62`

**Current Code**:
```typescript
// Clean up transport when connection closes
res.on('close', () => {
  void transport.close();
});
```

**New Code**:
```typescript
// Clean up transport when connection closes
res.on('close', () => {
  transport.close().catch((error) => {
    // Log but don't throw - connection already closing
    console.error('Error closing transport:', error);
  });
});
```

---

### 1.7 Improve HTTP Error Messages

**Problem**: Generic "Internal server error" hides root cause from API consumers

**File**: `src/server/transports.ts:71-85`

**Current Code**:
```typescript
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
```

**New Code**:
```typescript
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
```

---

### 1.8 Fix Remaining Linting Issues

**Run and verify**:
```bash
npm run lint:fix
npm run lint  # Should show 0 errors
```

If any unfixable errors remain, address manually.

---

### Phase 1 Verification Checklist

- [ ] `npm run lint` shows 0 errors, 0 warnings
- [ ] `npm test` shows 85/85 passing (no regressions)
- [ ] `npm run build` succeeds without warnings
- [ ] Manual test: str_replace with both naming conventions works
- [ ] Manual test: Error triggers create debug log entries
- [ ] Review git diff for Phase 1 changes

**Expected Changes**:
- Modified: `src/memory/schemas.ts` (~15 lines changed)
- Modified: `src/memory/command-executor.ts` (~10 lines changed)
- Modified: `src/server/mcp-server.ts` (~10 lines changed)
- Modified: `src/server/transports.ts` (~15 lines changed)

---

## Phase 2: Test Coverage (STRONGLY RECOMMENDED)

**Estimated Time**: 4-6 hours
**Goal**: Add tests for new validation layer (0 → ~55 tests)

### 2.1 Schema Validation Tests

**New File**: `tests/unit/schema-validation.test.ts`

**Test Categories**:
1. **Valid Commands** (~12 tests)
   - One test per command with valid input
   - Test both old_str/new_str and old_string/new_string naming for str_replace
   - Test optional fields (view_range)

2. **Invalid Commands** (~13 tests)
   - Invalid command name
   - Missing required fields (one per command type)
   - Wrong field types (path as number, etc.)
   - Invalid view_range (not tuple, non-numeric)
   - Invalid insert_line (string that won't parse)

**Test Template**:
```typescript
import { MemoryCommandSchema } from '../../src/memory/schemas.js';
import { describe, it, expect } from '@jest/globals';

describe('Schema Validation', () => {
  describe('Valid commands', () => {
    it('should accept valid view command', () => {
      const result = MemoryCommandSchema.parse({
        command: 'view',
        path: '/memories',
      });
      expect(result.command).toBe('view');
      expect(result.path).toBe('/memories');
    });

    it('should accept str_replace with snake_case', () => {
      const result = MemoryCommandSchema.parse({
        command: 'str_replace',
        path: '/memories/test.txt',
        old_str: 'foo',
        new_str: 'bar',
      });
      expect(result).toHaveProperty('old_str', 'foo');
    });

    it('should accept str_replace with old_string/new_string', () => {
      const result = MemoryCommandSchema.parse({
        command: 'str_replace',
        path: '/memories/test.txt',
        old_string: 'foo',
        new_string: 'bar',
      });
      expect(result).toHaveProperty('old_string', 'foo');
    });

    // Add tests for create, insert, delete, rename...
  });

  describe('Invalid commands', () => {
    it('should reject invalid command name', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'invalid',
          path: '/memories',
        })
      ).toThrow();
    });

    it('should reject missing required path', () => {
      expect(() =>
        MemoryCommandSchema.parse({
          command: 'view',
        })
      ).toThrow(/path/i);
    });

    // Add more invalid input tests...
  });
});
```

**Estimated**: ~25 tests, 2-3 hours

---

### 2.2 Command Executor Tests

**New File**: `tests/unit/command-executor.test.ts`

**Test Categories**:
1. **Parameter Normalization - insert_line** (~8 tests)
   - Accept number: `insert_line: 5`
   - Accept numeric string: `insert_line: "5"`
   - Reject non-numeric: `insert_line: "abc"` → clear error
   - Reject float: `insert_line: "2.5"` → clear error
   - Reject negative: `insert_line: "-1"` → clear error
   - Accept string with whitespace: `insert_line: " 5 "` (or reject)
   - Very large number edge case
   - Zero line number edge case

2. **Parameter Normalization - str_replace** (~6 tests)
   - Accept snake_case only
   - Accept old_string/new_string only
   - With union schema, mixing is impossible (validated by schema)

3. **Command Dispatch** (~6 tests)
   - Each command variant dispatches correctly
   - Verify correct operation function called
   - Verify parameters passed correctly

4. **Error Handling** (~8 tests)
   - Invalid insert_line produces clear error message
   - File not found propagates correctly
   - Path validation errors propagate
   - Each error includes helpful context

**Test Template**:
```typescript
import { executeMemoryCommand } from '../../src/memory/command-executor.js';
import type { OperationsContext } from '../../src/memory/operations.js';
import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import * as fs from 'fs/promises';
import * as path from 'path';

describe('Command Executor', () => {
  let testDir: string;
  let context: OperationsContext;

  beforeEach(async () => {
    testDir = path.join('/tmp', `test-executor-${Date.now()}`);
    await fs.mkdir(testDir, { recursive: true });

    context = {
      memoryRoot: testDir,
      logger: { /* mock logger */ },
      treeView: false,
    };
  });

  afterEach(async () => {
    await fs.rm(testDir, { recursive: true, force: true });
  });

  describe('insert_line normalization', () => {
    beforeEach(async () => {
      await fs.writeFile(path.join(testDir, 'test.txt'), 'line1\nline2\nline3\n');
    });

    it('should accept insert_line as number', async () => {
      const result = await executeMemoryCommand(
        {
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: 2,
          insert_text: 'inserted',
        },
        context
      );
      expect(result).toContain('inserted');
    });

    it('should accept insert_line as numeric string', async () => {
      const result = await executeMemoryCommand(
        {
          command: 'insert',
          path: '/memories/test.txt',
          insert_line: '2',
          insert_text: 'inserted',
        },
        context
      );
      expect(result).toContain('inserted');
    });

    it('should reject insert_line as non-numeric string', async () => {
      await expect(
        executeMemoryCommand(
          {
            command: 'insert',
            path: '/memories/test.txt',
            insert_line: 'abc' as any,
            insert_text: 'inserted',
          },
          context
        )
      ).rejects.toThrow(/Invalid insert_line/);
    });

    // Add more normalization tests...
  });

  describe('str_replace parameter naming', () => {
    beforeEach(async () => {
      await fs.writeFile(path.join(testDir, 'test.txt'), 'foo bar baz');
    });

    it('should accept snake_case (old_str/new_str)', async () => {
      const result = await executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          old_str: 'bar',
          new_str: 'qux',
        },
        context
      );
      expect(result).toContain('qux');
    });

    it('should accept old_string/new_string naming', async () => {
      const result = await executeMemoryCommand(
        {
          command: 'str_replace',
          path: '/memories/test.txt',
          old_string: 'bar',
          new_string: 'qux',
        },
        context
      );
      expect(result).toContain('qux');
    });

    // Add more naming tests...
  });
});
```

**Estimated**: ~30 tests, 2-3 hours

---

### 2.3 MCP Error Handling Integration Tests

**Extend**: `tests/integration/test-error-detection.js`

**New Test Cases** (~10 tests):
```javascript
// Add after existing tests

console.log('\n--- SCHEMA VALIDATION ERRORS ---');

// Invalid command name
results.push(await testCommand(client, 'invalid command name', {
  command: 'invalid',
  path: '/memories'
}, true));

// Missing required field
results.push(await testCommand(client, 'missing path field', {
  command: 'view'
}, true));

// Wrong field type
results.push(await testCommand(client, 'path as number', {
  command: 'view',
  path: 123
}, true));

// Invalid insert_line
results.push(await testCommand(client, 'insert_line non-numeric', {
  command: 'insert',
  path: '/memories/test.txt',
  insert_line: 'abc',
  insert_text: 'text'
}, true));

// Add more schema validation error tests...
```

**Estimated**: ~10 tests, 1 hour

---

### Phase 2 Verification Checklist

- [ ] `npm test` shows ~140/140 passing (85 existing + ~55 new)
- [ ] Test coverage includes schema validation
- [ ] Test coverage includes parameter normalization
- [ ] Test coverage includes error messages
- [ ] All edge cases identified in review are tested
- [ ] Tests demonstrate Claude Code compatibility (both naming styles work)

---

## Phase 3: Documentation Improvements (RECOMMENDED)

**Estimated Time**: 1-2 hours
**Goal**: Fix misleading/contradictory comments

### 3.1 Fix Critical Comment Issues

**File**: `src/memory/schemas.ts:27-28`

**Change**:
```typescript
// Before:
// UX: Accept both old_str/new_str and old_string/new_string parameter naming
// Use passthrough to allow both field names, normalize in command executor

// After:
// UX: Accept both old_str/new_str and old_string/new_string parameter naming
// Union schema allows users to use either convention naturally
// Both styles validated identically, normalized to snake_case in executor
```

---

**File**: `src/server/transports.ts:46`

**Change**:
```typescript
// Before:
exposedHeaders: ['Mcp-Session-Id'], // Required for session management

// After:
exposedHeaders: ['Mcp-Session-Id'], // Exposed for future stateful mode support (currently stateless)
```

---

**File**: `src/server/mcp-server.ts:115`

**Change**:
```typescript
// Before:
// Execute command with full type safety

// After:
// Execute command with type-safe dispatch
```

---

### 3.2 Improve Workaround Documentation

**File**: `src/memory/command-executor.ts:59`

**Change**:
```typescript
// Before:
// Normalize insert_line to number (handles Claude Code serialization issue)

// After:
// Normalize insert_line to number
// Claude Code's MCP client may serialize numeric parameters as strings (as of 2025-01)
// Accept both types for compatibility
```

---

**File**: `src/memory/schemas.ts:43-45`

**Change**:
```typescript
// Before:
// WORKAROUND: Accept both number and string for insert_line
// Claude Code's MCP parameter serialization may send numeric params as strings

// After:
// COMPATIBILITY: Accept both number and string for insert_line
// Claude Code's MCP client serializes all parameters as strings (as of v1.x)
// TODO: Test if string-only after Claude Code adds proper type marshalling
```

---

### 3.3 Add Protocol References

**File**: `src/server/mcp-server.ts:64-69`

**Change**:
```typescript
// Before:
// MCP protocol requires type: "object" at top level
// Add it to satisfy protocol validation while keeping anyOf structure

// After:
// MCP protocol requires type: "object" at top level for tool input schemas
// See: https://spec.modelcontextprotocol.io/specification/server/tools/
// Add it to satisfy protocol validation while keeping oneOf discriminated union
```

---

### 3.4 Fix Terminology Inconsistency

**Files**: `src/server/mcp-server.ts:68, 89`

**Change**: Replace "anyOf" with "oneOf" (discriminated unions use oneOf, not anyOf)

---

### 3.5 Remove Redundant Comments

**File**: `src/memory/command-executor.ts:23-24`

**Remove**:
```typescript
// TypeScript provides exhaustive checking here
// Each case has automatic type narrowing
```

(Individual case comments already explain this)

---

**File**: `src/server/mcp-server.ts:48-53`

**Change**:
```typescript
// Before:
// Create operations context

// After:
// Bundle configuration for memory operations (passed to all command handlers)
```

---

### Phase 3 Verification Checklist

- [ ] All comments accurately reflect code behavior
- [ ] No contradictions between comments and implementation
- [ ] Workarounds documented with version context
- [ ] Protocol references included where relevant
- [ ] Redundant/vague comments removed
- [ ] Terminology consistent (oneOf for discriminated unions)

---

## Phase 4: Polish (POST-MERGE OK)

**Estimated Time**: 2-3 hours
**Goal**: Production-grade improvements (can be follow-up PR)

### 4.1 Consider Branded Types for Paths

**File**: `src/memory/path-security.ts`

Add branded type for memory paths to enforce `/memories` prefix at type level:

```typescript
const MEMORY_PATH_BRAND = Symbol('MemoryPath');
export type MemoryPath = string & { readonly [MEMORY_PATH_BRAND]: true };

export function validateMemoryPath(path: string): MemoryPath {
  if (!path.startsWith('/memories')) {
    throw new Error('Path must start with /memories');
  }
  // ... existing validation
  return path as MemoryPath;
}
```

Update schemas to use branded type validation.

---

### 4.2 Make CORS Configurable

**File**: `src/server/transports.ts:45`

**Change**:
```typescript
// Before:
origin: '*', // Allow all origins

// After:
origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
// Allow all origins in development, configure for production via ALLOWED_ORIGINS env var
```

Add to README.md environment variables section.

---

### 4.3 Improve Error Messages with Troubleshooting

**File**: `src/index.ts:220-223`

Add helpful diagnostics to startup errors:

```typescript
} catch (error) {
  const errorMessage = error instanceof Error ? error.message : String(error);
  const errorStack = error instanceof Error ? error.stack : undefined;

  console.error('Fatal startup error:', errorMessage);
  if (errorStack && config.debug) {
    console.error('Stack trace:', errorStack);
  }

  console.error('\nTroubleshooting:');
  console.error('- Check memory root path exists and is writable:', config.memoryRootPath);
  console.error('- Check port is available (HTTP mode):', config.port);
  console.error('- Enable debug mode: --debug flag');
  console.error('- View debug logs: /tmp/memory-mcp/<instance-id>.log');

  process.exit(1);
}
```

---

### 4.4 Handle Signal Handler Errors

**File**: `src/index.ts:209-219`

**Change**:
```typescript
process.on('SIGINT', async () => {
  console.error('\nShutting down...');
  try {
    await logger.close();
    process.exit(0);
  } catch (error) {
    console.error('Error during shutdown:', error);
    process.exit(1);
  }
});

process.on('SIGTERM', async () => {
  console.error('Received SIGTERM, shutting down...');
  try {
    await logger.close();
    process.exit(0);
  } catch (error) {
    console.error('Error during shutdown:', error);
    process.exit(1);
  }
});
```

---

### Phase 4 Verification Checklist

- [ ] Branded types implemented (optional)
- [ ] CORS configurable via environment (optional)
- [ ] Error messages include troubleshooting hints
- [ ] Signal handlers properly handle cleanup errors
- [ ] README updated with new env vars if added

---

## Summary Timeline

| Phase | Status | Time | Tests | Outcome |
|-------|--------|------|-------|---------|
| Phase 1 | **REQUIRED** | 4-6h | 85 → 85 | Clean lint, type safety, error handling |
| Phase 2 | **RECOMMENDED** | 4-6h | 85 → 140 | Comprehensive test coverage |
| Phase 3 | Recommended | 1-2h | 140 → 140 | Accurate documentation |
| Phase 4 | Optional | 2-3h | 140 → 140 | Production polish |

**Total for Production-Ready Merge**: 9-14 hours (Phases 1-3)
**Minimum for Merge**: 4-6 hours (Phase 1 only, with TODO comments)

---

## Critical Success Factors

1. **Maintain Claude Code Compatibility**
   - Union schema approach preserves both naming conventions
   - insert_line still accepts number | string
   - All E2E tested functionality continues to work

2. **Type Safety Without Compromises**
   - Zero `as any` casts after Phase 1
   - Schema-driven types (not parallel hierarchies)
   - TypeScript enforces correctness

3. **Production Debugging Capability**
   - All errors logged server-side
   - Meaningful error messages for API consumers
   - No silent failures

4. **Test Coverage for Confidence**
   - New validation layer fully tested
   - Edge cases documented in tests
   - Regression protection for future refactoring

---

## Next Steps

1. **Create feature branch** for Phase 1 fixes (if not using existing)
2. **Implement Phase 1** changes systematically
3. **Verify** with checklist after each change
4. **Test manually** with Claude Code instance
5. **Run full test suite** before commit
6. **Decide** on Phase 2 (recommended) vs. merge with TODOs
7. **Update** CLAUDE.md handoff section
8. **Merge** to dev, then main

---

**Document Version**: 1.0
**Created**: 2025-10-28
**Author**: Claude Code Review (Session 0bb07c86)
