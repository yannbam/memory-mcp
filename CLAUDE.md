# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server implementation of Claude's native Memory tool. It provides persistent storage across Claude Code sessions through filesystem operations, enabling Claude to:
- Store and retrieve information across conversations
- Build knowledge over time without context window limitations
- Learn from past interactions and maintain project context

**Current Status**: Infrastructure complete, implementation pending.

## Architecture

### Core Design Pattern

The project implements Claude's Memory tool specification as an MCP server using the `@modelcontextprotocol/sdk`. The architecture consists of:

1. **Memory Commands** (6 operations matching the Memory tool spec):
   - `view` - Directory listings or file contents with optional line ranges
   - `create` - Create/overwrite files
   - `str_replace` - Replace unique text in files
   - `insert` - Insert text at specific line numbers
   - `delete` - Remove files/directories
   - `rename` - Move/rename files/directories

2. **Path Security Layer**: All operations must validate paths to prevent directory traversal attacks. All paths must:
   - Start with `/memories`
   - Be resolved to canonical form and verified to remain within the memory root
   - Reject patterns like `../`, `..\\`, URL-encoded traversal (`%2e%2e%2f`)

3. **MCP Server Integration**: Uses `@modelcontextprotocol/sdk` to expose memory operations as MCP tools with Zod schemas for validation.

### Key References

- **Memory Tool Specification**: `docs/039-Memory-tool.md` - Official Claude Memory tool documentation
- **MCP SDK Documentation**: `docs/MCP-SDK-README.md` - TypeScript SDK reference
- **Implementation Example**: `docs/tools-helpers-memory.ts` - TypeScript reference implementation (filesystem-based)

## Development Commands

### Build and Run
```bash
npm run build          # Compile TypeScript to dist/
npm run watch          # Watch mode for development
npm run dev            # Build and run the server
```

### Testing
```bash
npm test               # Run all tests
npm run test:watch     # Watch mode for tests
npm run test:coverage  # Run tests with coverage report (80% threshold)
```

### Code Quality
```bash
npm run lint           # Check for linting issues
npm run lint:fix       # Auto-fix linting issues
```

## TypeScript Configuration

### ES Modules with Node16

This project uses **ES Modules** (`"type": "module"` in package.json) with Node16 module resolution:
- Target: `es2018` (aligned with MCP SDK)
- Module: `Node16`
- `isolatedModules: true` required for ts-jest

### Testing with ESM

Jest requires special configuration for ESM:
- Tests use `NODE_OPTIONS=--experimental-vm-modules` to enable ESM support
- Jest config uses `preset: 'ts-jest/presets/default-esm'`
- This is **critical** - tests will fail without the NODE_OPTIONS flag

## Security Considerations

### Path Traversal Protection (CRITICAL)

When implementing memory commands, **ALL paths must be validated** to prevent security vulnerabilities:

```typescript
// Example validation pattern from TypeScript reference
private validatePath(memoryPath: string): string {
  if (!memoryPath.startsWith('/memories')) {
    throw new Error(`Path must start with /memories`);
  }

  const fullPath = path.join(this.memoryRoot, relativePath);
  const resolvedPath = path.resolve(fullPath);
  const resolvedRoot = path.resolve(this.memoryRoot);

  if (!resolvedPath.startsWith(resolvedRoot)) {
    throw new Error(`Path would escape /memories directory`);
  }

  return resolvedPath;
}
```

Attack vectors to test:
- `../` and `..\\` sequences
- URL-encoded traversal: `%2e%2e%2f`
- Absolute paths outside memory root
- Symlink attacks

### Memory Storage Location

Default: `./memory/memories/` relative to server working directory. Configurable via environment variables.

## Implementation Notes

### Error Handling

Follow the same error handling patterns as the text-editor tool:
- File not found errors
- Path validation errors
- Permission errors
- Invalid operation errors (e.g., text not unique in str_replace)

### MCP Tool Registration

Use Zod schemas with `zod-to-json-schema` for tool definitions. Each memory command needs:
1. Zod schema defining input parameters
2. Tool handler function
3. Registration with MCP server

### Dependencies Version Alignment

Key dependencies are pinned to match MCP SDK:
- `zod: ^3.23.8` - Schema validation
- `zod-to-json-schema: ^3.24.1` - Schema to JSON conversion
- `@modelcontextprotocol/sdk: ^1.0.4` - MCP server implementation

## Project Structure

```
src/
  index.ts              # MCP server entry point (currently placeholder)
test/
  example.test.ts       # Example test (to be replaced with real tests)
docs/
  039-Memory-tool.md           # Official Memory tool specification
  MCP-SDK-README.md            # MCP TypeScript SDK documentation
  tools-helpers-memory.ts      # Reference TypeScript implementation
```

## Next Steps for Implementation

1. **Study the reference implementation** in `docs/tools-helpers-memory.ts`
2. **Implement path validation** with comprehensive security tests
3. **Create Zod schemas** for all 6 memory commands
4. **Implement memory operations** with filesystem operations
5. **Register MCP tools** using the SDK
6. **Write security tests** for path traversal attacks
7. **Test integration** with Claude Code via `.mcp.json`

## CI/CD

GitHub Actions workflow runs on `push` and `pull_request` to `main` and `dev` branches:
- Linting with ESLint
- TypeScript compilation
- Tests with coverage on Node 18.x, 20.x, 22.x
- Coverage upload to Codecov (Node 22.x only)

---

## 🔄 SESSION HANDOFF

**⚠️ IMPORTANT**: This section must be **REWRITTEN** at the end of every session. Do NOT append - replace the entire content below with fresh handoff information.

### Current Implementation Status

**✅ CORE IMPLEMENTATION COMPLETE + TREE VIEW FEATURE** - All tests passing, spec-compliant, tree view optional feature added.

**Project State**: Complete implementation with unified tool interface + optional tree view mode for enhanced directory navigation.

### Recent Changes (This Session)
- 🌳 **Tree View Feature Added**: Optional `--tree-view` CLI flag enables hierarchical directory view
- ✅ **Tree view module**: New `src/memory/tree-view.ts` with formatting and rendering functions
- ✅ **CLI integration**: Added `--tree-view` flag parsing and propagation through system
- ✅ **Context propagation**: treeView flag passed through CLI → MCP server → operations
- ✅ **Conditional rendering**: viewDirectory() checks context.treeView flag
- ✅ **Comprehensive testing**: 24 new tree view tests added (85 total tests now)
- ✅ **Documentation updated**: README, ARCHITECTURE.md, CLAUDE.md reflect tree view feature
- ✅ **Manual testing complete**: Both simple and tree modes verified working

### What Works
- ✅ **Unified memory tool** with command-based dispatch (view, create, str_replace, insert, delete, rename)
- ✅ **Tree view mode**: Hierarchical directory view with sizes, line counts, modification times
- ✅ **Discriminated union validation**: Each command has only its relevant parameters
- ✅ Path security with 27 comprehensive tests (directory traversal protection)
- ✅ File locking with optimistic concurrency control
- ✅ stdio and streamable HTTP transports
- ✅ CLI argument parsing (--memory-root-path, --transport, --port, --tree-view, --debug, --version, --help)
- ✅ Debug logging to /tmp/memory-mcp/<instance-id>.log
- ✅ 85/85 tests passing (27 security + 34 operations + 24 tree view)
- ✅ Project compiles and lints successfully
- ✅ Comprehensive documentation (README.md, docs/ARCHITECTURE.md)

### Tool Interface
**Before (incorrect)**: 6 separate tools (`memory_view`, `memory_create`, etc.)
**Now (correct)**: Single `memory` tool with command parameter:
```typescript
memory({ command: "view", path: "/memories" })
memory({ command: "create", path: "/memories/file.txt", file_text: "..." })
memory({ command: "str_replace", path: "/memories/file.txt", old_str: "...", new_str: "..." })
// etc.
```

### What's NOT Done Yet
- ⚠️ No multi-process concurrency integration tests (unit tests only)
- ⚠️ Manual testing with MCP Inspector not done
- ⚠️ Manual testing with actual Claude Code instance not done
- ⚠️ Locking unit tests not written (locking is tested indirectly through operations tests)

### ⚠️ **CRITICAL CONCURRENCY ISSUES FOUND IN PR REVIEW** ⚠️

**PR Review Date**: 2025-10-16 (Session: 1c890f75-b8d0-425b-926b-90e71dc52c18)

**Status**: Blocking issues found. DO NOT merge to main until fixed.

#### Critical Issues (Must Fix Before Merge):

**1. Read Locks Are Actually Exclusive (Critical - Performance)**
- **Location**: `src/memory/locking.ts:86-109`
- **Issue**: Code claims "concurrent reads" but `acquireReadLock()` uses exclusive locks
- **Impact**: Multiple Claude instances viewing `/memories` will serialize unnecessarily
- **Root Cause**: proper-lockfile doesn't support shared read locks
- **Decision**: ☞ **Migrate to @esfx/async-readerwriterlock** for true RW locks

**2. Rename Destination Not Locked (Critical - Race Condition)**
- **Location**: `src/memory/operations.ts:410-475`
- **Issue**: Only locks source path, not destination - allows concurrent operations on dest
- **Impact**: Data corruption, race conditions, unpredictable behavior
- **Solution**: Lock BOTH source and destination in sorted order (prevents deadlock)
- **Implementation**: Create `withMultipleWriteLocks()` helper for atomic multi-path locking

**3. exists() Helper Swallows Errors (Correctness Bug)**
- **Location**: `src/memory/operations.ts:69-76`
- **Issue**: Permission denied reported as "file not found" - empty catch block
- **Impact**: Misleading error messages, hard to debug permission issues
- **Solution**: Only catch ENOENT, rethrow other errors with helpful messages

#### Library Decision: @esfx/async-readerwriterlock

**Chosen**: `@esfx/async-readerwriterlock` v1.0.0

**Why**:
- ✅ Purpose-built for read-writer locks (not a general mutex)
- ✅ True shared read locks - multiple concurrent readers
- ✅ Exclusive write locks
- ✅ Upgradeable read locks (read → write atomically)
- ✅ TypeScript-first design
- ✅ Actively maintained - repo updated 2025-10-16
- ✅ By Ron Buckton (Microsoft TypeScript team)
- ✅ Apache-2.0 license, 234 GitHub stars

**API Preview**:
```typescript
import { AsyncReaderWriterLock } from '@esfx/async-readerwriterlock';

const rwlock = new AsyncReaderWriterLock();

// Shared read lock (multiple concurrent readers)
const readLock = await rwlock.read();
try {
  // ... read operation
} finally {
  readLock.unlock();
}

// Exclusive write lock
const writeLock = await rwlock.write();
try {
  // ... write operation
} finally {
  writeLock.unlock();
}
```

#### Next Session Plan: Concurrency Hardening

**Priority 1: Fix Critical Locking Issues**
1. Install `@esfx/async-readerwriterlock`
2. Rewrite `src/memory/locking.ts` to use true RW locks
3. Add `withMultipleWriteLocks()` helper for atomic multi-path locking
4. Update `rename()` operation to lock both source and destination

**Priority 2: Fix Error Handling**
1. Fix `exists()` helper to only catch ENOENT
2. Review all empty catch blocks (9 instances found)
3. Ensure filesystem errors surface to users

**Priority 3: Testing & Validation**
1. Update locking tests for new RW lock behavior
2. Add multi-process concurrency tests
3. Verify all 85 tests still pass
4. Update documentation

**Files to Modify**:
- `src/memory/locking.ts` - complete rewrite for RW locks
- `src/memory/operations.ts` - fix rename() and exists()
- `test/memory-operations.test.ts` - update for new locking
- `package.json` - add @esfx/async-readerwriterlock dependency
- `docs/ARCHITECTURE.md` - document RW lock architecture

### Quick Start for Next Session

```bash
# Build and test
npm run build
npm test  # Should show 85/85 passing

# Test CLI
node dist/index.js --help
node dist/index.js --version

# Test stdio transport (MCP Inspector needed)
node dist/index.js
node dist/index.js --tree-view  # With tree view mode

# Test HTTP transport
node dist/index.js --transport http --port 3000
# Then connect with: npx @modelcontextprotocol/inspector http://localhost:3000/mcp
```

### Key Files to Know
- `src/index.ts` - CLI entry point and main setup
- `src/memory/operations.ts` - All 6 memory commands
- `src/memory/tree-view.ts` - **Tree view rendering** (optional feature)
- `src/memory/locking.ts` - File locking with optimistic concurrency
- `src/memory/path-security.ts` - Path validation (security critical!)
- `src/server/mcp-server.ts` - **Unified tool registration** with discriminated union schema
- `src/server/transports.ts` - stdio and HTTP transport initialization
- `test/path-security.test.ts` - 27 security tests
- `test/memory-operations.test.ts` - 34 operations tests
- `test/tree-view.test.ts` - **24 tree view tests**
- `docs/ARCHITECTURE.md` - **Includes unified tool interface + tree view design**

### Architecture Highlights
1. **Unified Tool Interface**: Single `memory` tool with discriminated union for type-safe dispatch
2. **Hybrid Concurrency**: File locking (proper-lockfile) + optimistic concurrency (mtime checks)
3. **Smart Locking**: Non-existent files lock parent directory, reads wait without errors
4. **Path Security**: Multi-layer validation prevents all known traversal attacks
5. **Stateless HTTP**: New transport per request prevents JSON-RPC ID collisions

### Next Steps (Priority Order)
1. **Integration Testing** - Test with MCP Inspector (stdio and HTTP) - verify unified tool works
2. **Real-World Testing** - Test with actual Claude Code instance via .mcp.json
3. **Concurrency Testing** - Spawn multiple processes, verify concurrent access works
4. **Locking Tests** (optional) - Dedicated unit tests for locking module

### Known Gotchas
- **proper-lockfile** can't lock non-existent files → solution: lock parent directory
- **mtime precision** varies by filesystem → using millisecond timestamps
- **HTTP transport** must create new transport per request → prevents ID collisions
- **Path validation** must happen BEFORE locking → prevents ENOENT errors
- **MCP SDK inputSchema**: Expects ZodRawShape, not discriminated union → workaround: all params optional in schema, strict validation in handler

### Dependencies Installed
- ✅ express, cors, proper-lockfile
- ✅ All @types packages
- ✅ All MCP SDK dependencies (zod, zod-to-json-schema)

### Test Coverage
- Path Security: 27/27 passing
- Memory Operations: 34/34 passing
- Tree View: 24/24 passing
- Total: 85/85 tests passing
- Coverage: Est. 80%+ (untested: multi-process scenarios)

---

**Last Updated**: 2025-10-16 (Session: pr-review-1c890f75)
**Status**: ⚠️ **PR BLOCKED** - Critical concurrency issues found, must fix before merge
**Next Session**: Concurrency hardening - migrate to @esfx/async-readerwriterlock
