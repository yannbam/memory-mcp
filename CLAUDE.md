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

**✅ PRODUCTION READY** - All critical error handling issues fixed, 92/92 tests passing, ready for merge to main.

**Project State**: Complete MCP server implementation with robust error handling, unified tool interface, and optional tree view mode.

### Recent Changes (This Session)
- 🔧 **Critical Error Handling Fixes** (PR review findings):
  - Fixed exists() helper to distinguish ENOENT from EACCES (no more misleading "Path not found" for permission errors)
  - Fixed viewDirectory() per-file stat errors (graceful handling with logging, continues on errors)
  - Fixed buildDirectoryTree() silent failures (proper error logging, distinguishes EACCES/ENOENT/other)
- ✅ **Error Handling Tests**: Added 7 new tests for permission errors, race conditions, per-file errors
- ✅ **All tests passing**: 92/92 tests (27 security + 41 operations + 24 tree view)
- ✅ **MCP-Debug tested**: Verified error handling works correctly in real MCP context
- 📝 **Test coverage analysis**: Documented in TEST-COVERAGE-ANALYSIS.md

### What Works
- ✅ **Unified memory tool** with command-based dispatch (view, create, str_replace, insert, delete, rename)
- ✅ **Robust error handling**: Permission errors logged and propagated, race conditions handled gracefully
- ✅ **Tree view mode**: Hierarchical directory view with sizes, line counts, modification times
- ✅ **Discriminated union validation**: Each command has only its relevant parameters
- ✅ **Path security**: 27 comprehensive tests (directory traversal protection)
- ✅ **File locking**: Optimistic concurrency control
- ✅ **Multiple transports**: stdio and streamable HTTP
- ✅ **CLI arguments**: --memory-root-path, --transport, --port, --tree-view, --debug, --version, --help
- ✅ **Debug logging**: /tmp/memory-mcp/<instance-id>.log
- ✅ **92/92 tests passing**: All unit and error handling tests
- ✅ **Clean build**: TypeScript compiles without errors, ESLint clean

### Error Handling Improvements (Session focus)

**1. exists() Helper** (src/memory/operations.ts:72-85)
- ❌ Before: Returned false for ALL errors (EACCES looked like ENOENT)
- ✅ After: Only returns false for ENOENT, throws for permission/IO errors with context

**2. viewDirectory() Per-File Errors** (src/memory/operations.ts:161-186)
- ❌ Before: No error handling around fs.stat() - one bad file crashed entire listing
- ✅ After: Try-catch per file, logs errors, continues processing other files

**3. buildDirectoryTree() Silent Failures** (src/memory/tree-view.ts:142-222)
- ❌ Before: Empty catch block returned empty array for ANY error (permission → looks empty)
- ✅ After: Per-entry AND directory-level error handling with console.error() logging

**Impact**: Users now see clear error messages, permission issues don't appear as "file not found", directory listings don't silently fail, race conditions are handled gracefully.

### Tool Interface
**Single unified `memory` tool** with command discriminated union:
```typescript
memory({ command: "view", path: "/memories" })
memory({ command: "create", path: "/memories/file.txt", file_text: "..." })
memory({ command: "str_replace", path: "/memories/file.txt", old_str: "...", new_str: "..." })
// etc.
```

### Quick Start for Next Session

```bash
# Build and test
npm run build
npm test  # Should show 92/92 passing

# Test CLI
node dist/index.js --help
node dist/index.js --version

# Test stdio transport
node dist/index.js --tree-view --debug

# Test with MCP-Debug
# (MCP-Debug server configured in .mcp.json)
```

### Key Files to Know
- `src/memory/operations.ts` - All 6 memory commands + **fixed error handling**
- `src/memory/tree-view.ts` - Tree view rendering + **fixed silent failures**
- `src/index.ts` - CLI entry point and main setup
- `src/memory/locking.ts` - File locking with optimistic concurrency
- `src/memory/path-security.ts` - Path validation (security critical!)
- `src/server/mcp-server.ts` - Unified tool registration with discriminated union
- `src/server/transports.ts` - stdio and HTTP transport initialization
- `test/memory-operations.test.ts` - 41 operations tests + **new error handling tests**
- `test/tree-view.test.ts` - 24 tree view tests + **new permission error tests**
- `test/path-security.test.ts` - 27 security tests
- `TEST-COVERAGE-ANALYSIS.md` - **Detailed test coverage analysis from PR review**

### Next Steps (Priority Order)
1. **Merge to main** - All critical issues fixed, ready for production
2. **Real-world testing** - Test with actual Claude Code instance
3. **Integration Testing** (optional) - MCP Inspector testing
4. **Future enhancements** - Multi-process concurrency tests, dedicated locking tests

### Test Coverage
- Path Security: 27/27 passing
- Memory Operations: 41/41 passing (includes 7 new error handling tests)
- Tree View: 24/24 passing (includes 3 new permission error tests)
- **Total: 92/92 tests passing** ✅
- Coverage: Est. 80%+ (untested: multi-process scenarios)

### Files Modified (This Session)
- src/memory/operations.ts - Fixed exists() and viewDirectory()
- src/memory/tree-view.ts - Fixed buildDirectoryTree() catch blocks
- test/memory-operations.test.ts - Added error handling tests
- test/tree-view.test.ts - Added permission error tests
- TEST-COVERAGE-ANALYSIS.md - PR review test coverage analysis

---

**Last Updated**: 2025-10-16 (Session: critical-error-handling-fixes)
**Status**: ✅ Production Ready - All Critical Issues Fixed
