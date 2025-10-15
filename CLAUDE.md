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

**✅ CORE IMPLEMENTATION COMPLETE** - Unit tests passing, ready for integration testing.

**Project State**: All core functionality implemented and unit-tested. Not yet manually tested with actual MCP clients.

### What Works
- ✅ All 6 memory operations (view, create, str_replace, insert, delete, rename)
- ✅ Path security with 27 comprehensive tests (directory traversal protection)
- ✅ File locking with optimistic concurrency control
- ✅ stdio and streamable HTTP transports
- ✅ CLI argument parsing (--memory-root-path, --transport, --port, --debug, --version, --help)
- ✅ Debug logging to /tmp/memory-mcp/<instance-id>.log
- ✅ 59/59 tests passing (27 security + 32 operations)
- ✅ Project compiles successfully
- ✅ Comprehensive documentation (README.md, docs/ARCHITECTURE.md)

### What's NOT Done Yet
- ⚠️ No multi-process concurrency integration tests (unit tests only)
- ⚠️ Manual testing with MCP Inspector not done
- ⚠️ Manual testing with actual Claude Code instance not done
- ⚠️ Locking unit tests not written (locking is tested indirectly through operations tests)

### Quick Start for Next Session

```bash
# Build and test
npm run build
npm test  # Should show 59/59 passing

# Test CLI
node dist/index.js --help
node dist/index.js --version

# Test stdio transport (MCP Inspector needed)
node dist/index.js

# Test HTTP transport
node dist/index.js --transport http --port 3000
# Then connect with: npx @modelcontextprotocol/inspector http://localhost:3000/mcp
```

### Key Files to Know
- `src/index.ts` - CLI entry point and main setup
- `src/memory/operations.ts` - All 6 memory commands
- `src/memory/locking.ts` - File locking with optimistic concurrency
- `src/memory/path-security.ts` - Path validation (security critical!)
- `src/server/mcp-server.ts` - MCP tool registration with Zod schemas
- `src/server/transports.ts` - stdio and HTTP transport initialization
- `test/path-security.test.ts` - 27 security tests
- `test/memory-operations.test.ts` - 32 operations tests
- `docs/ARCHITECTURE.md` - Detailed design decisions

### Architecture Highlights
1. **Hybrid Concurrency**: File locking (proper-lockfile) + optimistic concurrency (mtime checks)
2. **Smart Locking**: Non-existent files lock parent directory, reads wait without errors
3. **Path Security**: Multi-layer validation prevents all known traversal attacks
4. **Stateless HTTP**: New transport per request prevents JSON-RPC ID collisions

### Next Steps (Priority Order)
1. **Integration Testing** - Test with MCP Inspector (stdio and HTTP)
2. **Real-World Testing** - Test with actual Claude Code instance via .mcp.json
3. **Concurrency Testing** - Spawn multiple processes, verify concurrent access works
4. **Locking Tests** (optional) - Dedicated unit tests for locking module

### Known Gotchas
- **proper-lockfile** can't lock non-existent files → solution: lock parent directory
- **mtime precision** varies by filesystem → using millisecond timestamps
- **HTTP transport** must create new transport per request → prevents ID collisions
- **Path validation** must happen BEFORE locking → prevents ENOENT errors

### Dependencies Installed
- ✅ express, cors, proper-lockfile
- ✅ All @types packages
- ✅ All MCP SDK dependencies (zod, zod-to-json-schema)

### Test Coverage
- Path Security: 27/27 passing
- Memory Operations: 32/32 passing
- Total: 59/59 tests passing
- Coverage: Est. 80%+ (untested: multi-process scenarios)

---

**Last Updated**: 2025-10-15 (Session: memory-mcp-implementation)
**Status**: ⚠️ Untested - Core Complete, Needs Manual Verification
