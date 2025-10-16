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

**⚠️ CI INVESTIGATION NEEDED** - All fixes complete locally (92/92 tests pass), but CI failing. Likely ESLint config difference.

**Project State**: Complete implementation, all local tests pass, awaiting CI resolution.

### Recent Changes (This Session)

**PR Review & Critical Fixes:**
- ✅ Comprehensive PR review using specialized agents (code-reviewer, silent-failure-hunter, pr-test-analyzer, comment-analyzer)
- ✅ Fixed 3 critical error handling bugs with test-first approach
- ✅ All 92 tests passing locally
- ✅ Linter passing locally
- ⚠️ CI failing with ESLint errors (local/CI environment mismatch)

**Fixes Applied:**
1. **exists() helper** - Now distinguishes ENOENT from EACCES (permission vs not-found)
2. **viewDirectory()** - Per-file error handling, graceful continuation  
3. **buildDirectoryTree()** - Proper error logging, no silent failures
4. **error-utils.ts** - Created shared error handling utilities to fix circular dependency
5. **Fixed .gitignore** - Changed `memory/` to `/memory/` to not block `src/memory/`

### CI Issue Details

**Local Environment:** ✅ All passing
- `npm run build` - Success
- `npm run lint` - 0 errors
- `npm test` - 92/92 passing

**CI Environment:** ❌ Failing
- ESLint reporting unsafe error handling on lines that use helper functions
- Possible causes:
  1. ESLint version difference
  2. TypeScript version difference  
  3. Node version difference (testing 18.x, 20.x, 22.x)
  4. GitHub CDN caching (tried workarounds)

**Current PR:** #2 https://github.com/yannbam/memory-mcp/pull/2
- Clean branch: `fix-error-handling-v2`
- Old PR #1 closed due to potential caching issues

### Files Modified

**New Files:**
- `src/memory/error-utils.ts` - Type-safe error handling helpers
- `test/memory-operations.test.ts` - Added 7 error handling tests
- `test/tree-view.test.ts` - Added 3 permission error tests
- `TEST-COVERAGE-ANALYSIS.md` - PR review test coverage analysis

**Modified Files:**
- `src/memory/operations.ts` - Uses error-utils helpers
- `src/memory/tree-view.ts` - Uses error-utils helpers
- `.gitignore` - Fixed pattern to not block src/memory/
- `package-lock.json` - Now committed for CI
- `CLAUDE.md` - This handoff section

### Next Steps (Priority Order)

1. **INVESTIGATE CI FAILURE** - Local passes, CI fails
   - Check ESLint/TypeScript/Node version differences
   - Compare package.json versions with CI environment
   - May need to adjust ESLint rules or update dependencies
   - Check if error-utils.ts is actually being used in CI build

2. **After CI passes:**
   - Merge PR #2 to main
   - Real-world testing with Claude Code instance

### Quick Start for Next Session

```bash
# Current branch
git checkout fix-error-handling-v2

# Verify local state
npm run build  # Should pass
npm run lint   # Should pass  
npm test       # 92/92 should pass

# Check CI
gh pr checks 2

# Debug CI vs local difference
# Compare versions, check what CI actually builds
```

### Test Coverage
- Path Security: 27/27 passing
- Memory Operations: 41/41 passing (includes 7 new error handling tests)
- Tree View: 24/24 passing (includes 3 new permission error tests)  
- **Total: 92/92 tests passing** ✅ (locally)

---

**Last Updated**: 2025-10-16 (Session: ci-debugging-investigation-needed)
**Status**: ⚠️ Awaiting CI Investigation - Local Perfect, CI Failing
