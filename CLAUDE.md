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

**✅ E2E TESTING COMPLETE - PRODUCTION READY** - All functionality validated with actual Claude Code instance. PR #3 created for merge to main.

**Project State**: Fully implemented, comprehensively tested, security hardened, documented, and validated in real-world conditions.

### Recent Changes (This Session - 874e17d6-aa8b-4234-9f5b-004fddb4cbb6)

**End-to-End Testing Complete:**

Conducted comprehensive E2E testing with actual Claude Code instance via MCP integration:

1. **All 6 Memory Commands Validated** - 100% pass rate:
   - ✅ **view**: Directory listings, file contents, line ranges, unicode display
   - ✅ **create**: Files, nested directories (auto-create), unicode content, special chars
   - ✅ **str_replace**: Unique replacements, multiline, error handling (non-unique, not found)
   - ✅ **insert**: All positions (beginning, middle, end), multiline, validation
   - ✅ **delete**: Files, directories (recursive), error handling
   - ✅ **rename**: Files, directories, move operations, collision detection

2. **Path Security Validated** - All attacks blocked:
   - ✅ Directory traversal: `../../../etc/passwd` blocked
   - ✅ Absolute paths: `/etc/passwd` rejected
   - ✅ URL-encoded traversal: `%2e%2e%2f` handled
   - ✅ Security enforced across ALL operations
   - ✅ Clear, descriptive error messages

3. **Error Handling Verified**:
   - ✅ MCP Zod validation (invalid commands, missing parameters)
   - ✅ File not found errors
   - ✅ Path validation errors
   - ✅ Operation-specific errors (line ranges, non-unique text, etc.)

4. **Unicode & Special Characters** - Full support:
   - ✅ Emoji: 🚀 💻 🎨 🔥 ☞ 🐾 ∞ ≠ ≈ ∑
   - ✅ Japanese: こんにちは
   - ✅ Arabic: مرحبا
   - ✅ Special chars: !@#$%^&*()_+-={}[]|:";'<>?,./

5. **PR Created** - https://github.com/yannbam/memory-mcp/pull/3:
   - Comprehensive PR description with test results
   - All 6 commits from dev branch
   - Ready for merge to main

### What Works (100% Tested)
- ✅ **Unified memory tool** with command-based dispatch
- ✅ **Tree view mode**: Hierarchical display with sizes, line counts, timestamps
- ✅ **Path security**: Multi-layer validation, all traversal attacks blocked
- ✅ **True RW locks**: @esfx/async-readerwriterlock (38x speedup)
- ✅ **Atomic multi-path locking**: Deadlock-safe rename operations
- ✅ **MCP error handling**: Proper isError flag, clear error messages
- ✅ **stdio transport**: Validated with actual Claude Code instance
- ✅ **CLI argument parsing**: All flags working correctly
- ✅ **Debug logging**: /tmp/memory-mcp/<instance-id>.log
- ✅ **Unicode support**: Full UTF-8 including emoji, CJK, Arabic
- ✅ **85/85 unit tests passing** + comprehensive integration tests + E2E validation

### Tool Interface
Single `memory` tool with discriminated union schema:
```typescript
memory({ command: "view", path: "/memories" })
memory({ command: "create", path: "/memories/file.txt", file_text: "..." })
memory({ command: "str_replace", path: "/memories/file.txt", old_str: "...", new_str: "..." })
memory({ command: "insert", path: "/memories/file.txt", insert_line: 1, insert_text: "..." })
memory({ command: "delete", path: "/memories/file.txt" })
memory({ command: "rename", old_path: "/memories/old.txt", new_path: "/memories/new.txt" })
```

### What's NOT Tested Yet
- ⚠️ HTTP transport (stdio validated, HTTP not tested in E2E)
- ⚠️ Multi-process concurrency (single-process validated with RW locks)

### Test Coverage Summary

**Unit Tests (85/85 passing):**
- Path Security: 27/27 ✅
- Memory Operations: 34/34 ✅
- Tree View: 24/24 ✅

**Integration Tests:**
- Concurrent reads: ✅
- Write serialization: ✅
- Stress test (50 clients): ✅
- Error detection: 14/14 ✅

**E2E Tests (100% pass rate):**
- View command: 7/7 scenarios ✅
- Create command: 5/5 scenarios ✅
- str_replace command: 5/5 scenarios ✅
- Insert command: 5/5 scenarios ✅
- Delete command: 3/3 scenarios ✅
- Rename command: 5/5 scenarios ✅
- Path security: 6/6 attacks blocked ✅
- Error handling: 5/5 cases ✅

### Architecture Highlights
1. **Unified Tool Interface**: Single `memory` tool with discriminated union
2. **True RW Locks**: 38x performance improvement on concurrent reads
3. **Atomic Multi-Path Locking**: Deadlock-safe rename with sorted acquisition
4. **Multi-Layer Path Security**: Prevents all known traversal attacks
5. **Stateless HTTP**: New transport per request prevents ID collisions
6. **Comprehensive Validation**: Zod schemas + runtime checks

### Key Files to Know
- `src/index.ts` - CLI entry point and main setup
- `src/memory/operations.ts` - All 6 memory commands (185-231: view range validation)
- `src/memory/locking.ts` - LockManager with true RW locks (381 lines)
- `src/memory/tree-view.ts` - Tree view rendering
- `src/memory/path-security.ts` - Path validation (security critical!)
- `src/server/mcp-server.ts` - Unified tool registration (112-159: error handling)
- `src/server/transports.ts` - stdio and HTTP transport initialization
- `tests/integration/` - Integration test suite (4 files)
- `docs/LOCKING-REDESIGN.md` - RW lock architecture documentation

### Quick Start for Next Session

```bash
# Build and test
npm run build
npm test  # Should show 85/85 passing

# Run with Claude Code (stdio transport)
node dist/index.js              # Standard mode
node dist/index.js --tree-view  # With tree view

# Run with HTTP transport (not E2E tested yet)
node dist/index.js --transport http --port 3000
```

### Next Steps (Priority Order)
1. **Merge PR #3 to main** - All validation complete
2. **Publish to npm** - Ready for public use
3. **HTTP transport E2E testing** (optional - stdio fully validated)
4. **Multi-process stress testing** (optional - single-process validated)

### Known Working Patterns
- **Path security**: Multi-layer validation prevents all traversal attacks
- **RW locks**: Concurrent reads work, writes serialize correctly
- **Unicode**: Full UTF-8 support validated with emoji, CJK, Arabic
- **Error handling**: MCP isError flag + clear error messages
- **Atomic operations**: Multi-path locking with deadlock prevention

### Dependencies (All Validated)
- ✅ `@esfx/async-readerwriterlock: ^1.0.0` - True RW locks
- ✅ `@modelcontextprotocol/sdk: ^1.0.4` - MCP server implementation
- ✅ `zod: ^3.23.8` - Schema validation
- ✅ `express`, `cors` - HTTP transport
- ✅ All @types packages

---

**Last Updated**: 2025-10-25 (Session: 36ff98ba-97e8-4c1b-a9b2-92918fbe0395)
**Status**: 🔄 **PREPARING PUBLIC BETA RELEASE** - v0.1.0

## 🔄 Session Handoff

**Current Task**: Preparing repository for public beta release

**Plan**: View plan "public-release-beta" (39 tasks, 8 phases)
- Phase 1: Adopt Memory System (URGENT - dogfood our own tool!)
- Phase 2: Configure MCP for Project
- Phase 3: Repository Cleanup
- Phase 4: Documentation Updates
- Phase 5: Prepare for npm (@yannbam/memory-mcp - don't publish yet)
- Phase 6: Optional Professional Touches
- Phase 7: Pre-Release Verification
- Phase 8: Comprehensive Code Review

**Key Decisions Made**:
- ✅ Actually USE memory-mcp to document memory-mcp (dogfooding)
- ✅ Commit .memory/ and .mcp.json as living examples
- ✅ Add .mcp.example.json for users to copy
- ✅ Public beta (v0.1.0) without npm publish yet
- ✅ Scoped package: @yannbam/memory-mcp
- ✅ Rename reference file to tools-helpers-memory-anthropic-reference.ts

**Next Session TODO**:
1. Start Phase 1: Study ~/projects/mcp-ts-api/.memory/ structure
2. Create our own .memory/memories/ with architecture/, development/, releases/
3. Restructure this CLAUDE.md:
   - Remove README.md duplications
   - Split into: Static section + Memory @ references
   - Move handoff to .memory/memories/short-term/
   - Stop updating CLAUDE.md every session (only when static content changes)

**Philosophy Change**: CLAUDE.md becomes stable reference, memory system handles session-to-session info.

---
[the whole memory section needs to be improved!!]

## Memory 

**Use project memory PROACTIVELY throughout development!**

### Memory Philosophy

Memory is numbered and **line-based** for easy editing - each line is an independent fact.
Use memory("cmd": "add") [cont...]


Memory captures **evolving reality** discovered during development - NOT static documentation (that's in docs/).

### What Goes Where

[needs to be corrected! this is backwards]
**short-term.md** - Session context and handoff:
- Current session info, what was done, what's next
- Active tasks and immediate blockers
- Quick freeform notes
- Updated at **END of session**

**long-term.md** - Cross-session wisdom:
- Architecture insights (design discoveries, validation results)
- Performance measurements (actual vs expected)
- Runtime behavior (execution quirks, edge cases)
- MCP SDK quirks and workarounds
- Common mistakes (what NOT to try again!)
- Proven patterns (what WORKS)
- Testing insights, workflow commands
- Updated **DURING session** when discovering important things

### Critical Rules

**PRESERVE structure**:
- Never modify or delete section headers (lines starting with `##`)
- Never modify or delete italic descriptions (lines with `_text_`)
- These are template instructions that persist across sessions

**ADD sections when needed**:
- You CAN add new sections if discoveries don't fit existing ones
- New sections must have: header (`## Name`) + italic description (`_what goes here_`)

**Edit content effectively**:
- Use `str_replace` for updating specific lines
- Each line is an independent fact
- Be specific: include file paths, function names, exact errors, numbers
- Remove outdated information when reality changes

**Avoid duplication**:
- Don't duplicate README.md or docs/ content
- Memory is for **discovered reality**, not planned architecture

### When to Update

**During work**:
- Use short-term.md as scratchpad for active session

- Add discoveries to long-term.md immediately

**End of session**:
- Review short-term discoveries - what's worth keeping forever?
- Transfer important findings to long-term.md
- Update short-term.md handoff for next session
- Clean up outdated entries in both files

---

## Project Memory

DO NOT EDIT MEMORY IN THIS FILE - USE THE project_memory TOOL INSTEAD.

## <project_memory>
@.memory/memories/short-term.md
@.memory/memories/long-term.md
</project_memory>