# Long-Term Memory (Persistent Knowledge)

## Architecture & Design
_How the system actually works vs how it was intended to work_
## Concurrency & Multi-Instance Behavior
_How the system handles multiple Claude instances accessing same memory files_

[🏗️🔒💡] Hybrid concurrency strategy combining true reader-writer locks (@esfx/async-readerwriterlock) with optimistic concurrency control (mtime-based)
Implementation: src/memory/locking.ts (LockManager class L43-216, withReadLock L291-308, withWriteLock L322-361)
Architecture documented in docs/ARCHITECTURE.md L60-81 and docs/LOCKING-REDESIGN.md (complete redesign spec)
[⚡🎯] Read-read concurrency: Multiple readers on same file execute simultaneously with ZERO blocking (shared read locks) - ~38x speedup tested with 50 concurrent clients
Read-write conflict: Reader completes first, writer waits for exclusive lock
Write-write race: First writer succeeds, second fails with mtime mismatch error prompting re-read and retry
[🔒💡] Lock granularity: One RW lock per unique file path (reference counted, auto-cleanup at zero refs)
Non-existent files: Lock parent directory instead (prevents creation races)
Multi-path operations (rename): Deadlock-free via sorted path order acquisition
[⚠️🤯] Optimistic concurrency control pattern: (1) Capture mtime BEFORE lock, (2) Acquire lock, (3) Check mtime AFTER lock, (4) Throw if changed
Error message: "File has been modified by another process. Please read the file again and retry your operation."
Ensures Claude detects concurrent modifications and retries with fresh data - no silent corruption
[⚠️💀] Current limitation: Locks are in-memory only (not cross-process filesystem locks)
Works perfectly for multiple Claude instances connecting to SAME memory-mcp server process
Does NOT coordinate between multiple independent memory-mcp server processes (different lock pools)
This is intentional - MCP servers designed for single-process multi-client usage
[💀🔧⚠️→✅] CRITICAL LIMITATION OF MTIME **SOLVED** (Session 76621b6d design, be292b0b implementation, Oct 30 2025)
Problem: Mtime only detected CONCURRENT modifications (during lock wait), NOT SEQUENTIAL (between separate operations)
Scenario: Claude reads file, another process modifies minutes later, Claude writes → confusing "text not found" errors
Solution: SHA-256 content checksum caching - detects ALL modifications since last access by THIS process
[🚀✅💯] Checksum-based concurrency detection IMPLEMENTED (Session be292b0b, Oct 30 2025)
Implementation: src/memory/checksums.ts (6 functions), modified locking.ts (two-layer detection), operations.ts (cache after ops)
Two-layer: (1) Cache vs disk (sequential), (2) Pre-lock vs post-lock (concurrent), both use SHA-256 checksum comparison
Cross-process detection: Each stdio server has own in-memory cache, all compare against shared disk state
Error messages: Show current file contents (full, no truncation as of Session 84de9e4e), clear explanation, actionable guidance
Performance: ~0.4ms overhead for 10KB files (tested: 162 tests pass in 3.5s, 93.4% coverage)
Testing: +45 new tests (18 checksum utils, 15 locking, 9 operations, 3 integration), all passing
Commit: 1322798 on feature/checksum-concurrency-detection branch, ready to merge to dev
Design: docs/CHECKSUM-CONCURRENCY-DESIGN.md | Plan: checksum-concurrency-implementation (88% complete)


[⚠️🔒💡] **INTENTIONAL SPEC DEVIATION**: create command fails if file exists (Session d53d3ec2, Oct 30 2025)
Anthropic spec says "Create or overwrite" but this implementation enforces create-only semantics for safety
Implementation: operations.ts L303-305 checks exists() before writeFile(), throws "File already exists at {path}"
Test coverage: test/memory-operations.test.ts L149-161 verifies error on existing file
Documented in README (deviation note) and CHANGELOG (BREAKING change)
Reason: User requested fail-fast behavior instead of silent overwrite for safety

[🏗️💡] MCP server supports dual tool exposure modes: unified tool with command parameter (default) vs separate tools per command
Conditional registration in createMemoryServer() controlled by oneToolPerCommand boolean flag
Both modes use identical underlying operations - only tool registration differs
[🧪💡] Unified tool description is intentionally minimal/commented out - experiment to test HOW Claude uses commands/parameters intuitively without detailed manual
This is NOT incomplete - it's deliberate UX testing for future session analysis
[🎯💡] Parameter combination design principle: Never mix paradigms (position-based + content-based = confusing)
Position-based: insert_line, delete_line, view_range (specific line numbers)
Content-based: old_str, new_str, file_text (text search/replacement)
Read vs Write: view operations never modify files
Mixing these creates ambiguity and should be rejected at schema validation level


## Performance & Optimization
_Measured performance characteristics, bottlenecks, successful optimizations_

[⚠️🔧] Zod .refine() creates ZodEffects wrapper which breaks discriminated unions and .shape property access - for discriminated unions, move validation logic into the handler function instead of schema refinement
[✅🎯] Forgiving parameter naming implementation: Accept both old_str/new_str AND old_string/new_string by making all optional in schema, then validate in function - allows flexible mixed usage while detecting conflicts

## Environment & Dependencies
_Runtime quirks, version sensitivities, configuration gotchas_


## Mistakes to Avoid
_Failed approaches, time sinks, what NOT to do (saves future sessions from repeating)_

[💀🔧] When adding new optional parameters to commands, MUST update BOTH locations: the discriminated union schema AND the unified tool inputSchema - missing from inputSchema causes parameters to be silently dropped, leading to incorrect behavior
[💀🔧] Empty string split gives [''] not [] - when handling empty files with line operations, must check `content === '' ? [] : content.split('\n')` to avoid off-by-one errors in append logic


## Proven Solutions
_Patterns that work, reliable approaches, validated fixes_

[✅🔧] Use .shape property to extract raw Zod schema for MCP SDK inputSchema (SDK expects ZodRawShape not ZodObject)
[✅] Type assertions needed when constructing command objects from parsed params: `{ command: 'view', ...parsed } as operations.ViewCommand`
Individual tool schemas omit command field since tool name implies command
[✅💡] Parameter combination analysis methodology: Create full combinatorial matrix (every command × every parameter), categorize each as useful/questionable/confusing/nonsensical, identify patterns, design behaviors, create self-contained implementation spec
Resulted in 4 approved features: create empty file, insert append, delete matching text, document str_replace deletion
[✅🔒] Unique text requirement for safety: Both str_replace AND delete with old_str should require text appears exactly once - prevents accidental mass deletions/modifications
Fail fast with clear error message showing occurrence count when text appears multiple times
[✅🔧] Regex special character escaping: When searching for literal text (not patterns), use escapeRegExp helper: `text.replace(/[.*+?^${}()|[\]\\]/g, '\\## Proven Solutions
_Patterns that work, reliable approaches, validated fixes_

[✅🔧] Use .shape property to extract raw Zod schema for MCP SDK inputSchema (SDK expects ZodRawShape not ZodObject)
[✅] Type assertions needed when constructing command objects from parsed params: `{ command: 'view', ...parsed } as operations.ViewCommand`
Individual tool schemas omit command field since tool name implies command
[✅💡] Parameter combination analysis methodology: Create full combinatorial matrix (every command × every parameter), categorize each as useful/questionable/confusing/nonsensical, identify patterns, design behaviors, create self-contained implementation spec
Resulted in 4 approved features: create empty file, insert append, delete matching text, document str_replace deletion')`
Critical for handling text containing $, ., *, +, ?, etc.
[✅🎯] Optional parameter defaults in Zod: `.default('')` for create file_text, `.optional()` for insert insert_line
Handler uses nullish coalescing: `const content = command.file_text ?? ''`

[🎨✅💡] Shared formatting module pattern (Session 84de9e4e, Oct 30 2025)
Problem: Both operations.ts (view command) and locking.ts (error messages) needed same line-numbering logic
Solution: Created src/memory/formatting.ts with formatFileContent() function
Avoids circular dependency: operations.ts imports locking.ts, so locking.ts can't import operations.ts
Both modules import from neutral third module (formatting.ts)
Benefits: DRY principle, consistent UX, single source of truth for formatting changes
Implementation: formatFileContent(content, viewRange?) handles line numbering with 4-space padding


## Testing & Debugging
_Test strategies that work, debugging approaches, tools that help_

[🔧✨] MCP-Debug tool excellent for live testing MCP servers: connect, initialize, list tools, call tools
Can test different CLI flags by reconnecting with different args array
[⚠️🧪] MCP-Debug is useful for quick iteration BUT always verify with actual Claude Code MCP client before marking complete
Different MCP client implementations may handle schemas/parameters differently
[✅🧪] Parameter combination testing strategy: Test BOTH tool modes (unified + one-tool-per-command), test with and without optional params, test error cases (multiple occurrences, parameter conflicts)
[✅🎨] Empty content UX messaging implemented (Session e0d8aaf1, Oct 28 2025):
Empty file view returns "Memory file is empty.", empty directory returns "Directory is empty." (both tree and simple modes), empty file creation returns "Created empty memory file."
Implementation: operations.ts (viewFile L194-197, viewDirectory L183-186, create L286-288) + tree-view.ts (renderDirectoryTree L273-276)
Tested: 117 unit tests + MCP-Debug integration (all pass)
[✅🧪] Multi-process concurrent operation testing IMPLEMENTED (Session be292b0b, Oct 30 2025)
Created test/integration/concurrent-checksum.test.ts (3 tests) using child_process to spawn separate stdio MCP servers
Tests: (1) Sequential modification detection across processes, (2) Operations succeed when content matches, (3) File creation
Validates real-world scenario: Server A reads (caches checksum), Server B modifies, Server A detects via checksum mismatch
All integration tests passing - proves cross-process detection works via shared disk state


## Deferred Work
_Complex tasks or investigations postponed for future sessions_


## Project-Specific Knowledge
_Unique aspects of this particular codebase/project_

[📖💡] Parameter combinations quick reference added to README.md (table format) and src/memory/operations.ts (code comments)
Shows required vs optional parameters for all 6 commands with usage notes
Highlights flexible usage: create empty file, append to end, delete text with str_replace, etc.
Key location: README line 140-151, operations.ts line 14-43

[📊✅] Parameter combinations feature COMPLETE (Sessions f7051c6f + ff6a2f81, Oct 28 2025):
4 features fully implemented AND tested: create empty file, insert append, delete unique text, str_replace deletion
Comprehensive testing: 30+ test cases, both tool modes, edge cases, error conditions - ALL PASS
Safety validated: Both delete and str_replace require unique text (fail fast if multiple occurrences)
114 tests passing, fully documented in README/CHANGELOG/TEST-FINDINGS.md
Commits: bbd1a83 (docs) + 4950c2e (implementation)
Status: Ready for production 🚀
[🏗️💡] Dual schema locations for parameter changes: When adding optional params, update BOTH MemoryCommandSchema (21-64) AND individual command schemas (76-119) in src/server/mcp-server.ts
Also update TypeScript interfaces in src/memory/operations.ts
[🔧💡] Empty file handling pattern: Check `content === ''` before splitting to avoid [''] array
Append logic: `insertLine = lines.length + 1` works for both empty and non-empty files when using empty array for empty content


---
Note: These are starter sections - add new sections as your understanding evolves!

[🧪💡] Test cache contamination prevention: ALWAYS call clearAllCachedChecksums() in beforeEach hook
Checksum cache is a global singleton - persists across tests in same process
Without clearing: Test A caches checksum, Test B uses same path with different content → false "file modified" error
Critical for all test files that use checksum-based operations

[🔌💡] MCP tool error response pattern: Errors returned in result.content with isError: true (NOT in error field)
Example: { result: { content: [{ type: 'text', text: 'Error message...' }], isError: true } }
JSON-RPC error field reserved for protocol/infrastructure errors, not tool operation errors
Integration tests must check result.isError, not response.error

[🔧✅] EISDIR handling in withWriteLock: Directory paths (EISDIR) treated same as ENOENT (file not found)
Both skip concurrency checking (can't checksum directories), let operation validate and throw appropriate error
Prevents "EISDIR: illegal operation on a directory, read" from being thrown before operation-level validation
Operations throw better errors: "Path is not a file" or "Cannot delete line from directory"