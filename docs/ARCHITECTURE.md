# Architecture Documentation

## Overview

This MCP server implements Claude's native memory tool specification as a standalone server that can be used with any MCP client (Claude Code, MCP Inspector, etc.). The implementation prioritizes **simplicity, security, and concurrent access safety**.

## Design Principles

1. **Simple & Minimal**: No extra features beyond the 6 core memory commands
2. **Security First**: Comprehensive path validation prevents directory traversal attacks
3. **Concurrent Safe**: File locking with optimistic concurrency control for multi-instance safety
4. **Type Safe**: Full TypeScript with Zod schemas for runtime validation
5. **Observable**: Optional debug logging for troubleshooting

## Architecture Layers

```
┌─────────────────────────────────────────┐
│         CLI Entry Point (index.ts)       │
│   - Argument parsing                     │
│   - Memory root initialization           │
│   - Transport selection                  │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌─────▼──────┐
│   stdio     │  │    HTTP     │
│  Transport  │  │  Transport  │
└──────┬──────┘  └─────┬───────┘
       │                │
       └───────┬────────┘
               │
┌──────────────▼──────────────────────────┐
│      MCP Server (mcp-server.ts)         │
│   - Tool registration                   │
│   - Zod schema validation               │
│   - Request routing                     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   Memory Operations (operations.ts)     │
│   - view, create, str_replace           │
│   - insert, delete, rename              │
│   - Error handling                      │
└────┬─────────────────────────┬──────────┘
     │                         │
┌────▼──────────┐    ┌────────▼──────────┐
│  Locking &    │    │  Path Security    │
│  Concurrency  │    │  & Utilities      │
│               │    │                    │
│ locking.ts:   │    │ path-security.ts: │
│ - RW locks    │    │ - Traversal check │
│ - Multi-path  │    │ - /memories prefix│
│               │    │                    │
│ checksums.ts: │    │ formatting.ts:    │
│ - SHA-256     │    │ - Line numbering  │
│ - Cache       │    │                    │
│               │    │ tree-view.ts:     │
│               │    │ - Tree rendering  │
│               │    │ - Metadata        │
└───────────────┘    └───────────────────┘
```

## Key Design Decisions

### 1. Checksum-Based Concurrency Detection

**Problem**: Multiple Claude instances may try to modify the same memory files simultaneously.

**Solution**: Content-based detection using SHA-256 checksums combined with reader-writer locks.

**Architecture Components**:
- **Reader-Writer Locks**: True RW locks via `@esfx/async-readerwriterlock` for concurrent reads
- **Checksum Cache**: In-memory SHA-256 cache per stdio server process
- **Two-Layer Detection**:
  1. **Sequential Detection**: Cache comparison before lock (detects changes between separate operations)
  2. **Concurrent Detection**: Checksum recheck after lock (detects changes during lock wait)

**Why Content Checksums Over Mtime**:
- Mtime only detects concurrent modifications (during lock wait, ~milliseconds to seconds)
- Checksums detect **all** modifications since last access (minutes, hours, or days apart)
- Solves: Claude reads file → another process modifies → Claude writes based on stale data

**Cross-Process Detection Mechanism**:

Each stdio MCP server has its own memory space with separate checksum cache, but all use the **shared filesystem** as source of truth:

```
┌─────────────────────┐         ┌─────────────────────┐
│ Server Process A    │         │ Server Process B    │
│ (stdio transport)   │         │ (stdio transport)   │
├─────────────────────┤         ├─────────────────────┤
│ Checksum Cache:     │         │ Checksum Cache:     │
│ notes.txt → abc123  │         │ notes.txt → xyz789  │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           └───────────┬───────────────────┘
                       ▼
              ┌────────────────┐
              │   Filesystem   │
              │  (Disk State)  │
              │ notes.txt:     │
              │ "current data" │
              └────────────────┘
```

Detection works because:
1. Process A caches checksum of "old data"
2. Process B modifies file on disk → new content
3. Process A reads current disk state before next write
4. Computes checksum → different from cached → modification detected

**Example Flow** (str_replace):
```
1. Read file content and compute SHA-256: abc123def...
2. Compare with cached checksum → MISMATCH! (sequential detection)
   OR proceed if match/no cache
3. Acquire exclusive write lock (wait if needed)
4. Re-read file and compute checksum: abc123def...
5. Compare with pre-lock checksum → DIFFERENT! (concurrent detection)
6. Throw error with content preview:
   "File has been modified by another process.

   Current contents of /memories/notes.txt:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   [shows complete current file content with line numbers, no truncation]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Please review the current contents and retry if appropriate."
7. Claude reads current content and retries with correct understanding
```

**Smart Locking**:
- Non-existent files → lock parent directory
- Read operations → no concurrency check, just acquire shared read lock
- Write operations → two-layer checksum detection
- Directory operations → skip checksumming (not applicable)

**Performance**:
- SHA-256 hashing: ~500 MB/s throughput
- 10 KB file: ~0.02ms hashing time
- Total overhead: ~0.4ms per write operation (one extra file read + two hash computations)
- Memory: ~270 bytes per cached file (negligible for typical usage)
  - Path key: ~100 bytes average
  - Checksum value: 64 chars × 2 bytes (UTF-16) = ~128 bytes
  - Map overhead: ~40-80 bytes (V8 implementation detail)

**Cache Management**:
- Cache checksum after: full file reads (not partial), all write operations
- Clear checksum on: file/directory deletion, file rename (old path)
- Cache lifecycle: In-memory only, clears on server restart

### 2. Path Security

**Problem**: Malicious paths could escape the `/memories` directory and access system files.

**Solution**: Multi-layer validation:

1. **Prefix Check**: Path must start with `/memories`
2. **Path Resolution**: Convert to canonical absolute path
3. **Boundary Check**: Resolved path must remain within memory root
4. **Attack Vectors Blocked**:
   - `../` traversal
   - `..\\` Windows-style
   - `%2e%2e%2f` URL-encoded
   - Absolute path escapes
   - Symlink attacks (via path.resolve)

All 27 security tests pass, covering known attack patterns.

### 3. Stateless HTTP Transport

**Problem**: Session management adds complexity and state.

**Solution**: Create new transport per HTTP request.

**Why This Works**:
- Different clients may use same JSON-RPC request IDs
- Separate transport per request prevents ID collisions
- Server state (registered tools) is shared
- Transport state (request/response mapping) is isolated
- Simpler than session tracking

**Trade-off**: Slightly less efficient (new transport per request), but much simpler and more robust.

### 4. Debug Logging

**Problem**: Debugging concurrent access and locking issues is hard.

**Solution**: Optional structured JSON logging.

- Only active with `--debug` flag (zero overhead otherwise)
- Logs to `/tmp/memory-mcp/<instance-id>.log`
- Each server instance gets unique log file
- JSON format for easy parsing/analysis
- Includes operation, timing, success/failure

**Example Log Entry**:
```json
{
  "timestamp": "2025-10-15T14:30:22.123Z",
  "level": "debug",
  "operation": "str_replace",
  "path": "/memories/notes.txt",
  "duration_ms": 45,
  "lock_wait_ms": 12,
  "mtime_changed": false,
  "success": true
}
```

### 5. Memory Root Structure

**Virtual vs Filesystem Paths**:
- Virtual: `/memories/notes.txt` (MCP tool interface)
- Filesystem: `<root>/memories/notes.txt` (actual storage)

**Why the extra "memories" subdirectory?**
- Allows future expansion (e.g., `/cache`, `/temp`)
- Matches Claude's native memory tool semantics
- Clear separation of concerns

**Default Root**: `./.memory` (hidden directory, relative to CWD)

### 6. Unified Tool Interface

**Problem**: How to expose 6 memory commands through the MCP protocol?

**Initial Approach** (incorrect): Register 6 separate tools (`memory_view`, `memory_create`, etc.)

**Correct Approach**: Single unified `memory` tool matching [Anthropic's spec](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool).

**Implementation**:
- Use Zod discriminated union for type-safe command dispatch
- The `command` field determines which operation runs
- Each command variant has only its relevant parameters

**Example Schema**:
```typescript
const MemoryCommandSchema = z.discriminatedUnion('command', [
  z.object({
    command: z.literal('view'),
    path: z.string(),
    view_range: z.tuple([z.number(), z.number()]).optional(),
  }),
  z.object({
    command: z.literal('create'),
    path: z.string(),
    file_text: z.string(),
  }),
  // ... other commands
]);
```

**Dispatch Pattern**:
```typescript
switch (command.command) {
  case 'view':
    return await operations.view(command, context);
  case 'create':
    return await operations.create(command, context);
  // ...
}
```

**Benefits**:
- ✅ Matches official specification exactly
- ✅ Type-safe command dispatch (TypeScript knows which fields exist)
- ✅ Clear validation errors (wrong parameters for a command are rejected)
- ✅ Single tool registration simplifies MCP client integration

**Trade-off**: MCP SDK's `inputSchema` expects `ZodRawShape` (object), not discriminated union. Solution: Pass all parameters as optional in `inputSchema`, but validate strictly with discriminated union in handler.

### 7. Tree View Feature

**Problem**: Directory listing with `memory:view "/memories"` provides minimal navigation context - just a flat list of items.

**Use Case**: Claude instances ALWAYS call `memory:view "/memories"` at the start of EVERY session (per official spec). This is the first thing Claude sees, and it's shared across all Claude instances (Code, claude.ai, etc.).

**Solution**: Optional tree view mode enabled via `--tree-view` CLI flag.

**Tree View Shows**:
- Hierarchical directory structure with unlimited depth
- File sizes in human-readable format (B, KB, MB, GB, TB)
- Line counts for all files (all assumed to be text since Claude writes them)
- Modification times in `[YYYY/MM/DD - HH:MM:SS]` format with timezone
- Directories marked with `/` suffix

**Design Rationale**:
1. **Optional**: Default simple mode preserves backward compatibility and minimal token usage
2. **Metadata-rich**: Helps Claude decide what to read without reading everything
3. **Recency signals**: Modification times show which files are active/recent
4. **No arbitrary limits**: No depth limits or file count truncation (memory directories expected to be reasonable)
5. **Clean implementation**: Separate module (`tree-view.ts`) with no external dependencies

**Example Output**:
```
Showing contents of: /memories
Modification dates shown in [YYYY/MM/DD - HH:MM:SS] format (UTC timezone)

├── notes.txt	(2.3KB / 45 lines)	[2025/10/15 - 14:23:17]
└── projects/		[2025/10/15 - 15:01:42]
    ├── backend/		[2025/10/14 - 09:15:33]
    │   └── api.md	(5.1KB / 128 lines)	[2025/10/14 - 09:15:33]
    └── frontend/		[2025/10/15 - 15:01:42]
        └── ui.md	(1.8KB / 42 lines)	[2025/10/15 - 15:01:42]
```

**Implementation**:
- Recursive directory traversal (no depth limit)
- File metadata via `fs.stat()` (sizes, modification times)
- Line counting by reading files and counting newlines
- Hidden files (starting with `.`) are automatically skipped
- Directories sorted before files, both sorted alphabetically

**Performance**: O(n) where n = total number of files/directories (must read all files to count lines). Acceptable for session start since this is a one-time operation that provides valuable context for the entire session.

**Not Included** (stripped from reference implementation):
- Symlink handling (won't exist in /memories)
- Executable markers (memory files are data, not programs)
- Clutter filtering (memory storage should be clean)
- Truncation limits (reasonable sizes expected)
- Text file detection (all files assumed to be text)

## File Structure

```
src/
├── index.ts                 # CLI entry, transport initialization
├── memory/
│   ├── operations.ts        # 6 memory commands implementation
│   ├── checksums.ts         # SHA-256 content checksums for concurrency detection
│   ├── formatting.ts        # Shared line numbering utility
│   ├── locking.ts           # Reader-writer locks + checksum-based concurrency
│   └── tree-view.ts         # Tree view rendering (optional feature)
├── server/
│   ├── mcp-server.ts        # MCP server setup, tool registration
│   └── transports.ts        # stdio and HTTP transport init
└── utils/
    └── logger.ts            # Debug logging

test/
├── checksum-utilities.test.ts          # 18 checksum utility tests
├── locking.test.ts                     # 15 locking + concurrency tests
├── memory-operations.test.ts           # 86 operation tests (includes checksum integration)
├── path-security.test.ts               # 27 path validation tests
├── tree-view.test.ts                   # 17 tree view tests
└── integration/
    └── concurrent-checksum.test.ts     # 3 multi-process integration tests

Total: 166 tests across 6 test files
```

## Testing Strategy

### Test Coverage (166 Total Tests)

**Checksum Utilities Tests (18 tests)**
- ✅ SHA-256 checksum computation
- ✅ Cache operations (get, set, clear)
- ✅ Cache statistics and memory estimates
- ✅ Path normalization

**Locking Tests (15 tests)**
- ✅ Reader-writer lock acquisition and release
- ✅ Concurrent read operations (no blocking)
- ✅ Exclusive write operations
- ✅ Multi-path atomic locking (deadlock prevention)
- ✅ Lock cleanup and reference counting
- ✅ Concurrent lock contention scenarios

**Path Security Tests (27 tests)**
- ✅ Valid paths accepted
- ✅ Invalid prefixes rejected
- ✅ Directory traversal blocked
- ✅ URL-encoded attacks blocked
- ✅ Edge cases handled

**Memory Operations Tests (86 tests)**
- ✅ Each operation tested in isolation
- ✅ Edge cases (empty files, nested dirs)
- ✅ Error conditions (not found, not unique)
- ✅ Concurrent access patterns
- ✅ Checksum integration (sequential & concurrent detection)
- ✅ Cache management after operations

**Tree View Tests (17 tests)**
- ✅ File size formatting (B, KB, MB, GB, TB)
- ✅ Modification date formatting ([YYYY/MM/DD - HH:MM:SS])
- ✅ Line counting (Unix/Windows line endings, edge cases)
- ✅ Tree structure rendering (hierarchy, indentation)
- ✅ Deep nesting support (no limits)
- ✅ Hidden file skipping (files starting with `.`)
- ✅ Alphabetical sorting (directories first, then files)

**Integration Tests (3 tests)**
- ✅ Multi-process concurrent modifications (cross-process checksum detection)
- ✅ Sequential modification detection across server instances
- ✅ File creation race conditions

### Manual Testing
- ✅ MCP protocol compliance (validated with Claude Code and MCP Inspector)
- ✅ Real-world usage (30+ scenarios across 9 categories documented)
- 🔄 Performance under sustained load (not yet tested)

## Performance Characteristics

**Operation Costs**:
- `view`: O(1) for directories, O(n) for files
- `create`: O(1) + mkdir cost
- `str_replace`: O(n) where n = file size
- `insert`: O(n) where n = file size
- `delete`: O(1) for files, O(n) for directories
- `rename`: O(1) (atomic filesystem operation)

**Locking Overhead**:
- Uncontended: ~1-2ms per operation
- Contended: Waits indefinitely (configurable)
- Stale timeout: 30s for writes, 10s for reads

## Deployment Considerations

### For stdio Transport
```bash
# In Claude Code .mcp.json:
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/memory-mcp/dist/index.js"]
    }
  }
}
```

### For HTTP Transport
```bash
# Start server
memory-mcp --transport http --port 3000

# Connect from Claude Code
claude mcp add --transport http memory http://localhost:3000/mcp
```

### Production Considerations
- **Memory Root**: Use absolute path for predictable storage
- **Debug Logging**: Disable in production (--debug omitted)
- **CORS**: Currently allows all origins (adjust for production)
- **File System**: Ensure sufficient disk space and permissions
- **Concurrency**: Tested for correctness, not for high throughput

## Future Enhancements (Out of Scope)

The following were considered but explicitly excluded for simplicity:

- ❌ Database backend (filesystem is simpler)
- ❌ Compression (adds complexity)
- ❌ Encryption (client-side concern)
- ❌ Metrics/monitoring (debug logging sufficient)
- ❌ Health checks (MCP protocol handles this)
- ❌ Multi-user support (single user assumed)
- ❌ Quota management (OS-level concern)

## Dependencies Rationale

- `@modelcontextprotocol/sdk`: Official MCP SDK
- `express`: Battle-tested HTTP server
- `cors`: Standard CORS middleware
- `proper-lockfile`: Cross-platform file locking
- `zod`: Runtime type validation

All dependencies are well-maintained, widely used, and have minimal transitive dependencies.

## Error Handling Philosophy

Errors are **explicit and actionable**:
- ✅ "Path not found: /memories/missing.txt"
- ✅ "File has been modified by another process. Please read the file again and retry your operation."
- ✅ "Text appears 3 times in /memories/file.txt. Must be unique."

Not:
- ❌ "Operation failed"
- ❌ "Error"
- ❌ Generic exceptions

Claude can parse these messages and take appropriate action (retry, read file, adjust strategy).

## Lessons Learned

1. **Locking non-existent files is tricky**: Had to lock parent directory instead
2. **Content checksums over mtime**: Mtime only detects concurrent modifications (milliseconds). Checksums detect all modifications including sequential ones (minutes/hours apart). Worth the ~0.4ms overhead for robust detection.
3. **Stateless HTTP is simpler**: Avoided session management complexity
4. **Tests drive design**: Security tests caught edge cases early
5. **E/code works**: Intention comments made implementation clearer
6. **Read the spec first**: Initial implementation had 6 separate tools instead of 1 unified tool. Refactored to match official spec using discriminated unions.
7. **Cross-process detection via shared state**: In-memory caches in separate processes can still coordinate by all verifying against shared filesystem state

## Contributors

Implemented by Claude Sonnet 4.5 with janbam.
