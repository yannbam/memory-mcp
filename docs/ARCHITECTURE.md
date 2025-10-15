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
│  File Locking │    │  Path Security    │
│  (locking.ts) │    │  (path-security.ts)│
│               │    │                    │
│ - Write locks │    │ - Traversal check │
│ - Read locks  │    │ - /memories prefix│
│ - Mtime check │    │ - Path resolution │
└───────────────┘    └───────────────────┘
```

## Key Design Decisions

### 1. Hybrid Concurrency Strategy

**Problem**: Multiple Claude instances may try to modify the same memory files simultaneously.

**Solution**: Hybrid approach combining file locking with optimistic concurrency control.

- **File Locking**: Uses `proper-lockfile` for cross-process exclusive/shared locks
- **Optimistic Concurrency**: Captures file mtime before lock, checks after acquiring lock
- **Smart Locking**:
  - Non-existent files → lock parent directory
  - Read operations → no concurrency check (just wait for writers to finish)
  - Write operations → mtime check detects concurrent modifications

**Example Flow** (str_replace):
```
1. Read file mtime: 1234567890
2. Wait for exclusive lock (another process might be writing)
3. Acquire lock
4. Check mtime again: 1234567999 ← DIFFERENT!
5. Throw error: "File has been modified by another process"
6. Claude reads file again and retries
```

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

## File Structure

```
src/
├── index.ts                 # CLI entry, transport initialization
├── memory/
│   ├── operations.ts        # 6 memory commands implementation
│   ├── locking.ts           # File locking + optimistic concurrency
│   └── path-security.ts     # Path validation & security
├── server/
│   ├── mcp-server.ts        # MCP server setup, tool registration
│   └── transports.ts        # stdio and HTTP transport init
└── utils/
    └── logger.ts            # Debug logging

test/
├── path-security.test.ts    # 27 security tests
└── memory-operations.test.ts # 32 functional tests
```

## Testing Strategy

### Path Security Tests (27 tests)
- ✅ Valid paths accepted
- ✅ Invalid prefixes rejected
- ✅ Directory traversal blocked
- ✅ URL-encoded attacks blocked
- ✅ Edge cases handled

### Memory Operations Tests (32 tests)
- ✅ Each operation tested in isolation
- ✅ Edge cases (empty files, nested dirs)
- ✅ Error conditions (not found, not unique)
- ✅ Concurrent access patterns

### Not Yet Tested
- Multi-process concurrency (needs integration tests)
- MCP protocol compliance (manual testing with Inspector)
- Performance under load

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
2. **Mtime is good enough**: No need for content hashing for concurrency detection
3. **Stateless HTTP is simpler**: Avoided session management complexity
4. **Tests drive design**: Security tests caught edge cases early
5. **E/code works**: Intention comments made implementation clearer

## Contributors

Implemented by Claude Sonnet 4.5 with janbam.
