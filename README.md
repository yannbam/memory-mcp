# Memory MCP Server

[![CI](https://github.com/yannbam/memory-mcp/workflows/CI/badge.svg)](https://github.com/yannbam/memory-mcp/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-117%20passing-success)](./test)

MCP server implementation of Claude's native memory tool for persistent storage across conversations.

## Overview

This project implements Claude's [memory tool](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool) as a Model Context Protocol (MCP) server. It enables any MCP client (Claude Code, MCP Inspector, etc.) to have persistent memory across sessions through secure, concurrent-safe filesystem storage.

## Features

- ✅ **All 6 Memory Commands**: view, create, str_replace, insert, delete, rename
- ✅ **Tree View Mode**: Optional hierarchical directory view with metadata (sizes, lines, timestamps)
- ✅ **High-Performance Concurrency**: True reader-writer locks for parallel reads (38x speedup)
- ✅ **Path Security**: Comprehensive directory traversal protection (27 security tests)
- ✅ **Dual Transport**: stdio (default) and streamable HTTP
- ✅ **Type-Safe**: Full TypeScript with Zod runtime validation
- ✅ **Debug Logging**: Optional structured JSON logging to `/tmp/memory-mcp/`
- ✅ **Production Ready**: Fully tested (117 unit + integration + E2E tests)

## Quick Start

```bash
# Install dependencies
npm install

# Build
npm run build

# Run with stdio transport (default)
node dist/index.js

# Run with HTTP transport
node dist/index.js --transport http --port 3000

# Run with custom memory root and debug logging
node dist/index.js --memory-root-path ~/my-memories --debug
```

## Installation

### As MCP Server for Claude Code

Add to your Claude Code `.mcp.json` configuration. Two common setups:

**1. Local Project Memory** (for a specific project):
```json
{
  "mcpServers": {
    "project_memory": {
      "command": "node",
      "args": ["./dist/index.js", "--tree-view"],
      "env": {}
    }
  }
}
```
Uses relative path, stores memory in `./.memory/memories/` within the project. Great for project-specific documentation.

**2. Global System Memory** (shared across all projects):
```json
{
  "mcpServers": {
    "system_memory": {
      "command": "node",
      "args": [
        "/absolute/path/to/memory-mcp/dist/index.js",
        "--memory-root-path",
        "/home/user/.memories",
        "--tree-view"
      ],
      "env": {}
    }
  }
}
```
Uses absolute path with custom memory root. Shared memory accessible from any Claude Code session.

**Note**: This repository includes a working `.mcp.json` and `.memory/` directory as examples of dogfooding the memory system.

### As Standalone HTTP Server

```bash
# Start server on port 3000
node dist/index.js --transport http --port 3000

# Connect from Claude Code
claude mcp add --transport http memory http://localhost:3000/mcp

# Or use MCP Inspector
npx @modelcontextprotocol/inspector http://localhost:3000/mcp
```

## CLI Options

```
memory-mcp [options]

Options:
  --memory-root-path PATH, -m PATH   Memory storage root (default: ./.memory)
  --transport TYPE, -t TYPE          Transport: stdio | http (default: stdio)
  --port PORT, -p PORT               HTTP port (default: 3000)
  --tree-view                        Enable tree view for directory listings (default: false)
  --debug, -d                        Enable debug logging
  --version, -v                      Show version
  --help, -h                         Show help
```

### Examples

```bash
# stdio transport (default)
memory-mcp

# Custom memory location
memory-mcp -m ~/my-memories

# HTTP server on port 8080
memory-mcp -t http -p 8080

# With debug logging
memory-mcp --debug

# With tree view for directory listings
memory-mcp --tree-view

# Full configuration
memory-mcp -m /var/memories -t http -p 3000 --tree-view -d
```

## Memory Tool

The server exposes a single unified **`memory`** tool with a `command` parameter that determines the operation.

This matches the [official Anthropic Memory tool specification](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool).

### view
Show directory contents or file contents with optional line ranges.

**Directory View Modes:**
- **Simple mode** (default): Flat list of files and directories
- **Tree view mode** (with `--tree-view` flag): Hierarchical structure with metadata

```typescript
// View directory (simple mode - default)
await memory({
  command: "view",
  path: "/memories"
})
// → "Directory: /memories\n- notes.txt\n- ideas/"

// View empty directory
await memory({
  command: "view",
  path: "/memories/empty"
})
// → "Directory is empty."

// View directory (tree view mode - with --tree-view flag)
// Shows hierarchical structure, file sizes, line counts, and modification times
// → "Showing contents of: /memories
// → Modification dates shown in [YYYY/MM/DD - HH:MM:SS] format (UTC timezone)
// →
// → ├── notes.txt	(2.3KB / 45 lines)	[2025/10/15 - 14:23:17]
// → └── projects/		[2025/10/15 - 15:01:42]
// →     ├── backend/		[2025/10/14 - 09:15:33]
// →     │   └── api.md	(5.1KB / 128 lines)	[2025/10/14 - 09:15:33]
// →     └── frontend/		[2025/10/15 - 15:01:42]
// →         └── ui.md	(1.8KB / 42 lines)	[2025/10/15 - 15:01:42]"

// View file
await memory({
  command: "view",
  path: "/memories/notes.txt"
})
// → "   1: First note\n   2: Second note"

// View specific lines
await memory({
  command: "view",
  path: "/memories/notes.txt",
  view_range: [2, 5]
})
// → "   2: Second note\n   3: Third note..."

// View empty file
await memory({
  command: "view",
  path: "/memories/empty.txt"
})
// → "Memory file is empty."
```

### create
Create or overwrite files (creates parent directories as needed).

```typescript
// Create file with content
await memory({
  command: "create",
  path: "/memories/todo.txt",
  file_text: "- Task 1\n- Task 2"
})
// → "File created successfully at /memories/todo.txt"

// Create empty file (omit file_text)
await memory({
  command: "create",
  path: "/memories/empty.txt"
})
// → "Created empty memory file."
```

### str_replace
Replace unique text in a file (text must appear exactly once).

```typescript
// Replace text
await memory({
  command: "str_replace",
  path: "/memories/notes.txt",
  old_str: "old value",
  new_str: "new value"
})
// → "File /memories/notes.txt has been edited"

// Delete text by omitting new_str (defaults to empty string)
await memory({
  command: "str_replace",
  path: "/memories/notes.txt",
  old_str: "debug code"
  // new_str omitted - deletes the text
})
// → "File /memories/notes.txt has been edited"
```

**Note**: Both `old_str`/`new_str` and `old_string`/`new_string` parameter names are accepted for flexibility.

### insert
Insert text at a specific line number, or append to end of file.

```typescript
// Insert at specific line
await memory({
  command: "insert",
  path: "/memories/todo.txt",
  insert_line: 2,
  insert_text: "- Urgent task"
})
// → "Text inserted at line 2 in /memories/todo.txt"

// Append to end (omit insert_line)
await memory({
  command: "insert",
  path: "/memories/todo.txt",
  insert_text: "- Last task"
})
// → "Text appended to end of /memories/todo.txt"
```

### delete
Delete files, directories, specific lines, or unique text occurrences.

```typescript
// Delete file
await memory({
  command: "delete",
  path: "/memories/old-notes.txt"
})
// → "File deleted: /memories/old-notes.txt"

// Delete directory
await memory({
  command: "delete",
  path: "/memories/archive"
})
// → "Directory deleted: /memories/archive"

// Delete specific line (1-based)
await memory({
  command: "delete",
  path: "/memories/notes.txt",
  delete_line: 5
})
// → "Line 5 deleted from /memories/notes.txt"

// Delete unique text occurrence
await memory({
  command: "delete",
  path: "/memories/notes.txt",
  old_str: "debug code"
})
// → "Deleted 1 occurrence(s) of "debug code" from /memories/notes.txt"
```

**Note**: `old_str` and `old_string` are interchangeable. Text must appear exactly once (unique occurrence).

### rename
Rename or move files/directories (creates parent directories as needed).

```typescript
await memory({
  command: "rename",
  old_path: "/memories/draft.txt",
  new_path: "/memories/final.txt"
})
// → "Renamed /memories/draft.txt to /memories/final.txt"
```

## Concurrent Access

The server supports **multiple Claude instances** accessing the same memory files simultaneously through a high-performance reader-writer lock system:

- **True RW Locks**: Powered by `@esfx/async-readerwriterlock`
- **Concurrent Reads**: Multiple readers can access the same file simultaneously (no blocking)
- **Exclusive Writes**: Writers get exclusive access, blocking both readers and other writers
- **Atomic Multi-Path Locking**: Deadlock-safe locking for operations like rename (source + destination)
- **Optimistic Concurrency**: Additional mtime-based change detection for extra safety

**Performance**: ~38x speedup for read-heavy workloads (tested with 50 concurrent clients)

**Example Scenario**:
1. Claude A reads `/memories/notes.txt` (acquires shared read lock)
2. Claude B reads the same file (also acquires shared read lock - no blocking!)
3. Claude C tries to edit the file (waits for exclusive write lock)
4. Claudes A & B finish reading (release read locks)
5. Claude C acquires write lock and modifies the file
6. If file was modified during lock wait: Error with prompt to re-read

**Key Benefits**:
- Read operations never block each other (true parallelism)
- Write operations serialize correctly (data integrity)
- Deadlock prevention through sorted lock acquisition
- Per-path lock granularity (different files don't interfere)

## Security

All paths are validated to prevent directory traversal attacks:

- ✅ Must start with `/memories`
- ✅ Blocked: `../`, `..\\`, `%2e%2e%2f`, absolute paths
- ✅ 27 comprehensive security tests

## Memory Storage

**Default**: `./.memory/memories/` (relative to server working directory)

**Custom**: Use `--memory-root-path` flag

**Structure**:
```
<memory-root>/
└── memories/          # All memory files go here
    ├── notes.txt
    ├── ideas/
    │   └── project.md
    └── archive/
        └── old.txt
```

Virtual paths (MCP interface): `/memories/notes.txt`
Filesystem paths: `<memory-root>/memories/notes.txt`

## Development

### Prerequisites

- Node.js >= 18.0.0
- npm

### Setup

```bash
git clone https://github.com/yannbam/memory-mcp.git
cd memory-mcp
npm install
npm run build
```

### Testing

```bash
npm test                 # Run all tests (117 passing)
npm run test:coverage    # Run with coverage report (80%+ target)
npm run test:watch       # Watch mode
```

**Test Coverage**:
- 27 path security tests (directory traversal attacks)
- 34 memory operations tests (all 6 commands + edge cases)
- 24 tree view tests (formatting, rendering, integration)

### Linting

```bash
npm run lint             # Check code
npm run lint:fix         # Auto-fix issues
```

### Development Mode

```bash
npm run watch            # Auto-rebuild on changes
npm run dev              # Build and run
```

## Architecture

See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) and [LOCKING-REDESIGN.md](./docs/LOCKING-REDESIGN.md) for detailed design decisions, including:
- Reader-writer lock architecture (38x performance improvement)
- Atomic multi-path locking with deadlock prevention
- Path security implementation
- Stateless HTTP transport design
- Error handling philosophy

**Quick Architecture Overview**:
```
CLI → Transport (stdio/HTTP) → MCP Server → Memory Operations
                                                ↓
                                    ┌───────────┴───────────┐
                                    ↓                       ↓
                            RW Lock Manager         Path Security
                       (concurrent reads, exclusive writes)
```

## Debugging

Enable debug logging with `--debug` flag:

```bash
memory-mcp --debug
```

Logs are written to `/tmp/memory-mcp/<instance-id>.log` in JSON format:

```json
{
  "timestamp": "2025-10-15T14:30:22.123Z",
  "level": "debug",
  "operation": "str_replace",
  "path": "/memories/notes.txt",
  "duration_ms": 45,
  "success": true
}
```

Each server instance gets a unique log file for multi-instance debugging.

## Documentation

- [Architecture Guide](./docs/ARCHITECTURE.md) - Design decisions and technical details
- [Claude Memory Tool Spec](./docs/039-Memory-tool.md) - Official memory tool documentation
- [MCP SDK Documentation](./docs/MCP-SDK-README.md) - TypeScript SDK reference

## Performance

**Operation Costs**:
- View: O(1) for directories, O(n) for files
- Create/Delete/Rename: O(1)
- Str_replace/Insert: O(n) where n = file size

**Concurrency Performance**:
- **38x speedup** for read-heavy workloads (tested with 50 concurrent clients)
- Read operations: True parallelism (non-blocking when no writers)
- Write operations: Exclusive access with minimal overhead
- Locking overhead: ~1-2ms per operation (uncontended)

**Stress Test Results** (50 concurrent clients, 40 readers + 10 writers):
- Theoretical serial: 3237ms
- Actual with RW locks: 85ms
- Speedup: 38x

## Future npm Package

This package is configured for npm publication as **`@yannbam/memory-mcp`** but is not yet published. Currently distributed via GitHub.

To prepare for future npm installation:
```bash
npm install @yannbam/memory-mcp
```

Stay tuned for the official npm release!

## License

MIT - See [LICENSE](./LICENSE) for details

## Contributing

Contributions welcome! Please:
1. Fork the repository and create a feature branch
2. Follow existing code style (e/code conventions from CLAUDE.md)
3. Add tests for new features (maintain 80%+ coverage)
4. Update documentation as needed
5. Ensure all tests pass (`npm test`)
6. Run linting (`npm run lint`)
7. Submit a pull request with clear description

## Acknowledgments

- [Anthropic](https://www.anthropic.com/) for Claude and the memory tool design
- [Model Context Protocol](https://github.com/modelcontextprotocol) for the MCP specification
- Implemented by Claude Sonnet 4.5 with janbam 🌱

---

**Status**: 🚀 Public Beta (v0.1.0) - Production Ready
**Tests**: 117/117 unit tests + integration tests + E2E validation with Claude Code
**Interface**: Unified `memory` tool matching official Anthropic spec
**Features**: All 6 commands + tree view + true RW locks (38x speedup)
**Repository**: https://github.com/yannbam/memory-mcp
