# Memory MCP Server

[![CI](https://github.com/janbam/memory-mcp/workflows/CI/badge.svg)](https://github.com/janbam/memory-mcp/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-59%20passing-success)](./test)

MCP server implementation of Claude's native memory tool for persistent storage across conversations.

## Overview

This project implements Claude's [memory tool](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool) as a Model Context Protocol (MCP) server. It enables any MCP client (Claude Code, MCP Inspector, etc.) to have persistent memory across sessions through secure, concurrent-safe filesystem storage.

## Features

- ✅ **All 6 Memory Commands**: view, create, str_replace, insert, delete, rename
- ✅ **Concurrent Access Safe**: File locking with optimistic concurrency control for multiple Claude instances
- ✅ **Path Security**: Comprehensive directory traversal protection (27 security tests)
- ✅ **Dual Transport**: stdio (default) and streamable HTTP
- ✅ **Type-Safe**: Full TypeScript with Zod runtime validation
- ✅ **Debug Logging**: Optional structured JSON logging to `/tmp/memory-mcp/`
- ✅ **Simple & Minimal**: No unnecessary features, just what you need

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

Add to your Claude Code `.mcp.json` configuration:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/absolute/path/to/memory-mcp/dist/index.js"],
      "env": {}
    }
  }
}
```

With custom memory root:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": [
        "/absolute/path/to/memory-mcp/dist/index.js",
        "--memory-root-path",
        "/home/user/.my-memories"
      ],
      "env": {}
    }
  }
}
```

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

# Full configuration
memory-mcp -m /var/memories -t http -p 3000 -d
```

## Memory Commands

### view
Show directory contents or file contents with optional line ranges.

```typescript
// View directory
await memory_view({ path: "/memories" })
// → "Directory: /memories\n- notes.txt\n- ideas/"

// View file
await memory_view({ path: "/memories/notes.txt" })
// → "   1: First note\n   2: Second note"

// View specific lines
await memory_view({
  path: "/memories/notes.txt",
  view_range: [2, 5]
})
// → "   2: Second note\n   3: Third note..."
```

### create
Create or overwrite files (creates parent directories as needed).

```typescript
await memory_create({
  path: "/memories/todo.txt",
  file_text: "- Task 1\n- Task 2"
})
// → "File created successfully at /memories/todo.txt"
```

### str_replace
Replace unique text in a file (text must appear exactly once).

```typescript
await memory_str_replace({
  path: "/memories/notes.txt",
  old_str: "old value",
  new_str: "new value"
})
// → "File /memories/notes.txt has been edited"
```

### insert
Insert text at a specific line number.

```typescript
await memory_insert({
  path: "/memories/todo.txt",
  insert_line: 2,
  insert_text: "- Urgent task"
})
// → "Text inserted at line 2 in /memories/todo.txt"
```

### delete
Delete files or directories (recursive for directories).

```typescript
await memory_delete({ path: "/memories/old-notes.txt" })
// → "File deleted: /memories/old-notes.txt"

await memory_delete({ path: "/memories/archive" })
// → "Directory deleted: /memories/archive"
```

### rename
Rename or move files/directories (creates parent directories as needed).

```typescript
await memory_rename({
  old_path: "/memories/draft.txt",
  new_path: "/memories/final.txt"
})
// → "Renamed /memories/draft.txt to /memories/final.txt"
```

## Concurrent Access

The server supports **multiple Claude instances** accessing the same memory files simultaneously through a hybrid locking strategy:

- **File Locking**: Cross-process locks via `proper-lockfile`
- **Optimistic Concurrency**: Detects concurrent modifications using mtime
- **Smart Behavior**:
  - Read operations: Wait for writers, then read (no errors)
  - Write operations: Detect changes during lock wait, error if file was modified

**Example Scenario**:
1. Claude A starts editing `/memories/notes.txt`
2. Claude B tries to edit the same file
3. Claude B waits for Claude A's lock
4. Claude B acquires lock, detects file changed
5. Error: "File has been modified by another process. Please read the file again and retry your operation."
6. Claude B reads fresh content and retries

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
git clone https://github.com/janbam/memory-mcp.git
cd memory-mcp
npm install
npm run build
```

### Testing

```bash
npm test                 # Run all tests (59 passing)
npm run test:coverage    # Run with coverage report (80%+ target)
npm run test:watch       # Watch mode
```

**Test Coverage**:
- 27 path security tests (directory traversal attacks)
- 32 memory operations tests (all 6 commands + edge cases)

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

See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for detailed design decisions, including:
- Hybrid concurrency strategy
- Path security implementation
- Stateless HTTP transport design
- Error handling philosophy

**Quick Architecture Overview**:
```
CLI → Transport (stdio/HTTP) → MCP Server → Memory Operations
                                                ↓
                                    ┌───────────┴───────────┐
                                    ↓                       ↓
                              File Locking           Path Security
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

**Locking Overhead**: ~1-2ms uncontended, waits indefinitely when contended

## License

MIT

## Contributing

Contributions welcome! Please:
1. Follow existing code style (e/code conventions)
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## Acknowledgments

- [Anthropic](https://www.anthropic.com/) for Claude and the memory tool design
- [Model Context Protocol](https://github.com/modelcontextprotocol) for the MCP specification
- Implemented by Claude Sonnet 4.5 with janbam 🌱

---

**Status**: ⚠️ Core Implementation Complete - Needs Integration Testing
**Version**: 0.1.0
**Tests**: 59/59 passing (unit tests only)
**Next Step**: Manual testing with MCP Inspector and Claude Code
