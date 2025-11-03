# Memory MCP Server

[![CI](https://github.com/yannbam/memory-mcp/workflows/CI/badge.svg)](https://github.com/yannbam/memory-mcp/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-166%20passing-success)](./test)

Memory MCP

✅ Based on Claude's native memory tool schema<br>
✅ Safe concurrent access with conflict detection<br>
🌳 Tree-view<br>
🙅‍♀️ `<boundary_setting_triggers>` *not* included

## Overview

This project implements Claude's [memory tool](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool) as a Model Context Protocol (MCP) server. It enables any MCP client (Claude Code, MCP Inspector, etc.) to have persistent memory across sessions through secure, concurrent-safe filesystem storage.

## Features

- ✅ **All 6 Memory Commands**: view, create, str_replace, insert, delete, rename
- ✅ **Flexible Tool Modes**: Single unified `memory` tool (default, matches Anthropic spec) or 6 separate tools (`memory_view`, `memory_create`, etc.)
- ✅ **Tree View Mode**: Optional hierarchical directory view with metadata (sizes, lines, timestamps)
- ✅ **High-Performance Concurrency**: True reader-writer locks for parallel reads
- ✅ **Path Security**: Comprehensive directory traversal protection
- ✅ **Dual Transport**: stdio (default) and streamable HTTP

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
      "args": ["/abs/path/to/memory-mcp/dist/index.js", "--tree-view"],
      "env": {}
    }
  }
}
```
Uses absolute path, stores memory in `./.memory/memories/` within the project. Great for project-specific documentation.

**2. Global Memory** (shared across all projects):
```json
{
  "mcpServers": {
    "global_memory": {
      "command": "node",
      "args": [
        "/absolute/path/to/memory-mcp/dist/index.js",
        "--memory-root-path",
        "/abs/path/to/global/.memory",
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
Memory MCP Server

MCP server implementation of Claude's native memory tool.
Provides persistent storage across conversations through filesystem operations.

USAGE:
  memory-mcp [options]

OPTIONS:
  --memory-root-path PATH, -m PATH   Memory storage root path (default: ./.memory)
  --transport TYPE, -t TYPE          Transport type: stdio | http (default: stdio)
  --port PORT, -p PORT               HTTP server port (default: 3000, http transport only)
  --tree-view                        Enable tree view for directory listings (default: false)
  --one-tool-per-command             Expose each command as separate tool (default: single tool)
  --debug, -d                        Enable debug logging to /tmp/memory-mcp/<instance-id>.log
  --version, -v                      Show version
  --help, -h                         Show this help message
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

# Expose each command as separate tool
memory-mcp --one-tool-per-command

# Full configuration
memory-mcp -m /var/memories -t http -p 3000 --tree-view --one-tool-per-command -d
```

## Usage

The server supports two tool exposure modes:

**Default Mode (Unified Tool)**: Exposes a single **`memory`** tool matching the [official Anthropic Memory tool specification](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool). All operations use a `command` parameter to dispatch.

**One-Tool-Per-Command Mode** (`--one-tool-per-command`): Exposes 6 separate tools (`memory_view`, `memory_create`, `memory_str_replace`, `memory_insert`, `memory_delete`, `memory_rename`). Each tool has only its relevant parameters (no `command` field).

### Commands

All 6 memory commands (unified tool uses `command` parameter, separate tools omit it):

| Command | Purpose | Example |
|---------|---------|---------|
| **view** | Show directory/file contents | `{ command: "view", path: "/memories" }` |
| **create** | Create new file | `{ command: "create", path: "/memories/notes.txt", file_text: "..." }` |
| **str_replace** | Replace unique text | `{ command: "str_replace", path: "...", old_str: "...", new_str: "..." }` |
| **insert** | Insert/append text | `{ command: "insert", path: "...", insert_text: "..." }` |
| **delete** | Delete file/dir/line/text | `{ command: "delete", path: "/memories/file.txt" }` |
| **rename** | Move/rename file/dir | `{ command: "rename", old_path: "...", new_path: "..." }` |

### Quick Examples

**View directory:**
```typescript
await memory({ command: "view", path: "/memories" })
```

**Create file:**
```typescript
await memory({
  command: "create",
  path: "/memories/notes.txt",
  file_text: "My notes here"
})
```

**Edit file:**
```typescript
await memory({
  command: "str_replace",
  path: "/memories/notes.txt",
  old_str: "old text",
  new_str: "new text"
})
```

📖 **[Full API Reference](./docs/USAGE.md)** - Complete documentation with all parameters and examples

## Concurrent Access

The server supports **multiple Claude instances** accessing the same memory files simultaneously with robust concurrency detection:

- **True RW Locks**: Powered by `@esfx/async-readerwriterlock`
- **Concurrent Reads**: Multiple readers can access the same file simultaneously (no blocking)
- **Exclusive Writes**: Writers get exclusive access, blocking both readers and other writers
- **Atomic Multi-Path Locking**: Deadlock-safe locking for operations like rename (source + destination)
- **Content-Based Modification Detection**: SHA-256 checksums detect file changes across sequential operations

**Performance**: ~38x speedup for read-heavy workloads (tested with 50 concurrent clients)

**\*Note**: Conflict detection currently works for stdio transport only (multiple separate server instances). HTTP transport runs as a single server instance with shared lock pool.*

**Modification Detection**:
- **Sequential Detection**: Detects when file modified between separate operations (read → external change → write)
- **Concurrent Detection**: Detects when file modified during lock acquisition
- **Clear Error Messages**: Shows current file contents when modification detected
- **Works Across Sessions**: Each stdio server has its own cache, all verify against disk state

**Example Scenario**:
1. Claude A reads `/memories/notes.txt` (acquires shared read lock, caches content checksum)
2. Claude B reads the same file (also acquires shared read lock - no blocking!)
3. Claude C tries to edit the file (waits for exclusive write lock)
4. Claudes A & B finish reading (release read locks)
5. Claude C acquires write lock and modifies the file
6. Claude A tries to write based on stale understanding → Error with current file contents shown

**Key Benefits**:
- Read operations never block each other (true parallelism)
- Write operations serialize correctly (data integrity)
- Detects modifications that happened minutes ago, not just during lock wait
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

**Quick Start:**
```bash
git clone https://github.com/yannbam/memory-mcp.git
cd memory-mcp
npm install
npm run build
npm test                 # 166 tests passing
```

📖 **[Contributing Guide](./CONTRIBUTING.md)** - Setup, testing, code style, and PR process

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

## License

MIT - See [LICENSE](./LICENSE) for details

## Contributing

Contributions welcome! See **[CONTRIBUTING.md](./CONTRIBUTING.md)** for guidelines on:
- Development setup and workflow
- Code style and testing requirements
- Pull request process

## Acknowledgments

- Implemented by Claude Sonnet 4.5 with janbam, curiosity and care 🐾 🌱
