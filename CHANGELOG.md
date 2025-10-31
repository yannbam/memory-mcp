# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-16 - Public Beta

### Added
- All 6 memory commands (view, create, str_replace, insert, delete, rename)
- Tree view mode for hierarchical directory display with sizes, line counts, timestamps
- Reader-writer locks using @esfx/async-readerwriterlock (38x performance improvement on concurrent reads)
- stdio and HTTP transports for MCP server
- Path security with 27 comprehensive unit tests preventing directory traversal attacks
- Debug logging to /tmp/memory-mcp/<instance-id>.log
- Full TypeScript implementation with strict mode and Zod validation
- 85 unit tests + integration tests + end-to-end validation with actual Claude Code instance
- Comprehensive documentation (README, ARCHITECTURE, LOCKING-REDESIGN, Memory tool spec)
- Project uses its own memory system for documentation (dogfooding)

### Security
- Multi-layer path traversal protection
- Comprehensive security test suite covering all known attack vectors
- URL-encoded traversal prevention
- Absolute path rejection outside memory root

## [Unreleased]

### Changed
- **Concurrency Detection**: Replaced mtime-based detection with content checksum verification (SHA-256)
  - Now detects modifications across sequential operations (not just during lock wait)
  - Works across separate stdio MCP server processes via shared disk state
  - Each server caches checksums, compares against current file state before writes
  - Performance overhead: ~0.4ms for typical 10KB files (negligible for human-readable notes)

### Improved
- **Error Messages**: File modification errors now show complete current file contents
  - Clear explanation: "File has been modified by another process"
  - Content preview with 5000-char truncation for large files
  - Actionable guidance: "Please review the current contents and retry if appropriate"

### Added
- **Parameter combinations** - Optional parameters for more intuitive usage:
  - `create` without `file_text` creates empty file (touch equivalent)
  - `insert` without `insert_line` appends to end of file
  - `delete` with `old_str`/`old_string` deletes unique text occurrence
  - `str_replace` without `new_str` deletes unique text (defaults to empty string)
- **Forgiving parameter naming** - Accept both `old_str`/`new_str` and `old_string`/`new_string` interchangeably
- **Improved UX messaging** - Friendly messages for empty content:
  - `view` on empty file: "Memory file is empty."
  - `view` on empty directory: "Directory is empty." (both simple and tree modes)
  - `create` without `file_text`: "Created empty memory file."

### Changed
- **BREAKING**: `create` command now fails if file already exists (deviation from Anthropic spec)
  - Previously: silently overwrote existing files
  - Now: throws error "File already exists at {path}"
  - Enforces create-only semantics for safety
- All text-based operations (`str_replace`, `delete` with `old_str`) now require unique occurrences
- Both operations fail with clear error if text appears multiple times

### Planned
- npm package publication as @yannbam/memory-mcp
- Additional configuration options
- Performance optimizations for large memory stores

---

[0.1.0]: https://github.com/yannbam/memory-mcp/releases/tag/v0.1.0
