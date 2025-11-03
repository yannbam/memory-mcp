# Memory Tool Usage Guide

Complete API reference for the memory MCP server's unified `memory` tool.

## Overview

The server exposes a single unified **`memory`** tool with a `command` parameter that determines the operation. This matches the [official Anthropic Memory tool specification](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool).

## Parameter Combinations Quick Reference

| Command | Required | Optional | Notes |
|---------|----------|----------|-------|
| **view** | `path` | `view_range` | `view_range` only for files: `[start, end]` or `[start, -1]` for EOF |
| **create** | `path` | `file_text` | Omit `file_text` to create empty file |
| **str_replace** | `path`<br>`old_str` or `old_string` | `new_str` or `new_string` | Omit `new_str` to delete text. Text must be unique. |
| **insert** | `path`<br>`insert_text` | `insert_line` | Omit `insert_line` to append to end |
| **delete** | `path` | `delete_line`<br>`old_str` or `old_string` | Choose one: line number, text match, or neither (deletes file/dir). Text must be unique. |
| **rename** | `old_path`<br>`new_path` | - | Creates parent directories as needed |

**Forgiving parameter naming:** Both `old_str`/`new_str` and `old_string`/`new_string` are accepted interchangeably.

---

## Commands

### view

Show directory contents or file contents with optional line ranges.

**Directory View Modes:**
- **Simple mode** (default): Flat list of files and directories
- **Tree view mode** (with `--tree-view` flag): Hierarchical structure with metadata

**Examples:**

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

---

### create

Create new files (fails if file already exists, creates parent directories as needed).

**Note**: This implementation differs from Anthropic's spec which allows overwriting. This MCP server enforces create-only semantics for safety.

**Examples:**

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

---

### str_replace

Replace unique text in a file (text must appear exactly once).

**Examples:**

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

---

### insert

Insert text at a specific line number, or append to end of file.

**Examples:**

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

---

### delete

Delete files, directories, specific lines, or unique text occurrences.

**Examples:**

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

---

### rename

Rename or move files/directories (creates parent directories as needed).

**Examples:**

```typescript
await memory({
  command: "rename",
  old_path: "/memories/draft.txt",
  new_path: "/memories/final.txt"
})
// → "Renamed /memories/draft.txt to /memories/final.txt"
```

---

## Path Requirements

All memory paths must:
- Start with `/memories`
- Be valid filesystem paths
- Not contain directory traversal patterns (`../`, `..\\`, etc.)

Virtual paths (MCP interface): `/memories/notes.txt`
Filesystem paths: `<memory-root>/memories/notes.txt`

## Error Handling

The server provides clear error messages for:
- File not found
- Path security violations
- Text not unique (for str_replace/delete with old_str)
- Concurrent modification detection
- Invalid parameter combinations

When a file is modified by another process, the error message includes:
- Clear explanation of what happened
- **Complete current file contents** (no truncation)
- Actionable guidance to retry with fresh data
