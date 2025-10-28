# Parameter Combinations - Implementation Specification

**Status**: Ready for implementation
**Created**: 2025-10-28
**Decision Authority**: janbam

This document provides complete implementation specifications for approved parameter combinations. This is a self-contained guide for the implementing session.

---

## Overview

Four parameter combinations have been approved for implementation. All are optional parameters that extend existing commands with intuitive default behaviors.

**Design Philosophy:**
- Optional parameters provide sensible defaults
- Behaviors feel natural and reduce cognitive load
- No mixing of position-based and content-based paradigms
- Empty content = deletion

---

## Feature 1: Create Empty File

**Command**: `create`
**Enhancement**: Make `file_text` parameter optional

### Current Behavior
```typescript
create({
  path: "/memories/file.txt",
  file_text: "content"  // Required
})
```

### New Behavior
```typescript
// Create file with content (existing behavior)
create({
  path: "/memories/file.txt",
  file_text: "content"
})

// Create empty file (new behavior - touch equivalent)
create({
  path: "/memories/file.txt"
})
```

### Schema Change

**Before:**
```typescript
z.object({
  command: z.literal('create'),
  path: z.string().describe('Memory path starting with /memories'),
  file_text: z.string().describe('File content to write'),
})
```

**After:**
```typescript
z.object({
  command: z.literal('create'),
  path: z.string().describe('Memory path starting with /memories'),
  file_text: z.string().default('').describe('File content to write. Defaults to empty string if omitted (creates empty file).'),
})
```

### Implementation

**Location**: `src/memory/operations.ts` - `create()` function

**Changes Required:**
- Schema already handles the default value via `.default('')`
- No handler code changes needed - empty string writes empty file
- Update interface in operations.ts to make file_text optional:

```typescript
export interface CreateCommand {
  command: 'create';
  path: string;
  file_text?: string;  // Make optional
}
```

### Tests

Add to `test/memory-operations.test.ts`:

```typescript
describe('create with optional file_text', () => {
  it('should create empty file when file_text omitted', async () => {
    const result = await operations.create(
      { command: 'create', path: '/memories/empty.txt' },
      context
    );

    expect(result).toContain('created successfully');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/empty.txt'),
      'utf-8'
    );
    expect(content).toBe('');
  });

  it('should create file with content when file_text provided', async () => {
    const result = await operations.create(
      { command: 'create', path: '/memories/data.txt', file_text: 'content' },
      context
    );

    expect(result).toContain('created successfully');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/data.txt'),
      'utf-8'
    );
    expect(content).toBe('content');
  });
});
```

---

## Feature 2: Append to File

**Command**: `insert`
**Enhancement**: Make `insert_line` parameter optional (defaults to append)

### Current Behavior
```typescript
insert({
  path: "/memories/file.txt",
  insert_line: 5,        // Required - must know line number
  insert_text: "new line"
})
```

### New Behavior
```typescript
// Insert at specific line (existing behavior)
insert({
  path: "/memories/file.txt",
  insert_line: 5,
  insert_text: "new line"
})

// Append to end (new behavior)
insert({
  path: "/memories/file.txt",
  insert_text: "new line"
})
```

### Schema Change

**Before:**
```typescript
z.object({
  command: z.literal('insert'),
  path: z.string().describe('Memory path starting with /memories'),
  insert_line: z.number().describe('Line number where text should be inserted (1-based)'),
  insert_text: z.string().describe('Text to insert'),
})
```

**After:**
```typescript
z.object({
  command: z.literal('insert'),
  path: z.string().describe('Memory path starting with /memories'),
  insert_line: z.number().optional().describe('Line number where text should be inserted (1-based). If omitted, appends to end of file.'),
  insert_text: z.string().describe('Text to insert'),
})
```

### Implementation

**Location**: `src/memory/operations.ts` - `insert()` function

**Handler Logic Changes:**

```typescript
export async function insert(command: InsertCommand, context: OperationsContext): Promise<string> {
  // Validate and convert path
  const fullPath = validateAndConvertPath(command.path, context.memoryRoot);

  // Execute with write lock and concurrency check
  return await executeWithLock(fullPath, 'write', true, async (mtimeBefore) => {
    // Check if file exists
    if (!(await exists(fullPath))) {
      throw new Error(`File not found: ${command.path}`);
    }

    // Verify it's a file, not a directory
    const stats = await fs.stat(fullPath);
    if (stats.isDirectory()) {
      throw new Error(`Cannot insert into directory: ${command.path}`);
    }

    // Read file content
    const content = await fs.readFile(fullPath, 'utf-8');
    const lines = content.split('\n');

    // Determine insert position
    let insertLine: number;
    if (command.insert_line === undefined) {
      // Append to end - insert after last line
      insertLine = lines.length + 1;
    } else {
      insertLine = command.insert_line;
      // Validate insert_line (1-based indexing)
      if (insertLine < 1) {
        throw new Error(`insert_line must be >= 1, got ${insertLine}`);
      }
      if (insertLine > lines.length + 1) {
        throw new Error(
          `insert_line ${insertLine} exceeds file length (${lines.length} lines). Maximum is ${lines.length + 1}.`
        );
      }
    }

    // Insert text at specified line (convert from 1-based to 0-based array index)
    // Remove trailing newline from insert_text to avoid double newlines
    const textToInsert = command.insert_text.replace(/\n$/, '');
    lines.splice(insertLine - 1, 0, textToInsert);

    // Write updated content
    await fs.writeFile(fullPath, lines.join('\n'));

    // Log operation
    context.logger.debug('insert', { path: command.path, insertLine, success: true });

    if (command.insert_line === undefined) {
      return `Text appended to end of ${command.path}`;
    } else {
      return `Text inserted at line ${insertLine} in ${command.path}`;
    }
  });
}
```

**Interface Update:**

```typescript
export interface InsertCommand {
  command: 'insert';
  path: string;
  insert_line?: number;  // Make optional
  insert_text: string;
}
```

### Tests

Add to `test/memory-operations.test.ts`:

```typescript
describe('insert with optional insert_line', () => {
  it('should append to end when insert_line omitted', async () => {
    // Create file with content
    await fs.writeFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'line 1\nline 2\nline 3'
    );

    const result = await operations.insert(
      {
        command: 'insert',
        path: '/memories/test.txt',
        insert_text: 'appended line'
      },
      context
    );

    expect(result).toContain('appended to end');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'utf-8'
    );
    expect(content).toBe('line 1\nline 2\nline 3\nappended line');
  });

  it('should insert at specific line when insert_line provided', async () => {
    // Create file with content
    await fs.writeFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'line 1\nline 2\nline 3'
    );

    const result = await operations.insert(
      {
        command: 'insert',
        path: '/memories/test.txt',
        insert_line: 2,
        insert_text: 'inserted line'
      },
      context
    );

    expect(result).toContain('inserted at line 2');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'utf-8'
    );
    expect(content).toBe('line 1\ninserted line\nline 2\nline 3');
  });

  it('should append to empty file when insert_line omitted', async () => {
    // Create empty file
    await fs.writeFile(
      path.join(memoryRoot, 'memories/empty.txt'),
      ''
    );

    const result = await operations.insert(
      {
        command: 'insert',
        path: '/memories/empty.txt',
        insert_text: 'first line'
      },
      context
    );

    expect(result).toContain('appended to end');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/empty.txt'),
      'utf-8'
    );
    expect(content).toBe('first line');
  });
});
```

---

## Feature 3: Delete Matching Text

**Command**: `delete`
**Enhancement**: Add `old_str/old_string` parameter for content-based deletion

### Current Behavior
```typescript
// Delete file or directory
delete({ path: "/memories/file.txt" })

// Delete specific line
delete({ path: "/memories/file.txt", delete_line: 5 })
```

### New Behavior
```typescript
// Delete file/directory (existing)
delete({ path: "/memories/file.txt" })

// Delete specific line (existing)
delete({ path: "/memories/file.txt", delete_line: 5 })

// Delete matching text from file (new)
delete({
  path: "/memories/file.txt",
  old_str: "debug code"
})
```

### Behavior Specification

**IMPORTANT**: This is NOT "delete lines containing text". This is "delete the matching text itself".

**Algorithm:**
1. Find all occurrences of `old_str` in the file
2. Replace each occurrence with empty string (like `str_replace` with `new_str=""`)
3. After deletion, if any line becomes empty, remove that empty line
4. Write the cleaned content back to file

**Example:**

**Before:**
```
This is line 1 with debug code here
This is line 2 without it
Line 3 debug code in the middle here
```

**Command:**
```typescript
delete({ path: "/memories/file.txt", old_str: "debug code " })
```

**After:**
```
This is line 1 with here
This is line 2 without it
Line 3 in the middle here
```

**Example with empty line removal:**

**Before:**
```
debug code
This is line 2
Just debug code on this line
Line 4
```

**Command:**
```typescript
delete({ path: "/memories/file.txt", old_str: "debug code" })
```

**After (empty lines removed):**
```
This is line 2
Line 4
```

### Schema Change

**Before:**
```typescript
z.object({
  command: z.literal('delete'),
  path: z.string().describe('Memory path starting with /memories'),
  delete_line: z.number().int().positive().optional()
    .describe('Optional: Delete specific line number (1-based). If provided, only that line is deleted.'),
})
```

**After:**
```typescript
z.object({
  command: z.literal('delete'),
  path: z.string().describe('Memory path starting with /memories'),
  delete_line: z.number().int().positive().optional()
    .describe('Optional: Delete specific line number (1-based). If provided, only that line is deleted.'),
  old_str: z.string().optional()
    .describe('Optional: Delete all occurrences of this text from the file. Empty lines resulting from deletion are removed. Use old_str OR old_string.'),
  old_string: z.string().optional()
    .describe('Optional: Delete all occurrences of this text from the file. Empty lines resulting from deletion are removed. Use old_str OR old_string.'),
})
```

### Implementation

**Location**: `src/memory/operations.ts` - `deleteOp()` function

**Add validation at the start:**

```typescript
export async function deleteOp(
  command: DeleteCommand,
  context: OperationsContext
): Promise<string> {
  // Prevent deletion of /memories root
  if (command.path === '/memories' || command.path === '/memories/') {
    throw new Error('Cannot delete /memories root directory');
  }

  // Normalize parameter names for old_str (forgiving naming)
  const old_str = command.old_str || command.old_string;

  // Validation: Cannot mix position-based and content-based deletion
  if (command.delete_line !== undefined && old_str !== undefined) {
    throw new Error(
      'Cannot use both delete_line and old_str - choose position-based OR content-based deletion'
    );
  }

  // Validate and convert path
  const fullPath = validateAndConvertPath(command.path, context.memoryRoot);

  // Handle text-based deletion
  if (old_str !== undefined) {
    return await deleteMatchingText(command.path, fullPath, old_str, context);
  }

  // Handle line-specific deletion (existing code)
  if (command.delete_line !== undefined) {
    // ... existing delete_line implementation ...
  }

  // Handle file/directory deletion (existing code)
  // ... existing file/directory deletion implementation ...
}
```

**Add new helper function:**

```typescript
/**
 * Delete all occurrences of matching text from a file
 * Empty lines resulting from deletion are also removed
 */
async function deleteMatchingText(
  memoryPath: string,
  fullPath: string,
  searchText: string,
  context: OperationsContext
): Promise<string> {
  // Execute with write lock and concurrency check
  return await executeWithLock(fullPath, 'write', true, async (mtimeBefore) => {
    // Check if file exists
    if (!(await exists(fullPath))) {
      throw new Error(`File not found: ${memoryPath}`);
    }

    // Verify it's a file, not a directory
    const stats = await fs.stat(fullPath);
    if (stats.isDirectory()) {
      throw new Error(`Cannot delete text from directory: ${memoryPath}`);
    }

    // Read file content
    const content = await fs.readFile(fullPath, 'utf-8');

    // Count occurrences before deletion
    const occurrences = (content.match(new RegExp(escapeRegExp(searchText), 'g')) || []).length;

    if (occurrences === 0) {
      throw new Error(`Text not found in file: "${searchText}"`);
    }

    // Delete all occurrences of the text
    const afterDeletion = content.replace(new RegExp(escapeRegExp(searchText), 'g'), '');

    // Remove empty lines
    const lines = afterDeletion.split('\n');
    const nonEmptyLines = lines.filter(line => line.trim() !== '');

    // Write cleaned content
    const finalContent = nonEmptyLines.join('\n');
    await fs.writeFile(fullPath, finalContent);

    // Log operation
    context.logger.debug('delete_text', {
      path: memoryPath,
      searchText,
      occurrences,
      success: true
    });

    return `Deleted ${occurrences} occurrence(s) of "${searchText}" from ${memoryPath}`;
  });
}

/**
 * Escape special regex characters for literal string matching
 */
function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
```

**Interface Update:**

```typescript
export interface DeleteCommand {
  command: 'delete';
  path: string;
  delete_line?: number;
  old_str?: string;
  old_string?: string;
}
```

### Tests

Add to `test/memory-operations.test.ts`:

```typescript
describe('delete with old_str (content-based deletion)', () => {
  it('should delete all occurrences of text and remove empty lines', async () => {
    await fs.writeFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'debug code\nThis is line 2\nJust debug code here\nLine 4'
    );

    const result = await operations.deleteOp(
      { command: 'delete', path: '/memories/test.txt', old_str: 'debug code' },
      context
    );

    expect(result).toContain('2 occurrence(s)');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'utf-8'
    );
    expect(content).toBe('This is line 2\nLine 4');
  });

  it('should delete text from middle of lines', async () => {
    await fs.writeFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'Start debug code end\nNormal line\nAnother debug code example'
    );

    const result = await operations.deleteOp(
      { command: 'delete', path: '/memories/test.txt', old_str: 'debug code ' },
      context
    );

    expect(result).toContain('2 occurrence(s)');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'utf-8'
    );
    expect(content).toBe('Start end\nNormal line\nAnother example');
  });

  it('should error when text not found', async () => {
    await fs.writeFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'Some content'
    );

    await expect(
      operations.deleteOp(
        { command: 'delete', path: '/memories/test.txt', old_str: 'nonexistent' },
        context
      )
    ).rejects.toThrow('Text not found');
  });

  it('should error when using both delete_line and old_str', async () => {
    await fs.writeFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'Some content'
    );

    await expect(
      operations.deleteOp(
        {
          command: 'delete',
          path: '/memories/test.txt',
          delete_line: 1,
          old_str: 'content'
        },
        context
      )
    ).rejects.toThrow('Cannot use both delete_line and old_str');
  });

  it('should accept old_string as alternative parameter name', async () => {
    await fs.writeFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'Text to delete\nKeep this'
    );

    const result = await operations.deleteOp(
      { command: 'delete', path: '/memories/test.txt', old_string: 'Text to delete' },
      context
    );

    expect(result).toContain('1 occurrence(s)');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'utf-8'
    );
    expect(content).toBe('Keep this');
  });

  it('should handle special regex characters in search text', async () => {
    await fs.writeFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'Price: $100.00\nTotal: $100.00 USD'
    );

    const result = await operations.deleteOp(
      { command: 'delete', path: '/memories/test.txt', old_str: '$100.00' },
      context
    );

    expect(result).toContain('2 occurrence(s)');
    const content = await fs.readFile(
      path.join(memoryRoot, 'memories/test.txt'),
      'utf-8'
    );
    expect(content).toBe('Price:\nTotal:  USD');
  });
});
```

---

## Feature 4: Document str_replace Deletion

**Command**: `str_replace`
**Enhancement**: Document that empty `new_str` deletes the matched text

### Current Behavior

This already works but is not documented:

```typescript
str_replace({
  path: "/memories/file.txt",
  old_str: "debug code",
  new_str: ""  // Empty string deletes the text
})
```

### Documentation Updates

**README.md** - Update the str_replace section:

```markdown
### str_replace
Replace unique text in a file (text must appear exactly once).

**Delete text by using empty replacement:**
```typescript
// Delete matching text
await memory({
  command: "str_replace",
  path: "/memories/notes.txt",
  old_str: "debug code",
  new_str: ""  // Empty string deletes the text
})
// → "File /memories/notes.txt has been edited"

// Standard replacement
await memory({
  command: "str_replace",
  path: "/memories/notes.txt",
  old_str: "old value",
  new_str: "new value"
})
// → "File /memories/notes.txt has been edited"
```

**Note**: Both `old_str`/`new_str` and `old_string`/`new_string` parameter names are accepted.
```

**Schema Description** - Update in `src/server/mcp-server.ts`:

```typescript
new_str: z.string().optional().describe(
  'Replacement text. Use new_str OR new_string. Empty string deletes the matched text.'
),
new_string: z.string().optional().describe(
  'Replacement text. Use new_str OR new_string. Empty string deletes the matched text.'
),
```

### Tests

No new tests needed - this already works. Existing tests cover this:

```typescript
// Verify existing test exists or add it
it('should delete text when new_str is empty', async () => {
  await fs.writeFile(
    path.join(memoryRoot, 'memories/test.txt'),
    'Keep this debug code and this'
  );

  const result = await operations.str_replace(
    {
      command: 'str_replace',
      path: '/memories/test.txt',
      old_str: 'debug code ',
      new_str: ''
    },
    context
  );

  expect(result).toContain('has been edited');
  const content = await fs.readFile(
    path.join(memoryRoot, 'memories/test.txt'),
    'utf-8'
  );
  expect(content).toBe('Keep this and this');
});
```

---

## Implementation Checklist

### Schema Updates
- [ ] `CreateCommandSchema`: Make file_text optional with `.default('')`
- [ ] `InsertCommandSchema`: Make insert_line optional
- [ ] `DeleteCommandSchema`: Add old_str and old_string optional parameters
- [ ] `StrReplaceCommandSchema`: Update descriptions to mention empty string deletion
- [ ] Update BOTH unified schema (`MemoryCommandSchema`) and individual schemas
- [ ] Update TypeScript interfaces in operations.ts

### Handler Implementation
- [ ] `create()`: Update interface to make file_text optional (no logic changes needed)
- [ ] `insert()`: Add logic to handle missing insert_line (append to end)
- [ ] `deleteOp()`: Add validation for parameter conflicts
- [ ] `deleteOp()`: Implement `deleteMatchingText()` helper function
- [ ] Add `escapeRegExp()` utility function

### Testing
- [ ] Add 3 tests for create with optional file_text
- [ ] Add 3 tests for insert with optional insert_line
- [ ] Add 6 tests for delete with old_str
- [ ] Verify existing str_replace empty string test exists (or add it)

**Total new tests**: ~12 tests

### Documentation
- [ ] Update README.md str_replace section with deletion example
- [ ] Update schema descriptions in mcp-server.ts
- [ ] Add release note to CHANGELOG.md

### Manual Testing
- [ ] Test with MCP-Debug tool or Claude Code
- [ ] Verify both unified tool mode and one-tool-per-command mode
- [ ] Test edge cases: empty files, special characters, multiple occurrences

---

## Expected Test Count

**Current**: 100 tests passing
**After implementation**: ~112 tests passing (12 new tests)

---

## Implementation Notes

### Case Sensitivity

All text matching is **case-sensitive**:
- `delete({ old_str: "TODO" })` will NOT match "todo"
- `str_replace({ old_str: "Debug" })` will NOT match "debug"

This is consistent with the existing `str_replace` behavior.

### Special Characters

The `escapeRegExp()` function handles special regex characters properly:
- `$100.00` matches literal dollar sign and period
- `(test)` matches literal parentheses
- `a.b` matches literal period, not "any character"

### Empty Line Definition

An empty line is defined as: `line.trim() === ''`

This means:
- Empty string: `` → empty line
- Only whitespace: `   ` → empty line
- Only tabs: `\t\t` → empty line
- Content: `  text  ` → NOT empty line

---

## Success Criteria

1. **All schemas updated** in both unified and individual modes
2. **All interfaces updated** to reflect optional parameters
3. **All handlers implemented** with proper validation
4. **All tests passing** (~112 total tests)
5. **Documentation updated** in README and CHANGELOG
6. **Manual testing successful** with both tool modes

---

## Estimated Complexity

- **Schema updates**: 30 minutes
- **Handler implementation**: 2 hours
- **Testing**: 1.5 hours
- **Documentation**: 30 minutes

**Total**: ~4.5 hours of focused work

---

## Questions & Edge Cases

### Handling Empty Files

**insert without insert_line on empty file:**
- Empty file has 0 lines
- Append means insert at position lines.length + 1 = 1
- Result: First line of file
- ✅ Correct behavior

**delete with old_str on empty file:**
- Search text not found
- Throw error: "Text not found in file"
- ✅ Correct behavior

### Multiple Empty Lines

**After delete with old_str, if multiple lines become empty:**
- All empty lines are removed
- File is compacted to only non-empty lines
- ✅ Correct behavior

### Whitespace Handling

**delete with old_str: "debug code " (trailing space):**
- Matches exactly "debug code " including the space
- Case-sensitive, space-sensitive
- ✅ Correct behavior

---

## End of Specification

This document contains all information needed to implement the approved parameter combinations. No additional design decisions are required.
