# Parameter Combinations Matrix

**Purpose**: Systematic analysis of ALL possible command+parameter combinations for the memory MCP server to identify useful, confusing, and nonsensical combinations.

**Philosophy**: Parameter combinations should feel natural and intuitive. They should reduce cognitive load, not increase it.

---

## Current State (Baseline)

### Commands and Their Current Parameters

| Command | Required Parameters | Optional Parameters | Notes |
|---------|-------------------|-------------------|-------|
| `view` | path | view_range | Line range for file viewing |
| `create` | path, file_text | - | Overwrites if exists |
| `str_replace` | path, old_str/old_string, new_str/new_string | - | Forgiving naming (both variants accepted) |
| `insert` | path, insert_line, insert_text | - | 1-based indexing |
| `delete` | path | delete_line | Can delete file/dir OR single line |
| `rename` | old_path, new_path | - | Atomic rename/move |

### Parameter Inventory

**Path-related:**
- `path` - Target file/directory (used by: view, create, str_replace, insert, delete)
- `old_path` - Source path (used by: rename)
- `new_path` - Destination path (used by: rename)

**Content-related:**
- `file_text` - Full file content (used by: create)
- `insert_text` - Text to insert (used by: insert)
- `old_str/old_string` - Text to find (used by: str_replace)
- `new_str/new_string` - Replacement text (used by: str_replace)

**Position-related:**
- `view_range` - Line range [start, end] (used by: view)
- `insert_line` - Line number for insertion (used by: insert)
- `delete_line` - Line number for deletion (used by: delete)

---

## Full Combinatorial Matrix

### Legend
- ✅ **Currently Implemented** - Already works
- 🟢 **Useful** - Should implement
- 🟡 **Questionable** - Needs discussion
- 🔴 **Confusing** - Should NOT implement
- ⚫ **Nonsensical** - Technically impossible or meaningless

---

## Command: VIEW

| Parameter Combination | Status | Usefulness | Notes |
|----------------------|--------|------------|-------|
| path | ✅ | Essential | View file or directory |
| path + view_range | ✅ | High | View specific lines |
| path + file_text | 🔴 | Confusing | View is read-only, file_text implies write |
| path + old_str | 🟡 | Questionable | "Show lines containing X"? Grep functionality? |
| path + new_str | 🔴 | Confusing | View doesn't modify |
| path + insert_line | 🟡 | Questionable | "View from line N onwards"? Covered by view_range |
| path + insert_text | 🔴 | Confusing | View doesn't modify |
| path + delete_line | 🔴 | Confusing | View doesn't modify |
| path + view_range + old_str | 🟡 | Questionable | "Find X within lines Y-Z"? Too complex |

**Recommendation**: Keep view simple. No new combinations.

---

## Command: CREATE

| Parameter Combination | Status | Usefulness | Notes |
|----------------------|--------|------------|-------|
| path + file_text | ✅ | Essential | Standard file creation |
| path + file_text + view_range | 🔴 | Nonsensical | Create doesn't read |
| path + file_text + old_str | 🔴 | Confusing | You're writing file_text - why specify old_str? |
| path + file_text + insert_line | 🔴 | Confusing | You're creating whole file - where would you insert? |
| path + file_text + delete_line | 🔴 | Confusing | Contradictory: create AND delete? |
| path alone (no file_text) | 🟢 | **Useful** | **Create empty file** (touch equivalent) |

**Useful Combination:**
- **`create` with path only** → Create empty file
  - Use case: `touch` equivalent, prepare file for future edits
  - Implementation: Accept optional file_text, default to empty string
  - Schema: `file_text: z.string().default('')`

---

## Command: STR_REPLACE

| Parameter Combination | Status | Usefulness | Notes |
|----------------------|--------|------------|-------|
| path + old_str + new_str | ✅ | Essential | Standard text replacement |
| path + old_str + new_str="" | 🟢 | **Useful** | **Delete matching text** |
| path + old_str + view_range | 🟡 | Questionable | "Replace only within lines X-Y"? Complex |
| path + old_str + insert_line | 🔴 | Confusing | str_replace finds text, insert_line is positional |
| path + old_str + delete_line | 🔴 | Confusing | str_replace is content-based, delete_line is position-based |
| path + old_str only (no new_str) | 🟢 | **Useful** | **Delete matching text** (same as empty new_str) |

**Useful Combinations:**
1. **`str_replace` with empty new_str** → Delete matching text
   - Already possible but should be documented
   - Example: `old_str: "debug code", new_str: ""` deletes all debug code

2. **`str_replace` without new_str parameter** → Delete matching text
   - More explicit than empty string
   - Implementation: Make new_str optional, default to ""
   - Schema: `new_str: z.string().default('')`

---

## Command: INSERT

| Parameter Combination | Status | Usefulness | Notes |
|----------------------|--------|------------|-------|
| path + insert_line + insert_text | ✅ | Essential | Standard line insertion |
| path + insert_text only | 🟡 | Questionable | Where to insert? Append to end? |
| path + insert_line + file_text | 🔴 | Confusing | insert_text vs file_text? Which one? |
| path + insert_line + old_str | 🔴 | Confusing | Positional (line) vs content-based (str)? |
| path + insert_line + view_range | 🔴 | Nonsensical | Insert doesn't read |

**Useful Combination:**
- **`insert` without insert_line** → Append to end of file
  - Use case: Quick append without knowing file length
  - Implementation: Make insert_line optional, default to file_length + 1
  - Schema: `insert_line: z.number().optional()` with handler default logic
  - Behavior: If omitted, append to end

---

## Command: DELETE

| Parameter Combination | Status | Usefulness | Notes |
|----------------------|--------|------------|-------|
| path only | ✅ | Essential | Delete file or directory |
| path + delete_line | ✅ | Essential | Delete specific line |
| path + old_str | 🟢 | **Useful** | **Delete lines containing text** |
| path + view_range | 🟡 | Questionable | Delete lines X-Y? Useful but risky |
| path + file_text | 🔴 | Nonsensical | Delete doesn't write |
| path + insert_line | 🔴 | Confusing | Use delete_line instead |
| path + old_str + delete_line | 🔴 | Confusing | Content-based vs position-based conflict |

**Useful Combinations:**
1. **`delete` with old_str** → Delete all lines containing text
   - Use case: Remove all TODO comments, delete debug statements
   - Implementation: Find all lines matching old_str, remove them
   - Question: Exact match or substring? (Suggest: substring for flexibility)
   - Example: `path: "/memories/notes.txt", old_str: "TODO"` removes all TODO lines

2. **`delete` with view_range** → Delete lines X through Y
   - Use case: Remove section of file
   - Risky: Off-by-one errors, large deletions
   - Safer alternative: Use str_replace to target content
   - **Recommendation**: Skip this - too error-prone

---

## Command: RENAME

| Parameter Combination | Status | Usefulness | Notes |
|----------------------|--------|------------|-------|
| old_path + new_path | ✅ | Essential | Standard rename/move |
| old_path + new_path + any other param | 🔴 | Confusing | Rename is atomic, shouldn't mix with content ops |

**Recommendation**: Keep rename pure. No combinations.

---

## Cross-Command Analysis

### Patterns That Emerged

#### 1. **Empty Content = Deletion**
- `str_replace` with empty new_str → delete matching text
- `create` with empty file_text → create empty file

#### 2. **Optional Position = Append**
- `insert` without insert_line → append to end

#### 3. **Content-Based Operations**
- `delete` with old_str → delete lines containing text
- `str_replace` with old_str → replace text

#### 4. **Position-Based Operations**
- `delete` with delete_line → delete specific line
- `insert` with insert_line → insert at specific line
- `view` with view_range → view specific lines

### Design Principle: Don't Mix Paradigms

**Position-based** and **content-based** operations should NOT be combined:
- ❌ `insert_line` + `old_str` (position + content)
- ❌ `delete_line` + `old_str` (position + content)
- ❌ `view_range` + `old_str` (position + content)

Mixing creates confusion: "Does it insert at the line OR where it finds the text?"

---

## Recommended Implementations

### Priority 1: High Value, Low Complexity

| Combination | Command | Behavior | Implementation |
|------------|---------|----------|----------------|
| Empty file_text | `create` | Create empty file (touch) | Make file_text optional with default `""` |
| Empty new_str | `str_replace` | Delete matching text | Already works, just document it |
| Missing insert_line | `insert` | Append to end of file | Make insert_line optional, default to EOF |

### Priority 2: High Value, Medium Complexity

| Combination | Command | Behavior | Implementation |
|------------|---------|----------|----------------|
| path + old_str | `delete` | Delete lines containing text | New parameter, filter + delete matching lines |

### Priority 3: Questionable Value

| Combination | Command | Behavior | Reasoning |
|------------|---------|----------|-----------|
| path + view_range | `delete` | Delete line range | Too risky, off-by-one errors, better to use content-based |
| path + old_str | `view` | Show lines containing text | Grep functionality, scope creep |

---

## Implementation Design

### 1. Create with Optional file_text

**Current:**
```typescript
z.object({
  command: z.literal('create'),
  path: z.string(),
  file_text: z.string(), // Required
})
```

**Proposed:**
```typescript
z.object({
  command: z.literal('create'),
  path: z.string(),
  file_text: z.string().default(''), // Optional, defaults to empty
})
```

**Behavior:**
- `create({ path: "/memories/empty.txt" })` → Creates empty file
- `create({ path: "/memories/data.txt", file_text: "content" })` → Creates file with content

---

### 2. Str_replace with Optional new_str

**Current:**
```typescript
z.object({
  command: z.literal('str_replace'),
  path: z.string(),
  old_str: z.string().optional(),
  old_string: z.string().optional(),
  new_str: z.string().optional(),    // Already optional for forgiving naming
  new_string: z.string().optional(),
})
```

**Enhancement:**
- Document that empty new_str deletes the matched text
- Consider: Make new_str truly optional (default to empty string)

**Behavior:**
- `str_replace({ path, old_str: "debug", new_str: "" })` → Delete "debug"
- `str_replace({ path, old_str: "debug" })` → Same as above (if we make new_str default to "")

**Question for User:** Should omitting new_str delete the text, or should it be an error?

---

### 3. Insert with Optional insert_line

**Current:**
```typescript
z.object({
  command: z.literal('insert'),
  path: z.string(),
  insert_line: z.number(), // Required
  insert_text: z.string(),
})
```

**Proposed:**
```typescript
z.object({
  command: z.literal('insert'),
  path: z.string(),
  insert_line: z.number().optional(), // Optional
  insert_text: z.string(),
})
```

**Implementation:**
```typescript
// In insert handler
if (command.insert_line === undefined) {
  // Append to end
  const lines = content.split('\n');
  command.insert_line = lines.length + 1; // Append after last line
}
```

**Behavior:**
- `insert({ path, insert_text: "new line" })` → Append to end
- `insert({ path, insert_line: 5, insert_text: "new line" })` → Insert at line 5

---

### 4. Delete with old_str (New Feature)

**Current:**
```typescript
z.object({
  command: z.literal('delete'),
  path: z.string(),
  delete_line: z.number().optional(),
})
```

**Proposed:**
```typescript
z.object({
  command: z.literal('delete'),
  path: z.string(),
  delete_line: z.number().optional(),
  old_str: z.string().optional(),
  old_string: z.string().optional(), // Forgiving naming
})
```

**Validation:**
```typescript
// Cannot mix position-based and content-based deletion
if (delete_line && old_str) {
  throw new Error('Cannot use both delete_line and old_str - choose position-based OR content-based deletion');
}
```

**Implementation:**
```typescript
if (command.old_str || command.old_string) {
  const searchText = command.old_str || command.old_string;
  const lines = content.split('\n');
  const filteredLines = lines.filter(line => !line.includes(searchText));
  const deletedCount = lines.length - filteredLines.length;

  if (deletedCount === 0) {
    throw new Error(`No lines found containing "${searchText}"`);
  }

  // Write filtered content
  await fs.writeFile(fullPath, filteredLines.join('\n'));
  return `Deleted ${deletedCount} line(s) containing "${searchText}" from ${command.path}`;
}
```

**Behavior:**
- `delete({ path, old_str: "TODO" })` → Delete all lines containing "TODO"
- `delete({ path, delete_line: 5 })` → Delete line 5
- `delete({ path })` → Delete entire file/directory
- `delete({ path, delete_line: 5, old_str: "TODO" })` → **ERROR** (can't mix)

**Question for User:**
- Should old_str match substring (`.includes()`) or exact line (`.===`)?
- Should it be case-sensitive or case-insensitive?

---

## Summary of Recommendations

### ✅ Implement (High Value)

1. **create with optional file_text** → Create empty file (touch equivalent)
2. **insert with optional insert_line** → Append to end of file
3. **str_replace documentation** → Clarify that empty new_str deletes text
4. **delete with old_str** → Delete lines containing text (NEW feature)

### 🟡 Consider (Questionable Value)

1. **delete with view_range** → Delete line range (risky, error-prone)
2. **str_replace with view_range** → Replace within line range (complex)
3. **view with old_str** → Grep functionality (scope creep)

### ❌ Reject (Confusing or Nonsensical)

- Any combination mixing position-based + content-based operations
- Any combination mixing read + write operations
- Any combination mixing whole-file + partial-file operations

---

## Design Philosophy

**Guiding Principles:**

1. **Consistency**: Similar parameters should behave similarly across commands
2. **Clarity**: Parameter combinations should have one obvious interpretation
3. **Safety**: Avoid combinations that could cause unexpected data loss
4. **Simplicity**: Don't mix paradigms (position vs content, read vs write)

**Key Insights:**

- **Empty = Create/Delete**: Missing or empty content parameters mean create empty or delete content
- **Optional Position = Append**: Missing line number means end of file
- **Never Mix Paradigms**: Position-based and content-based operations are incompatible
- **Read Commands Stay Pure**: View operations should never modify files

---

## Next Steps

1. **User Review**: Get feedback on Priority 1 recommendations
2. **Schema Updates**: Modify Zod schemas for approved combinations
3. **Implementation**: Add handler logic for new behaviors
4. **Testing**: Add comprehensive tests for all new combinations
5. **Documentation**: Update README with new parameter combinations

**Estimated Complexity:**
- Priority 1 implementations: ~2-3 hours
- Testing: ~1-2 hours
- Documentation: ~1 hour

**Total:** ~4-6 hours of focused work
