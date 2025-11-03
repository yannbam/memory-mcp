# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server implementation of Claude's native Memory tool. It provides persistent storage across Claude Code sessions through filesystem operations, enabling Claude to:
- Store and retrieve information across conversations
- Build knowledge over time without context window limitations
- Learn from past interactions and maintain project context

**Current Status**: Infrastructure complete, implementation pending.

## Architecture

### Core Design Pattern

The project implements Claude's Memory tool specification as an MCP server using the `@modelcontextprotocol/sdk`. The architecture consists of:

1. **Memory Commands** (6 operations matching the Memory tool spec):
   - `view` - Directory listings or file contents with optional line ranges
   - `create` - Create/overwrite files
   - `str_replace` - Replace unique text in files
   - `insert` - Insert text at specific line numbers
   - `delete` - Remove files/directories
   - `rename` - Move/rename files/directories

2. **Path Security Layer**: All operations must validate paths to prevent directory traversal attacks. All paths must:
   - Start with `/memories`
   - Be resolved to canonical form and verified to remain within the memory root
   - Reject patterns like `../`, `..\\`, URL-encoded traversal (`%2e%2e%2f`)

3. **MCP Server Integration**: Uses `@modelcontextprotocol/sdk` to expose memory operations as MCP tools with Zod schemas for validation.

### Key References

- **Memory Tool Specification**: `docs/039-Memory-tool.md` - Official Claude Memory tool documentation
- **MCP SDK Documentation**: `docs/MCP-SDK-README.md` - TypeScript SDK reference
- **Implementation Example**: `docs/tools-helpers-memory.ts` - TypeScript reference implementation (filesystem-based)

## Development Commands

### Build and Run
```bash
npm run build          # Compile TypeScript to dist/
npm run watch          # Watch mode for development
npm run dev            # Build and run the server
```

### Testing
```bash
npm test               # Run all tests
npm run test:watch     # Watch mode for tests
npm run test:coverage  # Run tests with coverage report (80% threshold)
```

### Code Quality
```bash
npm run lint           # Check for linting issues
npm run lint:fix       # Auto-fix linting issues
```

## TypeScript Configuration

### ES Modules with Node16

This project uses **ES Modules** (`"type": "module"` in package.json) with Node16 module resolution:
- Target: `es2018` (aligned with MCP SDK)
- Module: `Node16`
- `isolatedModules: true` required for ts-jest

### Testing with ESM

Jest requires special configuration for ESM:
- Tests use `NODE_OPTIONS=--experimental-vm-modules` to enable ESM support
- Jest config uses `preset: 'ts-jest/presets/default-esm'`
- This is **critical** - tests will fail without the NODE_OPTIONS flag

## Security Considerations

### Path Traversal Protection (CRITICAL)

When implementing memory commands, **ALL paths must be validated** to prevent security vulnerabilities:

```typescript
// Example validation pattern from TypeScript reference
private validatePath(memoryPath: string): string {
  if (!memoryPath.startsWith('/memories')) {
    throw new Error(`Path must start with /memories`);
  }

  const fullPath = path.join(this.memoryRoot, relativePath);
  const resolvedPath = path.resolve(fullPath);
  const resolvedRoot = path.resolve(this.memoryRoot);

  if (!resolvedPath.startsWith(resolvedRoot)) {
    throw new Error(`Path would escape /memories directory`);
  }

  return resolvedPath;
}
```

Attack vectors to test:
- `../` and `..\\` sequences
- URL-encoded traversal: `%2e%2e%2f`
- Absolute paths outside memory root
- Symlink attacks

### Memory Storage Location

Default: `./memory/memories/` relative to server working directory. Configurable via environment variables.

## Implementation Notes

### Error Handling

Follow the same error handling patterns as the text-editor tool:
- File not found errors
- Path validation errors
- Permission errors
- Invalid operation errors (e.g., text not unique in str_replace)

### MCP Tool Registration

Use Zod schemas with `zod-to-json-schema` for tool definitions. Each memory command needs:
1. Zod schema defining input parameters
2. Tool handler function
3. Registration with MCP server

### Dependencies Version Alignment

Key dependencies are pinned to match MCP SDK:
- `zod: ^3.23.8` - Schema validation
- `zod-to-json-schema: ^3.24.1` - Schema to JSON conversion
- `@modelcontextprotocol/sdk: ^1.0.4` - MCP server implementation

## CI/CD

GitHub Actions workflow runs on `push` and `pull_request` to `main` and `dev` branches:
- Linting with ESLint
- TypeScript compilation
- Tests with coverage on Node 18.x, 20.x, 22.x
- Coverage upload to Codecov (Node 22.x only)

---

## 🧠 Memory System: Evolutionary Learning Across Sessions

### Core Philosophy

**Memory is a LEARNING SYSTEM that evolves** - not just storage but active knowledge refinement across sessions. Each session doesn't just add memories; it improves, corrects, and synthesizes existing knowledge into deeper understanding.

Memory preserves what matters:
- **Surprises** - Non-obvious behaviors that violate expectations
- **Important facts** - Key information discovered during work
- **Deferred tasks** - "TODO later: investigate why X fails intermittently"
- **Reminders** - Things to check or revisit in future sessions
- **Partial progress** - Where you left off on complex problems
- **Contextual breadcrumbs** - Information that helps resume work

The test: **"Is this worth preserving across sessions?"**

### The Three Memory Zones

1. **Plan** (via PlanAndTrack) - Structured tasks and current objectives
2. **Short-term memory** (.memory/memories/short-term.md) - Active session discoveries + handoff
3. **Long-term memory** (.memory/memories/long-term.md) - Evolving wisdom that transcends sessions

### Session Lifecycle Workflow

#### 🌅 **SESSION START**
1. Read memories to absorb context
2. Create session plan from memories + new instructions
3. Move plan-captured items out of short-term Quick Notes
4. **PURGE leftover Quick Notes** - if any remain from previous session, delete them (they should have been processed during previous session end)
5. Space cleared for new discoveries

#### ⚡ **DURING SESSION**
- Add to Quick Notes immediately when you discover/decide/defer something
- Don't overthink categorization - just capture it
- One line = one atomic memory (1-3 sentences for completeness)
- Search existing memories before adding - maybe there's knowledge to refine?

**Quick Note Shortcut:**
```
memory(command: "insert", path: "/memories/short-term.md", insert_text: "Your memory here [emoji]")
```
Omitting `insert_line` appends to the end of the file - perfect for quickly adding Quick Notes without needing to know line numbers.

#### 🌙 **SESSION END**
1. Review Quick Notes - what patterns emerged?
2. Promote enduring insights to appropriate long-term sections
3. Update/delete obsolete long-term entries (knowledge evolution!)
4. Write clear handoff in short-term for next session
5. **PURGE ALL Quick Notes** - delete the entire ## Quick Notes section content (keep the section header with description, but remove all entries)

### Memory Format & Operations

**Core Rules:**
- **One line = one atomic memory** (can be 1-3 comprehensive sentences)
- **No human formatting** - This is Claude's notepad, not a markdown document
- **Self-contained clarity** - Each line must make sense without context
- **Immediate capture** - Write short-term memories when discovered, not batched later

**Long-term Memory Evolution Operations:**
- **SEARCH before adding** - Is there existing knowledge to refine?
- **UPDATE in place** - Use str_replace to evolve existing memories
- **CONSOLIDATE related** - Merge observations into unified insights
- **PRUNE obsolete** - Delete what's definitively wrong or superseded

Instead of accumulating redundant entries, refine knowledge in place using str_replace!

### Emoji Markers: Multi-Dimensional System

**Emoji markers per memory** - use them when they add meaningful dimensions to a memory.

**Format (when used):** `[[emoji1][emoji2][emoji3]]` at the end of the line

**Use 0 to 3 emojis as needed:**
- **No emojis:** Simple facts - "Tests run with `npm test` and require NODE_ENV=test"
- **One emoji:** Basic context - `Path validation must check resolved path stays within root [⚠️]`
- **Two emojis:** More context - `npm audit fix breaks @modelcontextprotocol/sdk peer dependencies [💀🔧]`
- **Three emojis:** Rich dimensions - `str_replace silently fails if old_str appears multiple times [⚠️🔧🤯]`

**Suggested dimensions (use any emojis you want!):**
1. **Kind** - Type of information (⚠️ warning, 💡 insight, 🔄 deferred, ✅ validated, 💀 fatal, etc.)
2. **Object** - Domain/area (🔧 tool, 📦 dependency, 🏗️ architecture, 🧪 testing, 📝 docs, etc.)
3. **Emotion** - Experience quality (🤯 surprising, 😅 relief, 🎯 clarity, 🔥 critical, 🤔 open question)

CRITICAL: **The entire emoji space is available** - these are just examples. Use ANY emoji and ANY combination that adds meaningful dimensions to the memory.

**IMPORTANT - Memory Markers vs Decorative Emojis:**

**Memory markers are visual highlighters** - they make specific memory lines stand out when scanning, like highlighting important passages in a book:
- ✅ **GOOD**: `WARNING: Path validation must fail on traversal attempts [⚠️]` - the [⚠️] marker makes this line jump out visually
- ✅ **GOOD**: `Checksum detection fully validated and working [✅🧪]` - markers help you spot validated facts quickly
- ✅ **GOOD**: `CRITICAL BUG: str_replace fails silently on duplicates [💀🔧]` - the [💀] makes you notice this immediately

**Decorative emojis are visual clutter** - they don't mark specific memories, they just decorate prose:
- ❌ **BAD**: "Feature complete, tested, and validated. Ready to merge! 🚀" - this is prose, not a memory entry
- ❌ **BAD**: "All 162 unit tests passing ✅" - this is a status bullet, not a memory entry
- ❌ **BAD**: "COMPLETED & ARCHIVED ✅" - this is a section heading, not a memory entry

**The distinction:**
- **Memory entries** (one-line atomic facts in long-term.md or short-term.md Quick Notes) → USE markers to highlight
- **Everything else** (handoffs, status updates, descriptions, headings, bullets) → NO emojis

Think of emoji markers as **visual bookmarks** in your memory - they help you quickly find "all warnings", "all critical issues", "all validated facts" when scanning hundreds of memory lines.

### What Makes Memory Valuable?

**KEEP memories that:**
- Save future sessions from wasting time
- Capture non-obvious behavior: `str_replace silently fails if old_str appears multiple times in file [⚠️🔧🤯]`
- Record costly mistakes: `Never run npm audit fix - breaks @modelcontextprotocol/sdk peer deps [💀📦]`
- Document workarounds: `Use path.resolve() + startsWith() check to prevent directory traversal`
- Track deferred work: `TODO later: investigate HTTP transport timeout after 5min idle [🔄]`
- Note simple facts: `Line 185-231 in operations.ts handles view range validation`

**SKIP memories that are:**
- Obvious from documentation
- Commentary without actionable content

### The Balance: Evolution Without Bloat

Long-term memory should become MORE VALUABLE over time, not just LARGER:
- Refine vague observations into precise knowledge
- Replace shallow understanding with deep insights
- Keep the BEST solution (unless alternatives serve different contexts)
- Consolidate related entries into comprehensive understanding
- Delete definitively wrong information immediately

**⚠️ CRITICAL - Avoid Forced Overdoing:**
When processing/managing long-term memory, make MINIMAL targeted changes after thoughtful reflection and consideration:
- **DON'T** add information that's already obvious in README/code/docs
- **DON'T** duplicate knowledge across multiple sections
- **DON'T** invent new details out of the blue
- **DON'T** reorganize or "improve" structure unnecessarily
- **DO** simply remove obsolete entries
- **DO** update memories when reality has changed
- **DO** consolidate memories when possible without losing relevant information

### Critical Implementation Notes

**For short-term.md:**
- Quick Notes section at the END for rapid capture
- Session handoff with clear continuation point
- Current sessions goals, session ID, branch 
- Prefer atomic line edits over multi-line replacements

**For long-term.md:**
- Generic sections that apply to ANY project
- Add new sections as needed - don't be constrained!
- Each section has italic description of what goes there
- Prefer atomic line edits over multi-line replacements

**Remember:** This is an evolving learning system. Each session builds on the last, creating collective intelligence that improves over time

---

## Project Memory

DO NOT EDIT MEMORY IN THIS FILE - USE THE project_memory TOOL INSTEAD.

## <project_memory>
@.memory/memories/short-term.md
@.memory/memories/long-term.md
</project_memory>