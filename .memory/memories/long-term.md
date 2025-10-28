# Long-Term Memory (Persistent Knowledge)

## Architecture & Design
_How the system actually works vs how it was intended to work_

[🏗️💡] MCP server supports dual tool exposure modes: unified tool with command parameter (default) vs separate tools per command
Conditional registration in createMemoryServer() controlled by oneToolPerCommand boolean flag
Both modes use identical underlying operations - only tool registration differs
[🧪💡] Unified tool description is intentionally minimal/commented out - experiment to test HOW Claude uses commands/parameters intuitively without detailed manual
This is NOT incomplete - it's deliberate UX testing for future session analysis
[🎯💡] Parameter combination design principle: Never mix paradigms (position-based + content-based = confusing)
Position-based: insert_line, delete_line, view_range (specific line numbers)
Content-based: old_str, new_str, file_text (text search/replacement)
Read vs Write: view operations never modify files
Mixing these creates ambiguity and should be rejected at schema validation level


## Performance & Optimization
_Measured performance characteristics, bottlenecks, successful optimizations_

[⚠️🔧] Zod .refine() creates ZodEffects wrapper which breaks discriminated unions and .shape property access - for discriminated unions, move validation logic into the handler function instead of schema refinement
[✅🎯] Forgiving parameter naming implementation: Accept both old_str/new_str AND old_string/new_string by making all optional in schema, then validate in function - allows flexible mixed usage while detecting conflicts

## Environment & Dependencies
_Runtime quirks, version sensitivities, configuration gotchas_


## Mistakes to Avoid
_Failed approaches, time sinks, what NOT to do (saves future sessions from repeating)_

[💀🔧] When adding new optional parameters to commands, MUST update BOTH locations: the discriminated union schema AND the unified tool inputSchema - missing from inputSchema causes parameters to be silently dropped, leading to incorrect behavior
[💀🔧] Empty string split gives [''] not [] - when handling empty files with line operations, must check `content === '' ? [] : content.split('\n')` to avoid off-by-one errors in append logic


## Proven Solutions
_Patterns that work, reliable approaches, validated fixes_

[✅🔧] Use .shape property to extract raw Zod schema for MCP SDK inputSchema (SDK expects ZodRawShape not ZodObject)
[✅] Type assertions needed when constructing command objects from parsed params: `{ command: 'view', ...parsed } as operations.ViewCommand`
Individual tool schemas omit command field since tool name implies command
[✅💡] Parameter combination analysis methodology: Create full combinatorial matrix (every command × every parameter), categorize each as useful/questionable/confusing/nonsensical, identify patterns, design behaviors, create self-contained implementation spec
Resulted in 4 approved features: create empty file, insert append, delete matching text, document str_replace deletion
[✅🔒] Unique text requirement for safety: Both str_replace AND delete with old_str should require text appears exactly once - prevents accidental mass deletions/modifications
Fail fast with clear error message showing occurrence count when text appears multiple times
[✅🔧] Regex special character escaping: When searching for literal text (not patterns), use escapeRegExp helper: `text.replace(/[.*+?^${}()|[\]\\]/g, '\\## Proven Solutions
_Patterns that work, reliable approaches, validated fixes_

[✅🔧] Use .shape property to extract raw Zod schema for MCP SDK inputSchema (SDK expects ZodRawShape not ZodObject)
[✅] Type assertions needed when constructing command objects from parsed params: `{ command: 'view', ...parsed } as operations.ViewCommand`
Individual tool schemas omit command field since tool name implies command
[✅💡] Parameter combination analysis methodology: Create full combinatorial matrix (every command × every parameter), categorize each as useful/questionable/confusing/nonsensical, identify patterns, design behaviors, create self-contained implementation spec
Resulted in 4 approved features: create empty file, insert append, delete matching text, document str_replace deletion')`
Critical for handling text containing $, ., *, +, ?, etc.
[✅🎯] Optional parameter defaults in Zod: `.default('')` for create file_text, `.optional()` for insert insert_line
Handler uses nullish coalescing: `const content = command.file_text ?? ''`


## Testing & Debugging
_Test strategies that work, debugging approaches, tools that help_

[🔧✨] MCP-Debug tool excellent for live testing MCP servers: connect, initialize, list tools, call tools
Can test different CLI flags by reconnecting with different args array
[⚠️🧪] MCP-Debug is useful for quick iteration BUT always verify with actual Claude Code MCP client before marking complete
Different MCP client implementations may handle schemas/parameters differently
[✅🧪] Parameter combination testing strategy: Test BOTH tool modes (unified + one-tool-per-command), test with and without optional params, test error cases (multiple occurrences, parameter conflicts)
[✅🎨] Empty content UX messaging implemented (Session e0d8aaf1, Oct 28 2025):
Empty file view returns "Memory file is empty.", empty directory returns "Directory is empty." (both tree and simple modes), empty file creation returns "Created empty memory file."
Implementation: operations.ts (viewFile L194-197, viewDirectory L183-186, create L286-288) + tree-view.ts (renderDirectoryTree L273-276)
Tested: 117 unit tests + MCP-Debug integration (all pass)


## Deferred Work
_Complex tasks or investigations postponed for future sessions_


## Project-Specific Knowledge
_Unique aspects of this particular codebase/project_

[📖💡] Parameter combinations quick reference added to README.md (table format) and src/memory/operations.ts (code comments)
Shows required vs optional parameters for all 6 commands with usage notes
Highlights flexible usage: create empty file, append to end, delete text with str_replace, etc.
Key location: README line 140-151, operations.ts line 14-43

[📊✅] Parameter combinations feature COMPLETE (Sessions f7051c6f + ff6a2f81, Oct 28 2025):
4 features fully implemented AND tested: create empty file, insert append, delete unique text, str_replace deletion
Comprehensive testing: 30+ test cases, both tool modes, edge cases, error conditions - ALL PASS
Safety validated: Both delete and str_replace require unique text (fail fast if multiple occurrences)
114 tests passing, fully documented in README/CHANGELOG/TEST-FINDINGS.md
Commits: bbd1a83 (docs) + 4950c2e (implementation)
Status: Ready for production 🚀
[🏗️💡] Dual schema locations for parameter changes: When adding optional params, update BOTH MemoryCommandSchema (21-64) AND individual command schemas (76-119) in src/server/mcp-server.ts
Also update TypeScript interfaces in src/memory/operations.ts
[🔧💡] Empty file handling pattern: Check `content === ''` before splitting to avoid [''] array
Append logic: `insertLine = lines.length + 1` works for both empty and non-empty files when using empty array for empty content


---
Note: These are starter sections - add new sections as your understanding evolves!