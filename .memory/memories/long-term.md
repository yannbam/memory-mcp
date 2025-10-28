# Long-Term Memory (Persistent Knowledge)

## Architecture & Design
_How the system actually works vs how it was intended to work_

[🏗️💡] MCP server supports dual tool exposure modes: unified tool with command parameter (default) vs separate tools per command
Conditional registration in createMemoryServer() controlled by oneToolPerCommand boolean flag
Both modes use identical underlying operations - only tool registration differs
[🧪💡] Unified tool description is intentionally minimal/commented out - experiment to test HOW Claude uses commands/parameters intuitively without detailed manual
This is NOT incomplete - it's deliberate UX testing for future session analysis


## Performance & Optimization
_Measured performance characteristics, bottlenecks, successful optimizations_

[⚠️🔧] Zod .refine() creates ZodEffects wrapper which breaks discriminated unions and .shape property access - for discriminated unions, move validation logic into the handler function instead of schema refinement
[✅🎯] Forgiving parameter naming implementation: Accept both old_str/new_str AND old_string/new_string by making all optional in schema, then validate in function - allows flexible mixed usage while detecting conflicts

## Environment & Dependencies
_Runtime quirks, version sensitivities, configuration gotchas_


## Mistakes to Avoid
_Failed approaches, time sinks, what NOT to do (saves future sessions from repeating)_

[💀🔧] When adding new optional parameters to commands, MUST update BOTH locations: the discriminated union schema AND the unified tool inputSchema - missing from inputSchema causes parameters to be silently dropped, leading to incorrect behavior


## Proven Solutions
_Patterns that work, reliable approaches, validated fixes_

[✅🔧] Use .shape property to extract raw Zod schema for MCP SDK inputSchema (SDK expects ZodRawShape not ZodObject)
[✅] Type assertions needed when constructing command objects from parsed params: `{ command: 'view', ...parsed } as operations.ViewCommand`
Individual tool schemas omit command field since tool name implies command


## Testing & Debugging
_Test strategies that work, debugging approaches, tools that help_

[🔧✨] MCP-Debug tool excellent for live testing MCP servers: connect, initialize, list tools, call tools
Can test different CLI flags by reconnecting with different args array


## Deferred Work
_Complex tasks or investigations postponed for future sessions_

[🔄💡] Implement forgiving parameter naming: accept both "old_str"/"new_str" AND "old_string"/"new_string" with optional parameters
- Allow mixed usage (old_str + new_string) but fail early if both variants provided for same param (old_str + old_string)
- Schema validation should catch conflicts before operation execution

[🔄🔧] Add "delete_line" parameter to delete command for targeted single-line deletion
- Alternative to reading file, removing line, writing back - more efficient and atomic

[🔄🎯] Create combinatorial matrix of ALL command+parameter combinations and implement sensible ones
- Example: delete with path+old_str => replace old_str with "" (delete matching text)
- Example: str_replace with path+old_str => replace old_str with "" (delete matching text)
- Analyze which combinations provide useful functionality vs which are nonsensical


## Project-Specific Knowledge
_Unique aspects of this particular codebase/project_


---
Note: These are starter sections - add new sections as your understanding evolves!