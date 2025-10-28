# Long-Term Memory (Persistent Knowledge)

## Architecture & Design
_How the system actually works vs how it was intended to work_

[🏗️💡] MCP server supports dual tool exposure modes: unified tool with command parameter (default) vs separate tools per command
Conditional registration in createMemoryServer() controlled by oneToolPerCommand boolean flag
Both modes use identical underlying operations - only tool registration differs


## Performance & Optimization
_Measured performance characteristics, bottlenecks, successful optimizations_


## Environment & Dependencies
_Runtime quirks, version sensitivities, configuration gotchas_


## Mistakes to Avoid
_Failed approaches, time sinks, what NOT to do (saves future sessions from repeating)_


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


## Project-Specific Knowledge
_Unique aspects of this particular codebase/project_


---
Note: These are starter sections - add new sections as your understanding evolves!