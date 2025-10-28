# Long-Term Memory (Persistent Knowledge)

## Architecture & Design
_How the system actually works vs how it was intended to work_

Memory tool uses discriminated union schema with low-level Server API (not McpServer.registerTool)
Zod discriminated union converted to JSON Schema using zodToJsonSchema with $refStrategy: "none" to inline the schema
MCP protocol requires type: "object" at root level - added manually after schema generation
Command executor provides type-safe dispatch with automatic type narrowing per command variant
Each command variant has exact required fields in JSON Schema (no optional pollution)


## Performance & Optimization
_Measured performance characteristics, bottlenecks, successful optimizations_


## Environment & Dependencies
_Runtime quirks, version sensitivities, configuration gotchas_


## Mistakes to Avoid
_Failed approaches, time sinks, what NOT to do (saves future sessions from repeating)_

Don't use McpServer.registerTool for discriminated unions - it always wraps inputSchema in z.object() making top-level unions impossible
Don't use zodToJsonSchema with default settings - produces $ref at top level which some validators reject when mixed with type: "object"
Don't use z.refine() on discriminated union schemas - breaks the discriminated union structure for Zod's discriminatedUnion type
Don't use .passthrough() for flexible parameter naming - security risk (accepts ANY fields) and requires type casts
Don't flatten discriminated union with multiple variants of same command - allows mixing conventions in single call (e.g., {old_str, new_string})
Never skip PR review for "working" code - Session 0bb07c86 found 6 critical issues in fully functional E2E-tested implementation
Terminology: old_string is NOT camelCase (uses underscores) - true camelCase would be oldString


## Proven Solutions
_Patterns that work, reliable approaches, validated fixes_

Discriminated union pattern: Use Server class with manual setRequestHandler for ListToolsRequestSchema and CallToolRequestSchema
Schema conversion: zodToJsonSchema(schema, { $refStrategy: "none", strictUnions: true }) then add type: "object"
UX flexibility: Accept union of types (number | string) in schema then normalize in command executor
Parameter naming flexibility: Use union of schemas (SnakeCase | CamelCase) NOT .passthrough() - maintains type safety while accepting both conventions
GPT-5 consultation effective for complex architectural decisions - session 1761595236443-75uvjxnu provided critical $refStrategy insight
Multi-agent PR review highly effective: code-reviewer, type-design-analyzer, silent-failure-hunter, comment-analyzer, pr-test-analyzer each found unique issues


## Testing & Debugging
_Test strategies that work, debugging approaches, tools that help_


## Deferred Work
_Complex tasks or investigations postponed for future sessions_

[URGENT] str_replace schema fix - requires GPT-5 consultation before implementation (Session 280f8ab4)
Challenge: Accept EITHER {old_str, new_str} OR {old_string, new_string} but prevent mixing
Flawed approaches: .passthrough() (security risk), flattened union (allows mixing)
Need: Proper Zod schema that enforces exactly one style per call within discriminated union


## Project-Specific Knowledge
_Unique aspects of this particular codebase/project_


---
Note: These are starter sections - add new sections as your understanding evolves!