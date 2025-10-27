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


## Proven Solutions
_Patterns that work, reliable approaches, validated fixes_

Discriminated union pattern: Use Server class with manual setRequestHandler for ListToolsRequestSchema and CallToolRequestSchema
Schema conversion: zodToJsonSchema(schema, { $refStrategy: "none", strictUnions: true }) then add type: "object"
UX flexibility: Accept union of types (number | string) in schema then normalize in command executor
Parameter naming: Accept both snake_case and camelCase variants using .passthrough() and normalize in executor
GPT-5 consultation effective for complex architectural decisions - session 1761595236443-75uvjxnu provided critical $refStrategy insight


## Testing & Debugging
_Test strategies that work, debugging approaches, tools that help_


## Deferred Work
_Complex tasks or investigations postponed for future sessions_


## Project-Specific Knowledge
_Unique aspects of this particular codebase/project_


---
Note: These are starter sections - add new sections as your understanding evolves!