# CLAUDE.md - Instructions for Claude AI

## Project Overview
- Memory systems for LLMs with various retrieval strategies
- Uses uv for package management and virtual environments
- Source code in `/src`, examples in `/src/examples`

## Development Guidelines

### Package Management
- Always use `uv` for package installation (NOT pip)
- Package dependencies already in requirements.txt
- Run with `uv run -m src.examples.[filename]` (no .py extension)

### Code Exploration
- Examine existing code thoroughly before implementing solutions
- Start with `git ls-files` to understand project structure
- Review existing tests to understand component usage
- Use existing examples/test files when available

### Implementation Process
- Use ask_human_tool to consult before making significant changes
- Implement changes incrementally with testing at each step
- Comment code appropriately for clarity

### Testing
- Use existing test scripts when available
- Main test file: `src/examples/memory_system_test.py`
- Test modules independently as needed

## Technical Notes

### Known Issues
- BashTool does not inherit environment variables by default - use `.env` file for API keys and configuration instead
- BashTool incorrectly flags 'git rm' as a banned command
