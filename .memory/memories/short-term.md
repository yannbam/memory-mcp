# Short-Term Memory (Session Context)

## Current Session
_Session ID, active branch, context usage_

Session: 280f8ab4-2fb0-4993-9662-eb8b9774c30e
Branch: feature/discriminated-union-schema
Context: ~100k tokens (at handoff target)
Status: ⚠️ Phase 1 attempted - CRITICAL FLAW DISCOVERED - need GPT-5 consultation

## Session Handoff
_What was done, what's next, blockers_

### This Session Accomplished
✅ Applied valid Phase 1 fixes (5 of 8 tasks):
  - Removed unused import
  - Fixed unnecessary async  
  - Added server-side error logging
  - Fixed transport cleanup error handling
  - Improved HTTP error responses with details
⚠️ DEFERRED str_replace schema fix - initial solution was critically flawed
✅ Fixed terminology: "camelCase" → "old_string/new_string notation"
✅ Updated pr-review-fixes plan with GPT-5 consultation requirement

### PR Review Findings Summary

**Critical Issues** (6 found):
- Type safety: 3x `as any` casts bypass TypeScript (11 linting errors)
- Schema: .passthrough() allows any fields (security issue)
- Error handling: Command errors not logged server-side
- Architecture: Type duplication between schemas and operations

**Test Coverage Gap**:
- New validation layer has ZERO tests (schema validation, command executor)
- Need ~55 new tests for production confidence
- Existing 85 tests cover operations layer perfectly

**Linting**: 14 errors (11 type safety, 3 misc)

**Verdict**: NOT ready for merge - needs Phase 1 fixes minimum

### What Next Session Should Do

**CRITICAL FIRST STEP: Consult GPT-5 about str_replace schema** (30 min)

**Problem**: Need Zod schema that:
- Accepts EITHER {old_str, new_str} OR {old_string, new_string}
- PREVENTS mixing (e.g., {old_str, new_string} must be invalid)
- Works within discriminated union (MemoryCommandSchema)
- Eliminates need for `as any` casts

**Flawed approaches tried**:
1. `.passthrough()` - allows ANY fields (security risk)
2. Flattening union - allows mixing conventions

**Potential solution to explore with GPT-5**:
- Union type for property itself?
- Custom Zod refinement?
- Conditional schema based on which fields are present?

**GPT-5 Consultation Format**:
```
Context: MCP server with discriminated union schema for memory commands
Environment: Zod v3.23.8, TypeScript, strict type safety required
Challenge: str_replace command accepts two parameter naming conventions
- Style A: {old_str, new_str} 
- Style B: {old_string, new_string}
- INVALID: {old_str, new_string} or {old_string, new_str}

Current approach (.passthrough()) allows any fields - security issue
Need: Zod schema that enforces exactly one style per call
Must work in: z.discriminatedUnion('command', [...])
```

**After schema solution found**:
1. Implement fix in src/memory/schemas.ts
2. Remove `as any` casts from src/memory/command-executor.ts
3. Run lint, tests, build
4. Complete Phase 1 remaining tasks (see pr-review-fixes plan)

**Then consider**: Phase 2 tests (recommended), Phase 3 docs, Phase 4 polish

### Current Blockers
⚠️ str_replace schema fix requires GPT-5 consultation (see "What Next Session Should Do")
Otherwise: clear path forward documented

## Active Plans
_Current PlanAndTrack references_

Plan: pr-review-fixes (0% complete - 0/30 tasks) ⚡ ACTIVE - implement Phase 1 next session
Plan: memory-system-implementation (75% complete - 12/16 tasks) - paused
Plan: public-release-beta (3% complete - 1/39 tasks) - resume after PR merge

## Quick Notes
_Rapid capture space - add memories here during work without categorization_

⚠️ Terminology error corrected: old_string/new_string is NOT camelCase (uses underscores)
True camelCase would be: oldString, newString
Current implementation uses .passthrough() - allows {old_str, new_string} which is INVALID
Flattening discriminated union to include both variants separately also flawed - still allows mixing
Need proper Zod schema approach from GPT-5 consultation