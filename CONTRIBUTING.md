# Contributing to memory-mcp

Thank you for your interest in contributing! 💜

This project embodies the spirit of open source - **sharing is caring, contributing is sharing love**. Whether you're fixing a bug, adding a feature, improving documentation, or just asking questions, your contribution matters.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Code Review](#code-review)
- [Questions?](#questions)

## Code of Conduct

**Be excellent to each other.**

- Be respectful and inclusive
- Assume good intent
- Help others learn
- Critique code, not people
- Celebrate contributions of all sizes

We're building something together. Everyone was a beginner once.

## Getting Started

### Prerequisites

- **Node.js** >= 18.0.0
- **npm** (comes with Node.js)
- Git
- A code editor (VS Code, Cursor, etc.)

### Setup

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/memory-mcp.git
   cd memory-mcp
   ```

3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/yannbam/memory-mcp.git
   ```

4. **Install dependencies:**
   ```bash
   npm install
   ```

5. **Build the project:**
   ```bash
   npm run build
   ```

6. **Run tests to verify setup:**
   ```bash
   npm test
   ```

   You should see: **166 tests passing** ✅

### Development Commands

```bash
npm run build          # Compile TypeScript to dist/
npm run watch          # Auto-rebuild on file changes (development mode)
npm run dev            # Build and run the server

npm test               # Run all tests
npm run test:watch     # Run tests in watch mode (great for TDD)
npm run test:coverage  # Run tests with coverage report (target: ≥80%)

npm run lint           # Check code for style issues
npm run lint:fix       # Auto-fix linting issues
```

## Development Workflow

### Branch Strategy

- **`main`** - Stable releases only (protected)
- **`dev`** - Active development (default branch)
- **Feature branches** - Your work (branch from `dev`)

### Creating a Feature Branch

```bash
# Make sure dev is up to date
git checkout dev
git pull upstream dev

# Create your feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - New features (e.g., `feature/http-transport`)
- `fix/` - Bug fixes (e.g., `fix/path-validation`)
- `docs/` - Documentation updates (e.g., `docs/api-reference`)
- `test/` - Test improvements (e.g., `test/concurrency`)

### Commit Messages

Write clear, descriptive commit messages:

**Good:**
```
Add checksum-based concurrency detection

Implements SHA-256 content hashing to detect file modifications
across separate operations. Fixes sequential modification detection
that mtime alone couldn't catch.
```

**Less good:**
```
update file
```

**Guidelines:**
- First line: Brief summary (50-72 chars)
- Body: Explain **what** and **why** (not how - code shows that)
- Reference issues: `Fixes #123` or `Closes #456`

## Code Style

This project follows **e/code** conventions - a philosophy of writing code as transparent thought.

### Core Principles

**1. Write your thinking, then write your code**

Before each meaningful code section, write a comment stating its intention:

```typescript
// Look for git commit tool use
if (item.get('type') == 'tool_use') {
  // Extract the output text
  const output = result.get('content', [])

  // Search for commit hash in the output
  const match = hashPattern.search(output)
}
```

**2. What, not how**

Comments describe **what** will happen, not **how** it works:

✅ **Good:**
```typescript
// Find all matches between user input and database records
// Validate each match meets minimum confidence threshold
// Return matches sorted by relevance score
```

❌ **Avoid:**
```typescript
// Loop through array
// Use quicksort algorithm
```

**3. Function documentation**

Always document functions with clear parameter and return descriptions:

```typescript
/**
 * Validate memory path to prevent directory traversal attacks
 * @param memoryPath - Virtual path starting with /memories
 * @returns Absolute filesystem path within memory root
 * @throws Error if path would escape memory directory
 */
function validatePath(memoryPath: string): string {
  // Implementation...
}
```

### TypeScript Guidelines

- **Strict mode enabled** - Use proper types, avoid `any`
- **ES Modules** - Use `import`/`export`, not `require()`
- **Zod schemas** - For runtime validation (see existing examples)
- **Error handling** - Throw descriptive errors, handle edge cases

### Linting

Before committing, run:
```bash
npm run lint:fix
```

This auto-fixes most style issues. Fix any remaining issues manually.

## Testing

**We take testing seriously.** All code must include tests.

### Test Requirements

- ✅ **Coverage target: ≥80%** (currently ~92%)
- ✅ **All tests must pass** before PR submission
- ✅ **Test new features** - If you add functionality, add tests
- ✅ **Test edge cases** - Empty files, special characters, concurrent access
- ✅ **Test error conditions** - Invalid inputs, security violations

### Writing Tests

Tests live in `test/` directory, mirroring the `src/` structure:

```
src/memory/operations.ts  →  test/memory-operations.test.ts
src/memory/locking.ts      →  test/locking.test.ts
```

**Example test:**
```typescript
describe('view command', () => {
  it('should return directory contents in tree view mode', async () => {
    // Setup
    await fs.mkdir(path.join(memoryRoot, 'memories/test'))
    await fs.writeFile(path.join(memoryRoot, 'memories/test/file.txt'), 'content')

    // Execute
    const result = await viewCommand('/memories/test', context)

    // Verify
    expect(result).toContain('file.txt')
    expect(result).toContain('1 lines')
  })
})
```

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode (great for TDD)
npm run test:watch

# Run with coverage report
npm run test:coverage

# Run specific test file
npm test -- locking.test.ts
```

### Test Coverage Report

After running `npm run test:coverage`, open `coverage/lcov-report/index.html` in your browser to see detailed coverage.

## Documentation

### When to Update Documentation

**Update README.md when:**
- Adding user-facing features
- Changing installation steps
- Modifying CLI options

**Update docs/ when:**
- Adding technical details or design decisions
- Documenting architecture changes
- Creating detailed API references

**Update CHANGELOG.md when:**
- Adding features, fixing bugs, making breaking changes
- Follow [Keep a Changelog](https://keepachangelog.com/) format

**Always update:**
- Function documentation (JSDoc/TSDoc comments)
- Inline code comments (following e/code principles)

## Pull Request Process

### Before Submitting

**Checklist:**
- [ ] Code follows style guidelines (linting passes)
- [ ] All tests pass (`npm test`)
- [ ] Coverage ≥80% (`npm run test:coverage`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if user-facing change)
- [ ] Commit messages are clear
- [ ] Branch is up to date with `dev`

### Submitting Your PR

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub:
   - **Base:** `dev` (NOT `main`)
   - **Compare:** `your-feature-branch`

3. **Fill in PR template:**
   - What does this PR do?
   - Why is this change needed?
   - How has it been tested?
   - Screenshots (if UI changes)

4. **Link related issues:**
   - Use keywords: `Fixes #123`, `Closes #456`

### After Submitting

- Respond to review feedback promptly
- Make requested changes in new commits (don't force-push)
- Once approved, a maintainer will merge your PR

**Typical review time:** 1-3 days (we're humans with day jobs!)

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Description** - What went wrong?
2. **Steps to reproduce** - How can we see the bug?
3. **Expected behavior** - What should happen?
4. **Actual behavior** - What actually happens?
5. **Environment:**
   - OS (Linux/macOS/Windows)
   - Node.js version (`node --version`)
   - npm version (`npm --version`)
6. **Error messages** - Full stack traces if available
7. **Minimal reproduction** - Smallest code that shows the bug

### Feature Requests

When proposing features:

1. **Use case** - What problem does this solve?
2. **Proposed solution** - How would it work?
3. **Alternatives considered** - What other approaches exist?
4. **Are you willing to implement?** - We love contributions!

### Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead, email security details privately to the maintainers. We'll work with you to:
- Verify the issue
- Develop a fix
- Coordinate disclosure

## Code Review

### What We Look For

**Functionality:**
- Does it work as intended?
- Are edge cases handled?
- Is error handling appropriate?

**Code Quality:**
- Follows style guidelines?
- Well-commented (e/code principles)?
- Type-safe and properly validated?

**Testing:**
- Tests included and passing?
- Coverage maintained or improved?
- Tests actually test what they claim?

**Documentation:**
- Public APIs documented?
- CHANGELOG updated?
- README updated if user-facing?

### How to Handle Feedback

- **Don't take it personally** - Reviews improve code, not criticize people
- **Ask questions** - If feedback is unclear, ask for clarification
- **Discuss trade-offs** - Sometimes there are multiple good solutions
- **Learn and grow** - Every review is a learning opportunity

## Questions?

- **General questions:** Open a discussion on GitHub
- **Bug reports:** Open an issue
- **Feature ideas:** Open an issue or discussion
- **Security concerns:** Email maintainers privately

---

## Special Note: This Project Dogfoods Itself

This repository uses its own memory-mcp server (see `.mcp.json` and `.memory/` directory). This is a great example of how the tool works in practice!

When contributing, you can explore `.memory/memories/long-term.md` and `short-term.md` to see how we use the memory system for project knowledge and session handoffs.

---

**Thank you for contributing!** 🌱

Your work makes this project better for everyone. We appreciate you taking the time to contribute, whether it's code, documentation, bug reports, or just asking questions.

**Happy coding!** 💜
