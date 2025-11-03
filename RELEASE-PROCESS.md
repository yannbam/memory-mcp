# Release Process for memory-mcp

This document describes the step-by-step process for creating a new release.

## Prerequisites

Before starting a release, ensure:
- [ ] All tests pass: `npm test` (166/166 tests)
- [ ] Linting is clean: `npm run lint`
- [ ] Coverage meets requirements: `npm run test:coverage` (≥80%, ideally 93%+)
- [ ] All planned features are complete and merged to `dev`
- [ ] CHANGELOG.md is updated with all changes in `[Unreleased]` section
- [ ] Documentation is accurate and up-to-date

## Release Workflow

### 1. Finalize CHANGELOG

**On `dev` branch:**

```bash
# Ensure you're on dev and up-to-date
git checkout dev
git pull origin dev

# Edit CHANGELOG.md
# Move [Unreleased] content to new [X.Y.Z] section with today's date
# Format: ## [X.Y.Z] - YYYY-MM-DD
# Leave [Unreleased] section empty for future changes
```

**Commit the CHANGELOG:**

```bash
git add CHANGELOG.md
git commit -m "chore: finalize CHANGELOG for v0.1.0"
git push origin dev
```

### 2. Merge dev → main

**Switch to main and merge:**

```bash
git checkout main
git pull origin main
git merge dev --no-ff
```

The `--no-ff` flag ensures a merge commit is created, preserving the branch history.

**Push to origin:**

```bash
git push origin main
```

### 3. Create Annotated Git Tag

**On `main` branch, create annotated tag:**

```bash
git tag -a v0.1.0 -m "Release v0.1.0

First public release of memory-mcp - MCP implementation of Claude's Memory tool.

Key features:
- All 6 memory commands (view, create, str_replace, insert, delete, rename)
- SHA-256 checksum-based concurrency detection
- Reader-writer locks (38x performance improvement)
- Tree view mode with comprehensive file information
- 166 tests with 93%+ coverage
- Extensive documentation and security hardening
"
```

**Push the tag:**

```bash
git push origin v0.1.0
```

### 4. Create GitHub Release (Recommended)

**Option A: Via GitHub Web UI**

1. Navigate to: https://github.com/yannbam/memory-mcp/releases/new
2. Select tag: `v0.1.0`
3. Release title: `v0.1.0 - First Public Release`
4. Description: Copy from CHANGELOG.md [0.1.0] section
5. Check "Set as the latest release"
6. Click "Publish release"

**Option B: Via GitHub CLI**

```bash
# Install gh CLI if not already installed: https://cli.github.com/

gh release create v0.1.0 \
  --title "v0.1.0 - First Public Release" \
  --notes-file <(sed -n '/## \[0.1.0\]/,/## \[/p' CHANGELOG.md | sed '$d')
```

### 5. Post-Release Version Bump

**On `dev` branch, bump version for next development cycle:**

```bash
git checkout dev
git pull origin dev

# Update package.json version to 0.2.0-dev (or next planned version)
npm version 0.2.0-dev --no-git-tag-version

# Commit the version bump
git add package.json
git commit -m "chore: bump version to 0.2.0-dev"
git push origin dev
```

### 6. Verify Release

**Checklist:**

- [ ] Tag exists: `git tag -l v0.1.0`
- [ ] Tag is on GitHub: https://github.com/yannbam/memory-mcp/tags
- [ ] GitHub Release is published: https://github.com/yannbam/memory-mcp/releases
- [ ] CHANGELOG.md link works: `[0.1.0]: https://github.com/yannbam/memory-mcp/releases/tag/v0.1.0`
- [ ] `main` branch is at the release commit
- [ ] `dev` branch is ahead of `main` with version bump

## Semantic Versioning Guidelines

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** (X.0.0): Incompatible API changes or breaking changes
- **MINOR** (0.X.0): New features, backward-compatible
- **PATCH** (0.0.X): Bug fixes, backward-compatible

**Version Increment Examples:**

- Adding new memory command → MINOR
- Changing existing command behavior (breaking) → MAJOR
- Fixing bug in path validation → PATCH
- Improving error messages → PATCH or MINOR
- Changing MCP protocol version (breaking) → MAJOR

## Emergency Hotfix Process

**For critical bugs in production (main branch):**

1. Create hotfix branch from `main`: `git checkout -b hotfix/critical-fix main`
2. Make the fix and test thoroughly
3. Update CHANGELOG.md with new PATCH version
4. Merge to `main`: `git checkout main && git merge hotfix/critical-fix --no-ff`
5. Tag with new version: `git tag -a v0.1.1 -m "Hotfix: description"`
6. Push: `git push origin main && git push origin v0.1.1`
7. Create GitHub release
8. Merge back to `dev`: `git checkout dev && git merge main`
9. Delete hotfix branch: `git branch -d hotfix/critical-fix`

## Notes

- **Always use annotated tags** (`-a` flag) for releases - they contain metadata
- **Tag on `main` branch** after merging from `dev`
- **Never force push** to `main` or tag commits
- **Keep `dev` as primary development branch** - all features merge to `dev` first
- **`main` = stable releases only** - production-ready code
- **Version prefix**: Always use `v` prefix for tags (e.g., `v0.1.0`, not `0.1.0`)
