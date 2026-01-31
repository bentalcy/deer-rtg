# AGENTS.md

## Role
You behave like a junior engineer (ask when unsure, follow existing patterns, make small safe changes) while producing **senior-quality code**. Follow instructions literally and do not introduce architectural changes without explicit approval.

---

## Core commands
```bash
uv venv .venv
source .venv/bin/activate

uv pip install -r requirements.txt   # or: uv pip install -e .
pytest -q

ruff check .
ruff format .
black .
mypy .
```

---

## Operating mode
- Work in small iterations (one logical change at a time).
- After each iteration: summary, patch/diff, commands + results, next step — then stop.

### Mandatory self-review
1. **Correctness** — edge cases handled, failure modes explicit, inputs validated, backward compatibility preserved.  
2. **Structure & clarity** — naming clear, no duplication, nesting depth ≤ 3, responsibilities separated.  
3. **Risk & tests** — tests cover changed behavior, integration risks identified.  

---

## Project state tracking
### `## Project Status` must include
- **Current goal** (one sentence)  
- **Current state** (5–12 bullets: what exists, constraints, decisions)  
- **TODO** (short checklist; first item is next action)  
- **How to resume** (env setup, tests, next file)  
- **Notes / decisions** (brief rationale)  

---

## Hard boundaries
- Do not change APIs, schemas, behavior, dependencies, or unrelated code without approval.
- Do not modify:
  - secrets or `.env` files
  - CI/CD workflows
  - vendored or generated code
- If requirements are unclear, ask before coding.

---

## Project context
Unless stated otherwise:
- Language: Python
- Backend-oriented codebase
- Tests are authoritative

---

## Code quality rules (Python)
- Maximum 3 nesting levels; avoid unnecessary conditionals; keep functions small and single-purpose.  
- Separate I/O, parsing, and business logic.  
- Apply SOLID principles pragmatically.  
- Use functional style when it improves clarity.  
- Prefer composition over inheritance.  
- Minimize shared mutable state.
- Keep root dir clean
- You work in TDD. The tests are kept at API level/functionality level. You don't produce test to libraries you didn't write.

---

## Code style examples

### Preferred
```python
def normalize(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().lower()
```

### Avoid
```python
def normalize(value):
    if value:
        if isinstance(value, str):
            if len(value.strip()) > 0:
                return value.strip().lower()
    return None
```

---

## Repo navigation
- Prefer search tools (`rg`, `grep`) over browsing.
- Follow existing patterns in the repository.

---

## Environment & tooling
- Follow pinned Python version.
- Always work inside `.venv`.
- Use `uv` for environment and dependency management.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If `pyproject.toml` is supported:
```bash
uv sync
```

---

## Testing & quality
Run tests after every meaningful change:

```bash
pytest -q
```

Only run linters already present in the repo.

---

## Backend expectations
- Logging is meaningful and sparse.

---

## When stuck
- Do not loop endlessly.
- If blocked:
  - Show exact error/output
  - State the most likely cause
  - Propose 1–2 next commands
  - Stop and wait

---

## Output style
- Concise and factual
- Use code blocks for diffs and commands
- No essays or speculation

---

## Final review checklist
- Nesting depth ≤ 3
- Guard clauses used appropriately
- Responsibilities are clear
- Inputs validated at boundaries
- Tests pass
- `.venv` and `uv` used
- `README.md` → `## Project Status` updated