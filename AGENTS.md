
---

# AGENTS.md

## Multi-Agent Continuity

This project is worked on by multiple agents across sessions — sometimes at different capability levels. You may be picking up mid-task from a different agent, or handing off to one. Design your work accordingly.

**Rules:**
- **Always read `AGENT_HANDOFF.md` first.** It tells you the current goal, what is done, and the exact next step.
- **Update `AGENT_HANDOFF.md` at the end of every session**, and whenever you hit a stopping point or hand off. Write it as if the reader has zero memory of your session.
- **Update it mid-session** any time you make a meaningful decision or complete a discrete step — don't wait for the end.
- Be specific: include exact file paths, function names, decisions made, and the next command to run.
- If you're switching from a high-capability model (Opus) to a lower one (Sonnet), make sure `AGENT_HANDOFF.md` contains enough detail that no judgment calls are needed by the next agent.
- If you're Sonnet and hit a decision not covered in `AGENT_HANDOFF.md` or `docs/PROJECT_GUIDE.md`, **stop and tell the user** — do not invent a solution.

## Role
- Behave like a junior engineer (ask when unsure, follow existing patterns, make small safe changes) while producing **senior-quality code**.
- Follow instructions literally.
- Do not introduce architectural changes without explicit approval.

## Non-negotiable principles
- Be concise by default. Prefer bullets and commands over prose.
- Make **one change at a time**. Keep diffs small and reviewable.
- Minimize blast radius. Touch the fewest files needed.
- Preserve contracts. **Do not break public APIs** (HTTP/JSON, gRPC, CLI, config, DB schema, events).
- Prefer boring solutions. Avoid cleverness and new dependencies.
- Test early and repeatedly. Optimize for fast feedback loops.
- Separate refactors from behavior changes.
- Treat ambiguity/risk as a stop signal. Pause and ask.
- Always have a rollback story.

## Hard boundaries
- Do not change APIs, schemas, behavior, dependencies, or unrelated code without approval.
- Do not modify:
  - secrets or `.env` files
  - CI/CD workflows
  - vendored or generated code
- If requirements are unclear, ask before coding.

## Project context
Unless stated otherwise:
- Language: Python
- Backend-oriented codebase
- Tests are authoritative

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
````

## Environment & tooling

* Follow pinned Python version.
* Always work inside `.venv`.
* Use `uv` for environment and dependency management.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If `pyproject.toml` is supported:

```bash
uv sync
```

## Default workflow (Plan → Implement → Test repeatedly → Summarize)

1. **Plan (before editing)**

   * State intent in 1–3 bullets.
   * Identify impacted contracts (API/CLI/config/DB/etc.).
   * Define success criteria and what you will test.
   * Choose the smallest possible change.

2. **Implement**

   * Make the minimal diff that achieves the plan.
   * Keep changes localized; avoid drive-by edits.
   * Follow existing patterns in the repository.
   * If you must touch a contract, use the compatibility rules below.

3. **Test repeatedly**

   * Run fast checks first, then broader suites.
   * Re-run relevant tests after each meaningful edit.
   * Prefer deterministic tests; fix flakiness instead of retrying blindly.

4. **Summarize (then stop)**

   * What changed (bullets).
   * Patch/diff.
   * How it was tested (exact commands + results).
   * Compatibility notes (contracts, migrations, flags, deprecations).
   * Rollback plan.
   * Next step (1 item) — then stop.

## Mandatory self-review

1. **Correctness**

   * Edge cases handled.
   * Failure modes explicit.
   * Inputs validated at boundaries.
   * Backward compatibility preserved.
2. **Structure & clarity**

   * Naming clear.
   * No duplication.
   * Nesting depth ≤ 3.
   * Responsibilities separated.
3. **Risk & tests**

   * Tests cover changed behavior.
   * Integration risks identified.
   * Rollback path stated.

## API and compatibility rules

### General

* Do **not** change behavior of existing endpoints/fields/flags/commands without a compatibility plan.
* Do **not** remove or rename publicly documented fields/flags/metrics/events.
* Additive changes are the default: new fields, new endpoints, new optional args.

### HTTP/JSON

* New request fields must be optional.
* New response fields must be additive.
* Do not change meaning/type of existing fields.
* Preserve status codes, error shapes, and pagination semantics.
* Behavior changes must be gated:

  * Feature flag, or
  * Versioned endpoint, plus deprecation timeline.

### CLI

* Preserve command names, flags, output formats, and exit codes.
* Add new flags as optional with safe defaults.
* Treat machine-consumed output as a contract.

### Config

* Config files/env vars are contracts.
* Add new config keys with defaults.
* Do not change meaning of existing keys.
* Validate strictly but avoid breaking existing valid configs.


## Testing & quality (determinism + repeatability)

* Prefer deterministic tests:

  * Control time (fake clocks), randomness (fixed seeds), and external IO.
  * Avoid network calls in unit tests; use fakes/stubs.
  * Avoid sleeps; use polling with timeouts only when unavoidable.
* Treat flaky tests as a bug:

  * Do not paper over with retries unless it’s a known transient integration boundary.
  * If quarantined, require an issue + owner + deadline.

### Default local loop

Run after every meaningful change:

```bash
pytest -q
```

Only run linters/tools already present in the repo.

### Commands

```bash
pytest -q          # run all tests
ruff check .       # lint
ruff format .      # format
mypy .             # type check
```

## Refactoring policy (separate from behavior changes)

* Refactors must not change externally observable behavior.
* Do refactor-only changes when possible:

  * No logic changes.
  * No output changes.
  * No contract changes.
* If refactor is needed to enable behavior change:

  * Do it as two steps (refactor first, behavior second) unless explicitly approved.

## Dependency policy (default: don’t add)

* Default position: **do not add new dependencies**.
* If truly necessary:

  * Justify why existing deps can’t do it.
  * Prefer standard library / built-in tooling.
  * Evaluate license, maintenance health, security history, transitive deps, footprint.
  * Document the reason and pin versions per ecosystem norms.

## Code quality rules (Python)

* Nesting depth ≤ 3; prefer guard clauses.
* Keep functions small and single-purpose.
* Separate I/O, parsing, and business logic.
* Apply SOLID pragmatically.
* Prefer composition over inheritance.
* Minimize shared mutable state.
* Keep root directory clean.
* Work in TDD where feasible:

  * Tests at API/behavior level.
  * Do not write tests for third-party libraries you didn’t write.

## Backend expectations (logging / security)

* Logging:

  * Meaningful and sparse (actionable, not noisy).
  * Include stable identifiers (request id, job id, resource id) where relevant.
  * Avoid high-cardinality metrics/labels.
* Security:

  * Never log secrets (tokens, passwords, private keys, session cookies).
  * Redact sensitive fields by default.
  * Validate external inputs.
  * Use least privilege for access changes.

## Repo navigation

* Prefer search (`rg`, `grep`) over browsing.
* Follow existing patterns in the repository.

## Stop conditions (“pause and ask”)

Pause and ask for clarification/approval when:

* Contract impact is unclear (API/CLI/config/DB/event/output format).
* A breaking change might be required.
* A new dependency or major upgrade is needed.
* You cannot reproduce the issue or define a deterministic test.
* The change requires broad refactors across multiple modules.
* Performance, security, or data-loss risk seems plausible.
* Requirements conflict or acceptance criteria are missing.
* Rollback strategy is unclear.
* You are stuck/looping.

## When stuck

* Do not loop endlessly.
* If blocked:

  * Show exact error/output.
  * State the most likely cause.
  * Propose 1–2 next commands.
  * Stop and wait.

## Output style requirements for agents

* Concise and factual.
* Use code blocks for diffs and commands.
* Prefer patches/diffs over whole-file rewrites.
* Make one change per patch unless explicitly requested otherwise.
* No essays or speculation.
* Always include:

  * What changed
  * Why
  * How tested (commands + results)
  * Compatibility/rollout notes
  * Rollback steps

## Project state tracking

### `README.md` → `## Project Status` must include

* **Current goal** (one sentence)
* **Current state** (5–12 bullets: what exists, constraints, decisions)
* **TODO** (short checklist; first item is next action)
* **How to resume** (env setup, tests, next file)
* **Notes / decisions** (brief rationale)

## Definition of Done (DoD) + rollback mindset

A change is done only when:

* Scope is minimal and matches the plan.
* Relevant tests pass locally (record commands):

  * `pytest -q` (minimum)
  * plus any repo-specific suites affected
* Contracts preserved or properly versioned/flagged with a deprecation plan.
* Observability updated if behavior changes (logs/metrics/traces as appropriate).
* Docs updated if usage/behavior changed.
* Rollback plan exists and is safe:

  * Feature flag off path works (if used).
  * Migration rollback steps defined (or forward-only explicitly approved).
  * Previous version can run without data corruption.

### Rollback guidance (minimum)

* Prefer reversible rollouts:

  * Feature flag to disable new behavior.
  * Keep old code paths until stable.
* For DB changes:

  * Ensure rollback does not require data loss.
  * If forward-only, state it explicitly and provide mitigation steps.

## Examples: good vs bad changes (blast-radius thinking)

**Good**

* Add a new optional API field with defaults; behavior unchanged; update tests + docs.
* Introduce a feature flag for a behavior change; ship dark; observe; then enable gradually.
* Refactor internals in a dedicated change with no output/contract changes.

**Bad**

* Rename a JSON field or CLI flag “for consistency” without versioning/deprecation.
* Mix refactor + behavior change + dependency addition in one diff.
* Apply schema changes that force coordinated deploy of multiple services.

## Final review checklist

* [ ] One logical change; minimal diff; no drive-by edits.
* [ ] Nesting depth ≤ 3; guard clauses used appropriately.
* [ ] Responsibilities are clear; I/O separated from logic.
* [ ] Inputs validated at boundaries; failure modes explicit.
* [ ] No contract break (API/CLI/config/DB/output); or properly versioned/flagged.
* [ ] No new dependencies (or justified and approved).
* [ ] No secrets/logging of sensitive data.
* [ ] Tests pass; commands + results recorded.
* [ ] `README.md` → `## Project Status` updated.
* [ ] Rollback plan stated.

## Agent checklist (quick)

* [ ] Plan in 1–3 bullets.
* [ ] Implement smallest change.
* [ ] Test repeatedly.
* [ ] Self-review.
* [ ] Summarize + patch + commands + rollback.
* [ ] Stop.

