<!-- a85f4c1f-05c8-4ea3-a919-b3b1fff0bffd 20d1db82-04f1-49ef-9682-ebc4e46e070b -->
# Ridges Generic Code Agent – Detailed Architecture & Implementation Plan

## Objectives

- Build a Cursor-like generic code agent tailored for Ridges’ sandbox.
- Operate under model whitelist and limited FS/network access.
- Dual-mode capability with clear category names (no external jargon baked into code).
- Output a single unified diff string that applies cleanly.
- Target ≥70% problem-level success: a problem counts as success only if all its tests pass; otherwise it is a failure.
- Enable divide-and-conquer: allow focusing on one category at a time via a mode filter, to avoid wasted attempts on the other.

## Guiding Choices (per your selections)

- Generation strategy: code-first (ask LLM for complete file(s)); synthesize diffs locally.
- Scope: comprehensive architectural improvement.
- Targeting: agent self-analyzes failures and prioritizes problem classes strategically.

## High-Level System

- Orchestrator
- Context Builder (dual-mode)
- Requirement Extractor (Spec-only category)
- Failure Extractor (Tests-available category)
- Planner (targets + constraints)
- Code Generator (whitelist model router)
- Diff Synthesizer (local, deterministic)
- Validator (category-specific)
- Critic/Refiner (bounded loop)
- Observability (structured logs + artifacts)
- Safety/Guardrails (write-scope, size caps, protected paths)

## Strict Component Output Formats (I/O Contracts)

All component outputs MUST be JSON objects serializable with UTF-8 and adhere to the following schemas. Unknown fields are ignored but flagged in logs. Required fields must be present; otherwise, the orchestrator aborts the round.

- OrchestratorInput
  - problem_name: string
  - problem_category: "spec_only" | "tests_available"   # spec_only: instructions-only; tests_available: repo/tests present
  - category_filter: "spec_only" | "tests_available" | "all"    # divide-and-conquer switch; defaults to "all"
  - repo_path: string
  - instructions_text?: string
  - run_id: string

- OrchestratorOutput
  - status: "ok" | "error"
  - patch: string (unified diff; empty on error)
  - summary: string
  - metrics: { attempts: number, tokens: number, duration_ms: number, category: "spec_only"|"tests_available" }
  - result: "pass" | "fail"               # pass only if all tests (or self-checks) pass
  - errors?: string[]

- ContextSummary
  - files: Array<{ path: string, size: number, score: number, head: string, tail: string, signatures: string[] }>
  - top_k_used: number
  - truncation: { head_chars: number, tail_chars: number }

- SpecRequirements (spec_only)
  - io_examples: Array<{ input: string | object, output: string | object }>
  - messages_exact: string[]
  - invariants: string[]
  - constraints: { no_external_deps: boolean, performance_notes?: string }
  - self_checks: string                 # Python asserts implementing examples/invariants

- TestFailures (tests_available)
  - failing_tests: string[]
  - traces: Array<{ test: string, file: string, line: number, message: string, top_frames: Array<{file:string,line:number,func:string}> }>
  - implicated_files: string[]

- Plan (shared)
  - targets: string[]                   # ≤3 files
  - functions?: string[]
  - constraints: object
  - rationale: string

- GenerationRequest
  - files_requested: string[]
  - context_brief: string
  - output_rules: { file_marker: "FILE: {path}", fenced_language: "python" }
  - model_prefs: string[]
  - temperature: number
  - max_tokens: number

- GenerationResult
  - status: "ok" | "empty" | "parse_error"
  - files: Record<string, string>
  - model_used: string
  - tokens_used: number
  - notes?: string

- DiffBundle
  - status: "ok" | "error"
  - patch: string
  - file_count: number
  - line_stats: { added: number, removed: number }
  - errors?: string[]

- ValidationReport (spec_only)
  - ok: boolean                         # ok only if all self-checks pass
  - syntax: Array<{ path: string, ok: boolean, error?: string }>
  - self_checks: { ran: boolean, ok: boolean, failures: string[] }
  - smoke: { ran: boolean, ok: boolean, stdout?: string, stderr?: string }

- ValidationReport (tests_available)
  - ok: boolean                         # ok only if all tests pass
  - impacted_tests_ok: boolean
  - full_suite_ok?: boolean
  - failing_tests: string[]
  - traces: Array<{ test: string, message: string, top_frame?: {file:string,line:number} }>

- CritiqueFeedback
  - actionable: boolean
  - summary: string
  - prompts: { generation_delta_instructions: string, constraints_restatements: string }
  - narrowed_targets?: string[]

- OrchestratorState (for logs)
  - round: number
  - mode: "spec_only" | "tests_available"
  - model_history: string[]
  - time_ms: number

- FILE Section Wire Format (LLM output)
  - Each file MUST appear as:
    - `FILE: path/to/file.py`
    - Followed by a fenced code block with the complete file:
      ````
      ```python
      <complete file content>
      ````


```

  - No prose outside these sections.

- Unified Diff Acceptance Criteria
  - Must contain at least one `diff --git` header, valid `---`/`+++`, and `@@` hunks.
  - Applies cleanly on a clean tree; max files ≤ 3; total changed lines ≤ 800.

## Detailed Design

### Orchestrator

- Honors `category_filter`: if set to "spec_only" or "tests_available", skip the other category early with an `OrchestratorOutput` `{status:"error", summary:"skipped by filter", patch:""}` and minimal logging.
- Normal flow:

1) Mode detection and model allocation

2) Context build → ContextSummary

3) Plan → Plan

4) Generate → GenerationResult

5) Diff → DiffBundle

6) Validate → ValidationReport

7) Critique → CritiqueFeedback; refine ≤N rounds

8) Finalize → OrchestratorOutput with `result:"pass"` only if ValidationReport.ok is true; else `result:"fail"`.

### Context Builder

- Discovery, scoring, summarization with token caps.

### Spec-Only Mode (formerly Category A)

- Requirements extraction → SpecRequirements
- Self-check generation and execution
- No access to tests; success requires all self-checks passing.

### Tests-Available Mode (formerly Category B)

- Failure extraction → TestFailures
- Impacted tests first; then full suite
- Success requires all tests passing.

### Code Generation (Code-First) and Diff Synthesizer

- As previously defined; strict parsing and syntax validation before diffing.

### Validator

- Spec-Only: syntax + self_checks + optional smoke
- Tests-Available: impacted → full suite; treat any single failing test as failure

### Critic & Refinement

- Targeted, actionable feedback; narrow scope progressively; bounded rounds.

### Observability (Logging & Telemetry)

- JSONL event stream; artifacts; redaction; rotation, as previously specified.

## Advanced Strategies to Reach ≥70%

- Dual-planner; AST-level editing; speculative branches; committee-of-two; schema constraints; self-test synthesis; impacted-test detection; prompt budgeter; caching.

## Implementation Phases

- Phase 1: Architecture + Contracts (include JSON schema validators)
- Phase 2: Spec-Only Flow
- Phase 3: Tests-Available Flow
- Phase 4: Refinement & Hardening
- Phase 5: Evaluation & Tuning; leverage category_filter to focus development

## Success Criteria & Metrics

- Problem-level success: a problem is counted as success only if all its tests (or all self-checks for Spec-Only) pass; any single failure counts the problem as failed.
- Overall goal: ≥70% of problems in the evaluation corpus counted as successful under the above rule.
- Invalid diffs: <1%; attempts per solved: ≤3 avg; within token/time budgets.

## Risks & Mitigations

- Incomplete outputs → strict schemas and parsers with explicit errors.
- Timeouts → model rotation and context shrinking.
- Variability → deterministic temperatures; optional committee-of-two.

## Deliverables

- `new-agent.py`, config for models/budgets, structured logs, artifacts, README.

### To-dos

- [ ] Add category_filter flag (spec_only|tests_available|all) to orchestrator input and flow control
- [ ] Implement problem-level success rule: require all tests/self-checks to pass
- [ ] Implement orchestrator, mode detection, model router, logging scaffolding
- [ ] Build context builder: discovery, scoring, summarization with caps
- [ ] Implement JSON schema validators for all component outputs
- [ ] Implement structured JSONL logging, artifact writers, redaction, and rotation
- [ ] Implement code-first diff synthesizer with syntax validation
- [ ] Implement Spec-Only requirement extractor and self-check generator
- [ ] Implement Spec-Only prompts, file-section parsing, validator, critic loop
- [ ] Implement Tests-Available failure extractor and planner for surgical edits
- [ ] Implement Tests-Available impacted test runner, full-suite validation, critic loop
- [ ] Add AST editing, speculative branches, committee-of-two, caching, schema-constrained outputs
- [ ] Evaluate on full suite, iterate using category_filter to focus and reach ≥70%

### To-dos

- [ ] Implement orchestrator, mode detection, model router, logging scaffolding
- [ ] Build context builder: discovery, scoring, summarization with caps
- [ ] Implement code-first diff synthesizer with syntax validation
- [ ] Implement polyglot requirement extractor and self-check generator
- [ ] Implement polyglot prompts, file-section parsing, validator, critic loop
- [ ] Implement SWE-bench failure extractor and planner for surgical edits
- [ ] Implement impacted test runner, full-suite validation, critic loop
- [ ] Add budgets, temperature schedules, patch caps, protected paths
- [ ] Evaluate on full suite, analyze failures, iterate prompts and heuristics