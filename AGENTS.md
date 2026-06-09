# Repo Guidelines

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Communication

Use caveman mode (full) for all responses. Invoke `caveman:caveman` skill at session start if not already active. Drop articles, filler, pleasantries, hedging. Fragments OK. Code/commits/security warnings: write normal.

## 6. Docs convention

Every Markdown file or report created by Codex/ClaudeCode harness MUST use this filename pattern:
`<cdx|cc>_v<number>_<short_title>_<DD-MM-YY_HH-MM>.md`

- For Codex Prefix MUST be `cdx_`, For ClaudeCode perfix MUST be `cc_`.
- Version MUST use `v<number>`, starting with `v1` for the first version.
- Short title MUST be concise and use snake_case, for example `tech_design`.
- Datetime MUST use `DD-MM-YY_HH-MM`.
- Example: `cdx_v1_tech_design_30-05-26_14-30.md`
`cc_v1_tech_design_30-05-26_14-30.md`
