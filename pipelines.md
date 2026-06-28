# Pipeline algorithms

Reference specification for the pipeline levels, ported from `test_agent2` and
rewritten cleanly for the OpenAI-compatible chat API. This is the authoritative
description; the implementation in `pipelines/` must match it.

## Layering

The levels are nested wrappers. Each level calls the level below it to *produce
one answer* and adds only its own concern. From lowest (innermost, closest to
the raw model call) to highest (outermost):

```
tool_result_fix          (highest)  — tool-call loop + empty-output repair
  └ json_fix             (middle)   — structured output: generate → extract
      └ loop_guard_fix +              — one clean, COMPLETE generation
        max_tokens_fix   (lowest)     (re-roll hotter on a loop / longer on truncation)
          └ [ raw streaming chat call ]
```

- **loop_guard_fix + max_tokens_fix** is the lowest level: it works on the raw
  token stream and owns both single-generation retries (loop and truncation).
- **json_fix** is the middle level: it works on a complete answer.
- **tool_result_fix** is the highest level: it orchestrates multi-round tool use.

A level is **transparent** (pure pass-through) when its feature is not requested:
`json_fix` with no `schema`, `tool_result_fix` with no `tools`.

## Shared — retry temperature schedule

```
escalate(base, n):                 # n >= 1 (retry attempt number)
    b = base if (base and base > 0) else 0
    if b == 0:  return 0.1 * 2**(n-1)      # greedy/unset -> 0.1, 0.2, 0.4, ...
    else:       return b   * 2**(n-1)      # base -> base, 2*base, 4*base, ...
```

Off-by-one is intentional: with a positive base, the **first** retry equals the
base (a plain re-roll); doubling begins on the second retry. From greedy/unset
the first retry jumps to `0.1`.

---

## Level 0 — raw streaming chat call (the primitive)

Not a "fix" level; the call every level ultimately drives.

- Forces streaming; requests usage so token counts and `finish_reason` are known.
- Maps generation params to the OpenAI request (`max_tokens`, `temperature`,
  `top_p`, ...) and sends them **as provided**. Transport-specific params with no
  OpenAI equivalent are dropped. (The truncation `max_tokens` boost is owned by
  the lowest level, `loop_guard_fix + max_tokens_fix`, not here.)
- `schema` → `response_format` as a `json_schema` (with a `json_object` fallback
  if the server rejects `json_schema`).
- Tracks `finish_reason`; `"length"` means the output was **truncated**
  (token budget exhausted).
- Empty output (no text, no tool calls) is surfaced as an error to the caller.

---

## Level 1 (lowest) — `loop_guard_fix` + `max_tokens_fix`

**Wraps:** the raw streaming chat call.
**Concern:** produce ONE clean, COMPLETE generation (neither looping nor
truncated). Optionally appends a `schema` as a text hint to the messages before
generating. Validation and repair are NOT done here — that is `json_fix`.
**Inputs:** messages, gen params, optional `schema` (hint only),
optional `response_format`.
**Constants:** `max_loop_retries = 3`, `max_tokens_retries = 3`,
`max_tokens_step = 5000`.

```
# optional schema text hint — appended to messages, NO message-count condition
if schema:
    messages[-1].content += "\nOutput format schema:\n```json\n" + schema + "\n```"
    # appended to an existing message, NEVER a new system message

loop_retries = 0
boost        = 0
while True:
    temp       = base_temperature if loop_retries == 0
                 else escalate(base_temperature, loop_retries)
    max_tokens = base_max_tokens + boost
    gen = stream the chat call at (temperature = temp, max_tokens = max_tokens)
        # loop detection runs on each streamed thought/message token via check_loop;
        # a detected loop aborts the stream immediately

    if gen looped and loop_retries < max_loop_retries:
        discard the partial assistant turn        # never enters the conversation
        loop_retries += 1
        continue                                  # re-roll HOTTER, no prompt

    if gen truncated (finish_reason == "length") and boost/max_tokens_step < max_tokens_retries:
        discard the partial assistant turn
        boost += max_tokens_step
        continue                                  # re-roll LONGER, no prompt

    break                                         # clean, complete generation

# cheap heuristic validation FIRST — no extra LLM call
if schema:
    candidate = heuristic_fix(gen.text)
    try:
        gen.parsed = validate(schema, candidate)  # OK -> structured result, no LLM repair needed
    except ValidationError:
        pass                                      # leave repair to json_fix (LLM)

return gen                                        # { text, tool_calls, finish_reason, parsed? }
```

### Streaming loop detection

Loops are caught **while the response streams**, not after it completes:

- Two separate accumulators are filled token-by-token: a **message** buffer
  (`content`) and a **thought** buffer (`reasoning`/`thinking`).
- After each token, `check_loop(buffer)` runs on the buffer that just grew.
- On a hit the stream is **aborted immediately** (close the HTTP stream — stop
  generation mid-loop), the response is marked `finish_reason = "loop_guard"`,
  and the loop `reason` + `scope` (`thought` | `message`) are recorded.
- The retry logic then **discards the partial assistant turn** (it is never
  appended to the conversation) and re-rolls. So a looping turn leaves no trace.

### `check_loop(text)` — the detectors

Run in this order (cheapest / most specific first); the first hit returns its
reason string, else `None`:

1. **last_token_split** — split `text` by its second-to-last whitespace token; if
   the last three resulting segments are identical and non-trivial, it is a loop.
   O(n); guarded to apply only once output is long (≥ ~1000 tokens).
2. **sequence_loop** — the last `LOOP_WINDOW` chars reappear ≥ `LOOP_MIN_HITS`
   times within the last `LOOP_LOOKBACK` chars, AND those repeats cover ≥
   `LOOP_MIN_SUM_LEN` chars total (so short punctuation runs don't trip it).
3. **numeric_list** — a dense run of 50+ numbers separated by commas/space in the
   last 2000 chars (degenerate counting / id dumps).
4. **incrementing_sequence** — ≥ `LOOP_INCR_SEQ_MIN_LINES` consecutive lines that
   are identical after replacing every digit with `#` (e.g. `item 1`, `item 2`, …).
5. **numbered_block_cycle** — a multi-line block repeats with only its numbers
   changing; detected by stripping all digits, then applying the `sequence_loop`
   test.

**Thresholds (environment-tunable):** `LOOP_WINDOW=220`, `LOOP_LOOKBACK=4000`,
`LOOP_MIN_HITS=4`, `LOOP_MIN_SUM_LEN=700`, `LOOP_INCR_SEQ_MIN_LINES=15`.

### Escalation (independent loop vs truncation budgets)

- **loop** retries use the `escalate(base_temperature, loop_retries)` schedule;
  they do **not** change `max_tokens`.
- **truncation** retries grow `boost` by `max_tokens_step`; they do **not**
  change `temperature`.
- The two are mutually exclusive per generation: a detected loop aborts the
  stream (`finish_reason = "loop_guard"`) before a `length` finish can occur.
- Each has its own cap (`max_loop_retries`, `max_tokens_retries`); when a cap is
  reached the generation is returned as-is (looping or truncated) rather than
  retried forever.

### `heuristic_fix(text)` — text-only JSON repair (no model call)

1. strip code fences — remove ` ```json ` / ` ``` `;
2. fix stray `"""` delimiters (remove or add to balance);
3. drop trailing commas and `//` / `/* */` comments;
4. take the **last** balanced JSON block (try `[...]` before `{...}`);
5. wrap a bare array when the schema root is a single field; coerce item fields
   (e.g. `range:[a, b] -> start / end`).

### Output upward

One clean, complete generation: `{ text, tool_calls, finish_reason, parsed? }`.
`parsed` is set only when `schema` was given and `heuristic_fix` + `validate`
succeeded — signalling the caller (`json_fix`) that no LLM repair is needed.

---

## Level 2 (middle) — `json_fix`

**Wraps:** `loop_guard_fix`.
**Concern:** **LLM-based** JSON repair — invoked ONLY when the lowest level's
cheap heuristic validation could not produce valid JSON. Pass-through when no
`schema`.
**Constant:** `max_retries = 3`.

```
gen = loop_guard_fix(messages, schema = schema)   # generation + heuristic validate
if no schema:
    return gen
if gen.parsed is set:
    return gen.parsed                # heuristic path already produced valid JSON — no LLM call

# fit_to_schema: a SEPARATE, fresh conversation (LLM repair)
fit_msgs = [ user: FIT_PROMPT(schema, raw_text = gen.text) ]
for _ in 0 .. max_retries:
    body = loop_guard_fix(fit_msgs, response_format = schema)   # json_schema API, FIT prompt
    if body.parsed is set:
        return body.parsed
    fit_msgs += [ user:
        "Previous JSON failed validation. "
        "Repair ONLY the JSON. Do NOT add or remove semantic entities." ]
return model.model_construct(first_field = gen.text)           # graceful fallback — never raises
```

**`FIT_PROMPT`** instructs the model to *extract response as JSON from the
reply, choose the last JSON block, correct invalid JSON, preserve semantic
content exactly, and return JSON matching the given schema.*

**Order of effort (cheap → expensive):** heuristic text fix (lowest level) →
only on failure, one or more LLM repair rounds (here) → graceful
`model_construct` fallback.

**Schema reaches the model only in explicit, visible ways:** a user-message hint
(generation), and the `FIT_PROMPT` + `response_format` constraint (repair). No
hidden system-prompt injection. The level **never hard-raises** — exhausted
repair yields the `model_construct` fallback.

---

## Level 3 (highest) — `tool_result_fix`

**Wraps:** `json_fix`.
**Concern:** tool-call rounds and repair of empty tool output. Pass-through when
no `tools`. Generations arrive already complete (loop + truncation handled at the
lowest level), so this level never sees a truncated turn.
**Constants:** `max_rounds = 20`, `max_empty_streak = 3`.

```
empty_streak = 0
for round in 0 .. max_rounds:
    if empty_streak > 0:
        temperature = escalate(base_temperature, empty_streak)     # heat-up
    gen = json_fix(messages, tools = tools)   # one answer (clean, complete; schema-validated if final)

    if gen is EMPTY (no text and no tool_calls):
        empty_streak += 1
        if empty_streak >= max_empty_streak:
            break
        messages += [ user: "continue, check tool call correctness" ]
        continue

    empty_streak = 0

    if gen.tool_calls:
        for each call:
            if name == request_clarification and on_clarify:
                answer = on_clarify(question); append as a user message
            else:
                result = execute(name, arguments)
            if len(result) > threshold:
                result = compress(result)          # side LLM call, saves context
        append assistant(tool_calls) + tool result messages
        continue
    else:
        return gen                                 # final answer
```

**Repair strategy (no instruction injection except the single empty-output
nudge):**
- **empty output after a tool round** → heat up temperature and add one
  `"continue, check tool call correctness"` nudge, capped at `max_empty_streak`.

(Truncation is not handled here — the lowest level already returns a complete,
non-truncated generation.)

---

## Invariants (must hold in the implementation)

- **No hidden prompt injection.** The schema reaches the model only as the
  optional, visible user-message hint (lowest level), the `FIT_PROMPT`, and the
  `response_format` API. No system message and no corrective prompt are injected.
- **loop_guard_fix + max_tokens_fix** (lowest) owns both single-generation
  retries: a detected loop discards the partial turn and re-rolls hotter; a
  truncated turn (`finish_reason == "length"`) discards the partial turn and
  re-rolls with `max_tokens += max_tokens_step`. Neither adds a message.
- **Schema hint is optional and unconditional in placement** — appended to an
  existing message at the lowest level, with no message-count condition.
- **Heuristic before LLM.** The lowest level applies text-only `heuristic_fix` +
  validate first; the LLM repair (`json_fix`) runs ONLY when that fails. A model
  that already returned valid JSON costs no extra call.
- **json_fix** never adds a `system` message and never hard-raises (graceful
  `model_construct` fallback).
- **tool_result_fix**: empty-after-tool → temperature heat-up (+ one nudge),
  capped. Truncation is not its concern (handled lower).
- **Temperature/`max_tokens` escalation** uses the shared schedule and the fixed
  step constants above.
- **Schema validation** runs only on the **final** (no-tool-call) answer;
  `json_fix` is transparent during intermediate tool rounds.
