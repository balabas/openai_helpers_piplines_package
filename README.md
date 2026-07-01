Reusable helpers for OpenAI-compatible local model workflows.

## Core Idea

This is a thin hardening wrapper over `chat.completions.create`, aimed at **local / OpenAI-compatible model servers** (llama.cpp, vLLM, and similar) where the weak model and the endpoint are less reliable than a hosted frontier API.

You keep the OpenAI call you already know. Per call, you opt into the recovery behavior you want:

- **Tool calling** restore randomly failed local call or MCP-like clients.
- **Structured output** restore randomly failed against a `schema_dict`, with a separate repair round when the first parse fails.
- **Loop guard** restore randomly looped model output.
- **truncation recovery** restore if model exceeds tokens.
- **Logging** of the full request/stream/tool/trace timeline.


## API Shape

There are two call paths.

Stateless calls construct Pipelined Chat from Configured Pipelines, then call `create(...)`:

````python
chat = pipelined_chat(
    client.chat.completions,
    pipelines=pipelines,
)

result = await chat.create(
    messages=messages,
    **llm_request_params,
)
````

Stateful calls wrap the same Pipelined Chat with `chat_session(...)`:

````python
session = chat_session(
    chat,
    role_content=[
        {"system": "You are concise."},
    ],
    **default_llm_request_params,
)

result = await session.step(
    role_content=[
        {"user": "Answer in one sentence."},
    ],
    **override_llm_request_params,
)
````


## Main API

````python
from openai import OpenAI
from openai_helpers_piplines_package import (
    JsonFixPipeline,
    LoggerPipeline,
    LoopGuardPipeline,
    PipelineRequestError,
    ToolPipeline,
    attempt,
    chat_session,
    pipelined_chat,
)

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed-for-local-server",
)

chat = pipelined_chat(
    client.chat.completions,
    pipelines=[
        LoggerPipeline(path="pipeline.log"),
        ToolPipeline(max_retries=3),
        LoopGuardPipeline(max_retries=2),
        JsonFixPipeline(max_retries=2),
    ],
)

result = await attempt(chat.create(
    model="your-model",
    messages=[
        {"role": "user", "content": "Use tools if needed and return JSON."},
    ],

    # Normal chat.completions.create parameters are forwarded.
    temperature=0.2,
    max_tokens=1000,

    # Extra optional pipeline parameters.
    tool_sources=[{"add": add}, mcp_search_client],
    schema_dict={"answer": str, "sources": [str]},
    return_trace=True,
))

if isinstance(result, PipelineRequestError):
    print(result)
else:
    print(result.response)
    print(result.parsed)
    print(result.trace)
````

Pipelined Chat is the object returned by `pipelined_chat(...)`. It exposes an async `create(...)` method.

The mental model is:

````text
pipelined_chat(client.chat.completions, pipelines=...)
    .create(messages=[...], **request_params)

chat_session(
    pipelined_chat(...),
    role_content=[
        {"system": "..."},
    ],
    **default_request_params,
)
    .step(
        role_content=[
            {"user": "..."},
        ],
        **override_request_params,
    )
````

## Local OpenAI-Compatible Server

Use the OpenAI Python SDK for local OpenAI-compatible servers. This package does not implement its own HTTP transport.

````python
from openai import OpenAI
from openai_helpers_piplines_package import pipelined_chat

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed-for-local-server",
)

model_name = client.models.list().data[0].id

chat = pipelined_chat(
    client.chat.completions,
    pipelines=[...],
)
````

## Chat Sessions

Use `chat_session(...)` when you want to keep dialogue history and avoid repeating stable request parameters.

`session.step(...)` calls `chat.create(messages=session.messages.copy_messages(), **merged_params)`.

````python
session = chat_session(
    chat,
    role_content=[
        {"system": "You must call the add tool for arithmetic."},
    ],
    model=model_name,
    temperature=0.1,
    tool_sources=[{"add": add}],
    return_trace=True,
)

result = await session.step(
    role_content=[
        {"user": "Call the add tool to compute 7 + 8, then return JSON with answer and method."},
    ],
    max_tokens=600,
    schema_dict={"answer": str, "method": str},
)
````

`session.messages` is public message history. Use normal OpenAI message dicts when you want full control:

````python
session.messages.append({
    "role": "critic",
    "content": "Check whether the next answer follows the schema.",
})
````

Append a `role_content` dict where each key is a role and each value is content:

````python
session.messages.append_role_content({
    "user": "Continue: compute 20 + 22.",
})
````

Or pass pre-step `role_content` directly to `step(...)`:

````python
result2 = await session.step(
    role_content=[
        {"user": "Continue: compute 20 + 22."},
    ],
    max_tokens=300,
    schema_dict={"answer": str, "method": str},
)
````

Each step can override session defaults.

By default, `auto_append=True`, so the final assistant answer is appended to `session.messages` after a successful call.

Manual mode:

````python
result = await session.step(
    role_content=[
        {"user": "Try again with a shorter answer."},
    ],
    auto_append=False,
    max_tokens=300,
    schema_dict={"answer": str},
)

if result.parsed:
    session.append_result(result)
else:
    session.messages.append_role_content({
        "user": "Correction: return valid JSON.",
    })
````

## Pipeline Parameters

Pipeline configuration belongs to the pipeline objects:

````python
ToolPipeline(max_retries=3)
LoopGuardPipeline(max_retries=2)
JsonFixPipeline(max_retries=2)
LoggerPipeline(path="pipeline.log")
````

Per-call inputs belong to `create(...)`:

````python
result = await attempt(chat.create(
    model=model,
    messages=messages,
    tool_sources=[{"add": add}, mcp_client],
    schema_dict={"answer": str},
    return_trace=True,
))

if isinstance(result, PipelineRequestError):
    print(result)
else:
    print(result.parsed)
````

`schema_dict` is optional. If it is passed, include `JsonFixPipeline` in `pipelines=[...]`.

`tool_sources` is optional. If it is passed, include `ToolPipeline` in `pipelines=[...]`.

Normal OpenAI-compatible arguments such as `model`, `messages`, `temperature`, `max_tokens`, `tools`, `tool_choice`, and `response_format` are forwarded to the source `create(...)` call unless the pipeline needs to add generated tool schemas.

`stream` is managed by the wrapper. Callers do not need to pass it.

The pipeline objects are lightweight configuration markers. The wrapper owns the retry orchestration; the visible order in `pipelines=[...]` is for readability and tracing.

## Pipeline Semantics

Pipelines are configured wrappers, not one flat procedure.

Recommended visible order in `pipelines=[...]`:

````python
pipelines=[
    LoggerPipeline(path="pipeline.log"),
    ToolPipeline(max_retries=3),
    LoopGuardPipeline(max_retries=2),
    JsonFixPipeline(max_retries=2),
]
````

The list order is not an execution DSL. `PipelinedChatCompletions` uses fixed internal semantics:

````text
create(...)
-> prepare optional schema hint
-> stream a chat completion and retry on loop/truncation while generating
-> validate/repair structured output after a complete answer
-> run tool-call rounds when tool calls are present
````

If pipelines are configured in a different visible order, the wrapper emits a warning. The warning exists because wrong ordering makes traces and mental models confusing; it does not mean list position changes the execution graph.

`JsonFixPipeline.max_retries` controls the number of repair rounds after the initial heuristic parse fails.

`LoggerPipeline` records requests, streamed token chunks, tool results, trace events, and final metadata. It does not change model behavior.

`LoopGuardPipeline` checks streamed thought/message text while generation is running. If a loop is detected, the stream is stopped and the partial assistant turn is discarded before the retry logic continues.

`ToolPipeline` handles tool-call normalization, execution, and message construction. Empty tool-round output is nudged with a follow-up user message; truncated output is retried with a larger `max_tokens`.

Use `return_trace=True` when you need pipeline trace visibility:

````python
for event in result.trace:
    print(event.level, event.action, event.detail)
````

## Result Object

`chat.create(...)` returns `PipelinedChatCompletionResult`:

````python
result.response        # final raw chat completion response
result.parsed          # parsed dict if schema_dict was passed, else None
result.trace           # list[PipelineEvent] if return_trace=True, else None
result.messages        # final message history
result.raw_responses   # all raw chat responses from the run
result.tool_executions # executed tool calls
result.usage           # token usage for the final response
result.run_usage       # token usage summed across all raw_responses
````

Assistant answer text and optional reasoning text are inside the final response
message:

````python
message = result.response["choices"][0]["message"]

answer = message.get("content", "")
reasoning_content = message.get("reasoning_content", "")
````

Token usage is provider metadata:

````python
final_usage = result.usage
final_tokens = final_usage.get("total_tokens")

run_usage = result.run_usage
run_tokens = run_usage.get("total_tokens")
````

Use `result.usage` when you only need the final model response. Use
`result.run_usage` when the call may include tool rounds, loop/truncation
retries, or Structured Output repair calls.

Mechanism: Pipelined Chat streams requests with `stream=True` and
`stream_options={"include_usage": True}`. OpenAI-compatible providers that
support this option emit usage metadata in the stream; the package stores the
last usage object on the assembled response. See the OpenAI API reference for
`stream_options.include_usage`:
https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options

## Error Handling

Pipeline failures raise informative exceptions that carry the request context - the params, the full message history, and the provider's own error text - instead of a deep SDK traceback. In Jupyter only the message is shown; the redundant internal frames are suppressed.

The main request-failure type is `PipelineRequestError`. For application branching, use the package-defined names on `PipelineRequestError` plus `classify_pipeline_error(...)` instead of hard-coded strings.

Direct expected pipeline errors are raised as named exception classes:

````python
EmptyAssistantOutputError             # ValueError subclass
StructuredOutputRepairExhaustedError  # ValueError subclass, has .attempts
ToolIterationLimitExceededError       # RuntimeError subclass, has .limit
PipelineRequestError                  # provider/transport wrapper, has .error_kind
````

````python
from openai_helpers_piplines_package import (
    EmptyAssistantOutputError,
    PipelineRequestError,
    StructuredOutputRepairExhaustedError,
    ToolIterationLimitExceededError,
    attempt,
    classify_pipeline_error,
)

result = await attempt(chat.create(messages=msgs, schema_dict=schema))
kind = classify_pipeline_error(result)
if kind == PipelineRequestError.TOOL_REQUEST_FAILED:
    ...   # inspect result.request_stage / result.params / result.messages / result.original
elif kind == PipelineRequestError.JSON_REPAIR_REQUEST_FAILED:
    ...
elif kind == PipelineRequestError.CHAT_REQUEST_FAILED:
    ...
elif kind == PipelineRequestError.EMPTY_ASSISTANT_OUTPUT:
    ...
elif kind == PipelineRequestError.STRUCTURED_OUTPUT_REPAIR_EXHAUSTED:
    ...
elif kind == PipelineRequestError.TOOL_ITERATION_LIMIT_EXCEEDED:
    ...
elif isinstance(result, BaseException):
    raise result
else:
    use(result)

if isinstance(result, PipelineRequestError):
    print(result.request_stage)  # diagnostic string, for logging/debug dumps
````

Structured output repair is not silent when it runs out of retries: if heuristic parsing and repair exhaust their retries, the wrapper raises `StructuredOutputRepairExhaustedError`. The normal flow uses `result.parsed`.

### Errors as values

When a failure is an expected branch in a sequence, wrap the call so expected pipeline errors come back as a value while everything else still raises:

````python
result = await attempt(chat.create(messages=msgs, schema_dict=schema))
kind = classify_pipeline_error(result)
if kind is not None:
    ...  # expected pipeline branch
else:
    use(result.parsed)
````

A genuine bug (for example a `TypeError` in your own code) is not wrapped: it keeps its own type and traceback, `attempt` does not catch it, and it raises immediately. The pipeline still attaches the request params and message history to it as a note so it stays debuggable.

## Logging

Use `LoggerPipeline` as a normal pipeline:

````python
from openai_helpers_piplines_package import LoggerPipeline, pipelined_chat

chat = pipelined_chat(
    client.chat.completions,
    pipelines=[
        LoggerPipeline(path="pipeline.log"),
        ToolPipeline(max_retries=3),
        LoopGuardPipeline(max_retries=2),
        JsonFixPipeline(max_retries=2),
    ],
)
````

The log includes:

- request messages with repeated-message references
- request parameters without duplicating messages
- streamed thinking/message chunks while the model is generating
- tool calls and tool results
- pipeline trace events
- response end metadata, including final-response token usage when provided

When the provider streams thinking fields, the assembled Result response stores
them as `message["reasoning_content"]`. The final assistant answer remains
`message["content"]`.

Most code should use `LoggerPipeline(path=...)` directly. If you need to share one
file writer across multiple wrappers, build a `PipelineLogger` from the internal
module and pass it in:

````python
from openai_helpers_piplines_package import LoggerPipeline
from pipelines.logger import PipelineLogger

logger = PipelineLogger("pipeline.log")

chat = pipelined_chat(
    client.chat.completions,
    pipelines=[
        LoggerPipeline(logger=logger),
        ToolPipeline(max_retries=3),
    ],
)
````

## Tool Sources

### Local tools 
can be passed as a dictionary:

````python
def add(arguments: dict[str, int]) -> dict[str, int]:
    return {"sum": arguments["a"] + arguments["b"]}

result = await attempt(chat.create(
    model=model,
    messages=[
        {"role": "user", "content": "Use the add tool to compute 12 + 30."},
    ],
    tool_sources=[{"add": add}],
))

if isinstance(result, PipelineRequestError):
    print(result)
else:
    print(result.response)
````

Callable tools can also use keyword arguments:

````python
def add(a: int, b: int) -> dict[str, int]:
    return {"sum": a + b}

result = await attempt(chat.create(
    model=model,
    messages=messages,
    tool_sources=[{"add": add}],
))

if isinstance(result, PipelineRequestError):
    print(result)
else:
    print(result.response)
````

### MCP
MCP-like clients are supported when they expose:

````python
await mcp_client.list_tools()
await mcp_client.call_tool(name, arguments)
````

Then pass the client directly:

````python
result = await attempt(chat.create(
    model=model,
    messages=messages,
    tool_sources=[mcp_search_client, {"add": add}],
))

if isinstance(result, PipelineRequestError):
    print(result)
else:
    print(result.response)
````

## Structured Output

Use `schema_dict` per call:

````python
chat = pipelined_chat(
    client.chat.completions,
    pipelines=[JsonFixPipeline(max_retries=2)],
)

result = await attempt(chat.create(
    model=model,
    messages=[
        {"role": "user", "content": "Return a compact JSON profile."},
    ],
    schema_dict={
        "name": str,
        "score": float,
        "tags": [str],
    },
))

if isinstance(result, PipelineRequestError):
    print(result)
else:
    print(result.parsed)
````

The wrapper first tries heuristic JSON cleanup and validation. If that fails, it makes a repair call with an explicit fit prompt and a JSON-schema response format. If that still does not produce valid JSON after the configured retries, the wrapper raises `ValueError`.

## Individual Helpers

The lower-level helpers remain available when you do not want the combined wrapper.

### JsonFixPipeline

````python
from openai_helpers_piplines_package import JsonFixPipeline

pipeline = JsonFixPipeline(max_retries=2)

request = pipeline.build_request(
    messages=[{"role": "user", "content": "Return a person profile as JSON."}],
    schema_dict={"name": str, "age": int, "tags": [str]},
)

# Send request.messages to your model, then parse the returned text.
parsed = pipeline.parse(raw_text, request.model_cls)
````

The helper also exposes `build_fit_prompt(...)` and `build_retry_messages(...)` for the repair round used by the wrapper.

### LoopGuardPipeline

````python
from openai_helpers_piplines_package import LoopGuardPipeline

guard = LoopGuardPipeline(max_retries=2)
reason = guard.check(text)

if reason:
    retry_messages = guard.build_retry_messages(text, reason)
````

### ToolPipeline

````python
from dataclasses import asdict

from openai_helpers_piplines_package import ToolPipeline

tool_pipeline = ToolPipeline(max_retries=3)

executions = await tool_pipeline.execute_tool_calls(raw_tool_calls, {"add": add})
print([asdict(execution) for execution in executions])

assistant_message = tool_pipeline.assistant_message("", raw_tool_calls)
tool_messages = tool_pipeline.tool_messages(executions)
````

`ToolPipeline` also exposes `openai_tools(...)` for normalizing tool schemas before request submission.

## Dynamic Pydantic Schemas

````python
from openai_helpers_piplines_package import dict_to_pydantic_schema

Model = dict_to_pydantic_schema(
    {
        "name": str,
        "age": int,
        "active": True,
        "tags": [str],
        "profile": {
            "city": str,
            "score": float,
        },
    }
)

validated = Model.model_validate(
    {
        "name": "Ana",
        "age": 31,
        "tags": ["local", "demo"],
        "profile": {"city": "Berlin", "score": 0.95},
    }
)

print(validated.model_dump())
print(Model.model_json_schema())
````
