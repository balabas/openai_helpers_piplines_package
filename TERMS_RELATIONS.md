# Project Terms and Relations

This file defines project terminology and visibility boundaries for public
docs, notebooks, public API names, user-facing errors, explanations, and
internal diagnostic terms that must not leak into user-facing taxonomy.

Before changing public terminology, check this file. If a user instruction
conflicts with this file, warn about the conflict first and change the taxonomy
only after explicit user acceptance.

## Package

User-Facing:
Yes.

Description:
The package is the install/import surface for the helper project. It exposes
the public constructors, pipeline classes, session helper, result types, error
classes, and classification helper.

Relations:
The package exposes the public vocabulary used by docs, notebooks, examples,
errors, and helper names. It contains pipeline constructors, session helpers,
result types, and error taxonomy. It does not define a new provider transport
model. The package name uses the current project spelling; do not silently
introduce alternate spelling in user-facing text.

## Pipeline

User-Facing:
Yes.

Description:
A pipeline is a configured behavior unit that can participate in a chat
completion run. Pipeline is the user-facing term for configured behavior such
as tool handling, structured output repair, loop guarding, or logging.

Relations:
Pipeline is the family name used by public classes whose names end in
`Pipeline`. Pipelines belong to Pipelined Chat construction through the
configured pipeline list. Pipelines may consume request parameters, observe
stream output, repair output, execute tools, or emit trace events.

## Configured Pipelines

User-Facing:
Yes.

Description:
Configured Pipelines is the user-facing term for the list of pipeline objects
used to construct Pipelined Chat.

Relations:
Configured Pipelines belongs to Pipelined Chat construction. It does not belong
to individual session steps, Messages, Role_Content, Tool Sources, or request
history.

## Pipelined Chat

User-Facing:
Yes.

Description:
Pipelined Chat is the wrapped chat-completions object produced by applying
Configured Pipelines to the underlying provider client.

Relations:
Pipelined Chat owns the Configured Pipelines for future calls. It preserves the
OpenAI-compatible chat-completions request model while adding pipeline-managed
behavior around it. Pipelined Chat does not own conversation history unless it
is wrapped by a Session. The public constructor helper name is the snake_case
form of this term.

## Stateless Call

User-Facing:
Yes.

Description:
A Stateless Call is a direct call on Pipelined Chat.

Relations:
The caller supplies the full Messages history for a Stateless Call. Stateless
Call state lives outside the helper. Stateless Call is separate from Session
Step.

## Session

User-Facing:
Yes.

Description:
A Session is a stateful wrapper around Pipelined Chat. It owns persistent
conversation history and default request parameters.

Relations:
Session is the user-facing term for multi-step dialogue state. A Session stores
default Request Parameters separately from conversation history. Session
defaults are not Messages. Messages are not Request Parameters. A Session
produces Session Steps.

## Session Step

User-Facing:
Yes.

Description:
A Session Step is one model call through a Session.

Relations:
A Session Step uses Session history plus optional Role_Content additions, then
delegates to Pipelined Chat. Before the call, the Session may append
Role_Content to its history. After a successful call, the Session may append
the assistant result depending on Session Step options.

## Role_Content

User-Facing:
Yes.

Description:
Role_Content is one atomic term. It is Session-facing shorthand for appending
content under one or more roles.

Relations:
Role_Content means role-to-content mapping. Role_Content is not Messages and is
not the OpenAI-compatible provider-facing shape. Role_Content is converted into
Messages before a model request. Role_Content belongs to Sessions and Session
message helpers. Role_Content must not be described as a Request Parameter
forwarded to chat completion.

## Messages

User-Facing:
Yes.

Description:
Messages is one atomic term. Messages are OpenAI-compatible chat message
dictionaries stored in request history and sent to the model.

Relations:
Messages are provider-facing. Messages are not Role_Content. Messages are used
by Pipelined Chat requests and stored by Sessions as conversation history.
Role_Content is converted into Messages before a provider request.

## Request Parameters

User-Facing:
Yes.

Description:
Request Parameters are values that configure a model call.

Relations:
Request Parameters may be supplied directly to a Stateless Call or stored as
Session defaults and overridden per Session Step. Some Request Parameters are
OpenAI-compatible provider parameters. Other Request Parameters are
pipeline-managed controls. Both belong to the request-configuration category,
but their responsibilities must remain visible.

## Structured Output

User-Facing:
Yes.

Description:
Structured Output is pipeline-managed parsing and repair for responses expected
to match a schema.

Relations:
Structured Output is managed by the structured-output Pipeline. Schema-related
Request Parameters activate Structured Output behavior, but those parameters
are not themselves Pipelines. Structured-output repair is Pipeline behavior
after an invalid model response, not a separate user-facing request branch
taxonomy.

## Tool Sources

User-Facing:
Yes.

Description:
Tool Sources are caller-provided local or MCP-like tool providers.

Relations:
Tool Sources are used by the Tool Pipeline. Tool Sources are not Pipelines.
Tool Sources are per-call input interpreted by the Tool Pipeline. Tool
Executions belong to Result and Trace visibility, not to Configured Pipelines.

## Trace

User-Facing:
Yes.

Description:
Trace is the optional structured timeline of Pipeline events for a run.

Relations:
Trace belongs to observability. Trace does not replace provider response,
parsed output, Result, or error taxonomy.

## Result

User-Facing:
Yes.

Description:
Result is the object returned by a successful Pipelined Chat call or Session
Step.

Relations:
Result groups the final provider response, optional parsed Structured Output,
optional Trace, final Messages, raw responses, and Tool Executions.

## Reasoning Content

User-Facing:
Yes.

Description:
Reasoning Content is the optional assistant reasoning text assembled from
streamed provider thinking chunks.

Relations:
Reasoning Content belongs to Result response Messages. It is stored at
`result.response["choices"][0]["message"]["reasoning_content"]` when the
provider streams thinking fields. Reasoning Content is not Trace, parsed
Structured Output, Tool Executions, or the final assistant answer content.
LoggerPipeline may log the same streamed thinking chunks while generation is
running.

## Token Usage

User-Facing:
Yes.

Description:
Token Usage is the provider-reported token count for generated model
responses.

Relations:
Token Usage belongs to Result. `result.usage` is the final response usage.
`result.run_usage` is the aggregate usage across all raw responses in a
Pipelined Chat run, including tool rounds, retries, and Structured Output
repair calls. Token Usage comes from provider response usage metadata; it is
not Messages, Trace, Reasoning Content, or Tool Executions.

## Expected Pipeline Error

User-Facing:
Yes.

Description:
Expected Pipeline Error is a known operational branch that application code may
classify and handle.

Relations:
Expected Pipeline Error is distinct from a genuine bug. Genuine bugs should
keep their own type and traceback. Expected Pipeline Error is classified by
Error Kind.

## Attempt

User-Facing:
Yes.

Description:
Attempt is the package helper that turns Expected Pipeline Errors into return
values while allowing genuine bugs to raise.

Relations:
Attempt belongs to error handling. Attempt uses Error Kind classification to
decide whether an exception is an Expected Pipeline Error. Attempt does not
change Pipelined Chat, Session, Messages, Role_Content, or Request Parameters.

## Request Error

User-Facing:
Yes.

Description:
Request Error is a provider or transport failure wrapped with request context.

Relations:
Request Error is one category of Expected Pipeline Error. Request Error carries
Request Parameters, Messages, provider/transport details, and diagnostic
metadata.

## Error Kind

User-Facing:
Yes.

Description:
Error Kind is the public branch classification used by application code.

Relations:
Error Kind is the application branching taxonomy. Error Kind should be exposed
through package-defined constants, not informal strings invented in docs or
notebooks. Error Kind is separate from Request Stage and Debug Stage.

## Request Stage

User-Facing:
Diagnostic only.

Description:
Request Stage is diagnostic metadata attached to Request Error.

Relations:
Request Stage is for logging and debugging. Request Stage is not the primary
application branching taxonomy.

## Context Path

User-Facing:
Diagnostic only.

Description:
Context Path is the ordered diagnostic tag path attached to logger output.

Relations:
Context Path belongs to logging context. It groups log lines for readability.
Context Path is not Pipeline, Configured Pipelines, Error Kind, Request Stage,
or Debug Stage.

## Debug Stage

User-Facing:
No.

Description:
Debug Stage is a controlled insertion point for debug exception generation in
tests or demos.

Relations:
Debug Stage is not a user-facing error taxonomy. Debug Stage is separate from
Error Kind and Request Stage.
