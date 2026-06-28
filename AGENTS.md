# Global Agent Instructions

## Self-Correction

Be careful with semantics, levels, layers, and visibility before proposing names, APIs, or implementation changes.

When corrected by the user, treat the correction as a semantic constraint for future reasoning in the current project. Update explanations and future suggestions to preserve that constraint.

Do not flatten layered behavior into one opaque abstraction when the design requires visible layers, events, or responsibilities.

## Naming

Prefer unifying names with existing close semantic patterns in the codebase instead of inventing new names.

Use the same option name only when the semantic role is actually the same. If the same option name appears in multiple layers or components, its meaning may be scoped by that layer or component.

Avoid adding near-duplicate names for behavior already represented by an existing option.

Only introduce a new option name when it represents a genuinely different control or responsibility.

## Reasoning

Before suggesting an API or implementation, check whether the codebase already has an equivalent concept.

Identify distinct responsibilities from the actual code and user corrections before naming or combining behavior.

Do not treat separate responsibilities as interchangeable just because they look similar at a high level.

## project instruction

use /home/ubn/Documents/projects/unified_local_llm_server/.venv for venv