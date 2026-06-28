"""Schema conversion helper: ``dict`` → Pydantic ``BaseModel``.

Responsibility
--------------
Converts a caller-supplied ``schema_dict`` (a plain Python dict describing the
desired output shape) into a dynamically-created Pydantic ``BaseModel`` class.
The resulting class is used by ``JsonFixPipeline`` to validate and coerce the
model's JSON output, and to generate a JSON Schema string for the prompt.

The conversion supports:

- Primitive field types (``str``, ``int``, ``float``, ``bool``)
- Nested dicts → nested Pydantic models
- List fields with typed items or ``Literal`` constraints
- Optional fields (value ``None``)
- Literal defaults (value is an instance → type inferred from the value)
- Key-value list mode (``as_kv_list=True``) for flat mappings

Usage
-----
Only imported by ``pipelines/json_fix.py``.
No imports from the rest of this package.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Literal

from pydantic import BaseModel, RootModel, create_model


def dict_to_pydantic_schema(
    schema_dict: Dict[str, Any],
    name_for_llm: str = "DynamicSchema",
    *,
    as_kv_list: bool = False,
) -> Type[BaseModel]:
    """Create a Pydantic model class from a plain dict schema description.

    Parameters
    ----------
    schema_dict:
        Describes the expected output.  Supported value forms per key:

        - ``str`` / ``int`` / ``float`` / ``bool`` — the field's type (required)
        - ``"example"`` / ``42`` / ``True`` — literal default, type inferred
        - ``None`` — optional field, defaults to ``None``
        - ``dict`` — nested object; recursed into a sub-model
        - ``[type]`` — list of ``type``
        - ``[dict]`` — list of nested objects
        - ``["a", "b"]`` — list of literals (Literal["a", "b"])

    name_for_llm:
        Class name used in the generated JSON Schema; shown to the model in
        the prompt so choose something descriptive.
    as_kv_list:
        When ``True``, ignore ``schema_dict`` values and produce a
        ``RootModel[list[{key: str, value: <inferred>}]]``.  Useful when the
        model should return a list of key-value pairs rather than a flat dict.

    Returns
    -------
    Type[BaseModel]
        A dynamically created Pydantic model class ready for
        ``model_validate(…)`` and ``model_json_schema()``.
    """
    if as_kv_list:
        value_types = []
        for v in schema_dict.values():
            value_types.append(v if isinstance(v, type) else type(v))
        value_type = (
            value_types[0]
            if value_types and all(t is value_types[0] for t in value_types)
            else Any
        )
        item_fields = {
            "key":   (str,        ...),
            "value": (value_type, ...),
        }
        item_model = create_model(f"{name_for_llm}Item", **item_fields)
        return type(name_for_llm, (RootModel[list[item_model]],), {})

    fields = {}
    for k, v in schema_dict.items():
        if isinstance(v, list):
            if not v:
                fields[k] = (List[Any], ...)
            elif len(v) == 1:
                item = v[0]
                if isinstance(item, type):
                    fields[k] = (List[item], ...)
                elif isinstance(item, dict):
                    inner_model = dict_to_pydantic_schema(item, name_for_llm=f"{k.capitalize()}Item")
                    fields[k] = (List[inner_model], ...)
                else:
                    fields[k] = (List[Literal[item]], ...)
            elif all(isinstance(item, (str, int, float, bool)) for item in v):
                # Homogeneous list of scalars → Literal constraint
                fields[k] = (List[Literal[tuple(v)]], ...)
            else:
                raise ValueError(f"Ambiguous or unsupported list field '{k}': {v}")
        elif isinstance(v, dict):
            inner_model = dict_to_pydantic_schema(v, name_for_llm=f"{k.capitalize()}Model")
            fields[k] = (inner_model, ...)
        elif v is None:
            fields[k] = (Optional[Any], None)
        elif isinstance(v, type):
            # Bare type → required field
            fields[k] = (v, ...)
        else:
            # Instance value → infer type, use as default
            fields[k] = (type(v), v)

    return create_model(name_for_llm, **fields)
