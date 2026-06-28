"""Root-level import surface for the helper pipelines package.

This module lives at the repository root so ``import openai_helpers_piplines_package``
resolves here when the notebook is run from this folder.
"""
from helpers.pydantic_helper import dict_to_pydantic_schema
from pipelines.chat import with_pipelines
from pipelines.json_fix import JsonFixPipeline
from pipelines.logger import LoggerPipeline
from pipelines.loop_guard import LoopGuardPipeline
from pipelines.session import chat_session
from pipelines.tool import ToolPipeline

__all__ = [
    "with_pipelines",
    "chat_session",
    "JsonFixPipeline",
    "LoggerPipeline",
    "LoopGuardPipeline",
    "ToolPipeline",
    "dict_to_pydantic_schema",
]
