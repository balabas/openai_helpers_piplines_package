"""Root-level import surface for the helper pipelines package.

This module lives at the repository root so ``import openai_helpers_piplines_package``
resolves here when the notebook is run from this folder.
"""
from helpers.pydantic_helper import dict_to_pydantic_schema
from pipelines.chat import (
    EmptyAssistantOutputError,
    PipelineDebugStage,
    PipelineRequestError,
    StructuredOutputRepairExhaustedError,
    ToolIterationLimitExceededError,
    attempt,
    classify_pipeline_error,
    pipelined_chat,
)
from pipelines.json_fix import JsonFixPipeline
from pipelines.logger import LoggerPipeline
from pipelines.loop_guard import LoopGuardPipeline
from pipelines.session import chat_session
from pipelines.tool import ToolPipeline

__all__ = [
    "pipelined_chat",
    "PipelineRequestError",
    "PipelineDebugStage",
    "EmptyAssistantOutputError",
    "StructuredOutputRepairExhaustedError",
    "ToolIterationLimitExceededError",
    "attempt",
    "classify_pipeline_error",
    "chat_session",
    "JsonFixPipeline",
    "LoggerPipeline",
    "LoopGuardPipeline",
    "ToolPipeline",
    "dict_to_pydantic_schema",
]
