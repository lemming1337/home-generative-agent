"""Langgraph graphs for Home Generative Agent."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections import Counter
from enum import Enum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import voluptuous as vol
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from ..const import (  # noqa: TID252
    CONF_CHAT_MODEL_PROVIDER,
    CONF_GEMINI_CHAT_MODEL,
    CONF_OLLAMA_CHAT_MODEL,
    CONF_OPENAI_CHAT_MODEL,
    CONTEXT_MANAGE_USE_TOKENS,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_TOKENS,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    RECOMMENDED_SUMMARIZATION_INITIAL_PROMPT,
    RECOMMENDED_SUMMARIZATION_PROMPT_TEMPLATE,
    RECOMMENDED_SUMMARIZATION_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
    TOOL_CALL_TIMEOUT_SECONDS,
)
from ..core.logging_utils import (  # noqa: TID252
    format_message_for_log,
    format_messages_summary,
    log_with_context,
)
from ..core.utils import extract_final  # noqa: TID252
from .token_counter import count_tokens_cross_provider
from .tool_metrics import ToolCallMetrics

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from langchain_core.runnables import RunnableConfig
    from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)


class ToolErrorType(str, Enum):
    """Classification of tool call errors."""

    VALIDATION = "validation"  # Invalid parameters or schema mismatch
    EXECUTION = "execution"  # Tool executed but failed
    TIMEOUT = "timeout"  # Tool call exceeded time limit
    NOT_FOUND = "not_found"  # Tool does not exist
    UNKNOWN = "unknown"  # Unknown error type

    @classmethod
    def classify(
        cls, error: Exception, timeout_exceeded: bool = False
    ) -> "ToolErrorType":
        """Classify error type based on exception.

        Args:
            error: The exception to classify
            timeout_exceeded: Whether timeout was exceeded

        Returns:
            ToolErrorType classification
        """
        if timeout_exceeded:
            return cls.TIMEOUT
        if isinstance(error, ValidationError):
            return cls.VALIDATION
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return cls.VALIDATION
        if isinstance(error, AttributeError):
            return cls.NOT_FOUND
        if isinstance(error, HomeAssistantError):
            err_str = str(error).lower()
            if "not found" in err_str or "unknown" in err_str:
                return cls.NOT_FOUND
            if "invalid" in err_str or "schema" in err_str:
                return cls.VALIDATION
        return cls.EXECUTION


class State(MessagesState):
    """Extend MessagesState."""

    summary: str
    chat_model_usage_metadata: dict[str, Any]
    messages_to_remove: list[AnyMessage]


# ----- Utilities -----


async def _retrieve_camera_activity(
    hass: HomeAssistant, store: BaseStore
) -> list[dict[str, dict[str, str]]]:
    """Retrieve most recent camera activity from video analysis by the VLM."""
    camera_activity: list[dict[str, dict[str, str]]] = []
    for entity_id in hass.states.async_entity_ids():
        if not entity_id.startswith("camera."):
            continue
        camera = entity_id.split(".")[-1]
        results = await store.asearch(("video_analysis", camera), limit=1)
        if results and (la := results[0].value.get("content")):
            camera_activity.append(
                {
                    camera: {
                        "last activity": la,
                        "date_time": results[0].updated_at.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                }
            )
    if camera_activity:
        LOGGER.debug("Recent camera activity: %s", camera_activity)
        return camera_activity
    LOGGER.debug("No recent camera activity found.")
    return []


def _determine_model_name(provider: str, opts: dict[str, Any]) -> str:
    """Determine model name based on provider and options."""
    if provider == "openai":
        return opts.get(CONF_OPENAI_CHAT_MODEL, "")
    if provider == "gemini":
        return opts.get(CONF_GEMINI_CHAT_MODEL, "")
    return opts.get(CONF_OLLAMA_CHAT_MODEL, "")


# ----- Graph nodes and edges -----


async def _call_model(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> dict[str, Any]:
    """Coroutine to call the chat model."""
    if "configurable" not in config:
        msg = "Configuration for the model is missing."
        raise HomeAssistantError(msg)

    # Extract context for logging
    conversation_id = config.get("configurable", {}).get("thread_id", "unknown")
    run_id = config.get("configurable", {}).get("run_id", "unknown")
    user_id = config["configurable"]["user_id"]

    model = config["configurable"]["chat_model"]
    hass = config["configurable"]["hass"]
    opts = config["configurable"]["options"]
    chat_model_options = config["configurable"].get("chat_model_options", {})

    # Retrieve memories (semantic if last message is from user).
    last_message = state["messages"][-1]
    last_message_from_user = isinstance(last_message, HumanMessage)
    query_prompt = (
        EMBEDDING_MODEL_PROMPT_TEMPLATE.format(query=last_message.content)
        if last_message_from_user
        else None
    )
    mems = await store.asearch((user_id, "memories"), query=query_prompt, limit=10)

    # Recent camera activity.
    camera_activity = await _retrieve_camera_activity(hass, store)

    # Build system message.
    system_message = config["configurable"]["prompt"]
    if mems:
        formatted_mems = "\n".join(f"[{mem.key}]: {mem.value}" for mem in mems)
        system_message += f"\n<memories>\n{formatted_mems}\n</memories>"
    if camera_activity:
        ca = "\n".join(str(a) for a in camera_activity)
        system_message += f"\n<recent_camera_activity>\n{ca}\n</recent_camera_activity>"
    if summary := state.get("summary", ""):
        system_message += (
            f"\n<past_conversation_summary>\n{summary}\n</past_conversation_summary>"
        )

    # Model input = System + current messages.
    messages = [SystemMessage(content=system_message)] + state["messages"]

    # Trim messages to manage context window length.
    # TODO(goruck): Fix token counting.  # noqa: FIX002
    # If using the token counter from the chat model API, the method
    # 'get_num_tokens_from_messages()' will be called which currently ignores
    # tool schemas and under counts message tokens for the qwen models.
    # Until this is fixed, 'max_tokens' should be set to a value less than
    # the maximum size of the model's context window. See const.py.
    # https://github.com/goruck/home-generative-agent/issues/109

    provider = opts.get(CONF_CHAT_MODEL_PROVIDER)
    model_name = _determine_model_name(provider, opts)

    if CONTEXT_MANAGE_USE_TOKENS:
        max_tokens = CONTEXT_MAX_TOKENS
        token_counter = partial(
            count_tokens_cross_provider,
            model=model_name,
            provider=provider,
            options=opts,
            chat_model_options=chat_model_options,
        )
    else:
        max_tokens = CONTEXT_MAX_MESSAGES
        token_counter = len

    trimmed_messages = await hass.async_add_executor_job(
        partial(
            trim_messages,
            messages=messages,
            token_counter=token_counter,
            max_tokens=max_tokens,
            strategy="last",
            start_on="human",
            include_system=True,
        )
    )

    # Log input messages summary
    # Create full untruncated version for Loki
    loki_full_messages = "\n".join(
        [
            f"[{i}] {type(m).__name__}: {str(getattr(m, 'content', ''))}"
            for i, m in enumerate(trimmed_messages)
        ]
    )
    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"Calling model with messages:\n{format_messages_summary(trimmed_messages, title='Input messages')}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="agent",
        loki_message=f"Calling model with messages:\n{loki_full_messages}",
    )

    raw_response = await model.ainvoke(trimmed_messages)

    # Get full content for Loki (no truncation)
    raw_response_full = str(getattr(raw_response, "content", "") or "")
    if hasattr(raw_response, "tool_calls") and raw_response.tool_calls:
        raw_response_full += (
            f" [tool_calls: {json.dumps(raw_response.tool_calls, ensure_ascii=False)}]"
        )

    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"Raw model response: {format_message_for_log(raw_response, max_content_length=300)}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="agent",
        loki_message=f"Raw model response: {raw_response_full}",
    )

    response = extract_final(getattr(raw_response, "content", "") or "")

    # Create AI message, no need to include tool call metadata if there's none.
    if hasattr(raw_response, "tool_calls"):
        ai_response = AIMessage(content=response, tool_calls=raw_response.tool_calls)
    else:
        ai_response = AIMessage(content=response)

    # Get full AI response for Loki (no truncation)
    ai_response_full = str(getattr(ai_response, "content", "") or "")
    if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
        ai_response_full += (
            f" [tool_calls: {json.dumps(ai_response.tool_calls, ensure_ascii=False)}]"
        )

    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"AI response created: {format_message_for_log(ai_response, max_content_length=300)}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="agent",
        loki_message=f"AI response created: {ai_response_full}",
    )

    metadata: dict[str, str] = (
        raw_response.usage_metadata if hasattr(raw_response, "usage_metadata") else {}
    )

    if metadata:
        log_with_context(
            LOGGER,
            logging.DEBUG,
            "Token usage from metadata",
            conversation_id=conversation_id,
            run_id=run_id,
            node="agent",
            **metadata,
        )

    # Calculate messages to remove using ID-based comparison (more robust than equality)
    total_messages = len(state["messages"])
    trimmed_count = len(trimmed_messages)
    trimmed_ids = {id(m) for m in trimmed_messages}
    messages_to_remove = [m for m in state["messages"] if id(m) not in trimmed_ids]

    log_with_context(
        LOGGER,
        logging.DEBUG,
        "Message trimming",
        conversation_id=conversation_id,
        run_id=run_id,
        node="agent",
        total_messages=total_messages,
        kept_messages=trimmed_count,
        to_remove=len(messages_to_remove),
    )

    # Warning if something looks wrong
    if len(messages_to_remove) == 0 and trimmed_count < total_messages:
        log_with_context(
            LOGGER,
            logging.WARNING,
            "Trimming occurred but messages_to_remove is empty!",
            conversation_id=conversation_id,
            run_id=run_id,
            node="agent",
            total_messages=total_messages,
            trimmed_count=trimmed_count,
        )

    return {
        "messages": ai_response,
        "chat_model_usage_metadata": metadata,
        "messages_to_remove": messages_to_remove,
    }


async def _summarize_and_remove_messages(
    state: State, config: RunnableConfig
) -> dict[str, Any]:
    """Summarize trimmed messages and remove them from state."""
    if "configurable" not in config:
        msg = "Configuration is missing."
        raise HomeAssistantError(msg)

    # Extract context for logging
    conversation_id = config.get("configurable", {}).get("thread_id", "unknown")
    run_id = config.get("configurable", {}).get("run_id", "unknown")

    summary = state.get("summary", "")
    msgs_to_remove = state.get("messages_to_remove", [])

    if not msgs_to_remove:
        log_with_context(
            LOGGER,
            logging.DEBUG,
            "No messages to summarize",
            conversation_id=conversation_id,
            run_id=run_id,
            node="summarize",
        )
        return {"summary": summary}

    # Count message types before filtering
    total_to_remove = len(msgs_to_remove)
    type_counts = Counter(type(m).__name__ for m in msgs_to_remove)

    # Get configurable prompts from config, with fallbacks to defaults
    summarization_system_prompt = config["configurable"].get(
        "summarization_system_prompt", RECOMMENDED_SUMMARIZATION_SYSTEM_PROMPT
    )
    summarization_initial_prompt = config["configurable"].get(
        "summarization_initial_prompt", RECOMMENDED_SUMMARIZATION_INITIAL_PROMPT
    )
    summarization_prompt_template = config["configurable"].get(
        "summarization_prompt_template", RECOMMENDED_SUMMARIZATION_PROMPT_TEMPLATE
    )

    summary_message = (
        summarization_prompt_template.format(summary=summary)
        if summary
        else summarization_initial_prompt
    )

    # Build messages for the already-configured summarization model.
    # Only include HumanMessage and AIMessage (filter out System, Tool, etc.)
    filtered_messages = [
        m for m in msgs_to_remove if isinstance(m, (HumanMessage, AIMessage))
    ]
    messages = (
        [SystemMessage(content=summarization_system_prompt)]
        + filtered_messages
        + [HumanMessage(content=summary_message)]
    )

    # Log what's being summarized
    type_summary = ", ".join(
        f"{count} {msg_type}" for msg_type, count in sorted(type_counts.items())
    )
    # Create full untruncated version for Loki
    loki_full_summary_msgs = "\n".join(
        [
            f"[{i}] {type(m).__name__}: {str(getattr(m, 'content', ''))}"
            for i, m in enumerate(filtered_messages)
        ]
    )
    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"Summarizing messages:\n{format_messages_summary(filtered_messages, title='Messages to summarize')}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="summarize",
        total_to_remove=total_to_remove,
        filtered_for_summary=len(filtered_messages),
        types=type_summary,
        loki_message=f"Summarizing messages:\n{loki_full_summary_msgs}",
    )

    model = config["configurable"]["summarization_model"]
    raw_response = await model.ainvoke(messages)

    response = extract_final(getattr(raw_response, "content", "") or "")

    # Log summary result
    summary_preview = response[:200] + "..." if len(response) > 200 else response
    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"Summary created ({len(response)} chars): {summary_preview}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="summarize",
        messages_removed=total_to_remove,
        loki_message=f"Summary created ({len(response)} chars): {response}",
    )

    return {
        "summary": response,
        "messages": [
            RemoveMessage(id=m.id) for m in msgs_to_remove if m.id is not None
        ],
    }


async def _call_tools(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> dict[str, list[ToolMessage]]:
    """Call Home Assistant or LangChain tools requested by the model.

    Includes enhanced error handling, timeouts, and metrics collection.
    """
    if "configurable" not in config:
        msg = "Configuration is missing."
        raise HomeAssistantError(msg)

    # Extract context for logging
    conversation_id = config.get("configurable", {}).get("thread_id", "unknown")
    run_id = config.get("configurable", {}).get("run_id", "unknown")

    langchain_tools = config["configurable"]["langchain_tools"]
    ha_llm_api = config["configurable"]["ha_llm_api"]

    # Get metrics collector from config
    metrics_collector = config["configurable"].get("metrics_collector")

    # Expect tool calls in the last AIMessage.
    if not state["messages"] or not isinstance(state["messages"][-1], AIMessage):
        msg = "No tool calls found in the last message."
        raise HomeAssistantError(msg)

    tool_calls = state["messages"][-1].tool_calls or []
    tool_responses: list[ToolMessage] = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call.get("id") or ""

        # Log tool call with structured args
        args_summary = json.dumps(tool_args, ensure_ascii=False)[:200]
        if len(json.dumps(tool_args)) > 200:
            args_summary += "..."
        log_with_context(
            LOGGER,
            logging.DEBUG,
            f"Calling tool {tool_name}",
            conversation_id=conversation_id,
            run_id=run_id,
            node="action",
            tool_id=tool_id,
            args=args_summary,
        )

        # Create metric for this call
        metric = None
        if metrics_collector:
            metric = ToolCallMetrics(tool_name=tool_name, call_id=tool_id)
            metrics_collector.add_metric(metric)

        def _handle_tool_error(
            err: Exception,
            name: str,
            tid: str,
            error_type: ToolErrorType = ToolErrorType.UNKNOWN,
        ) -> ToolMessage:
            """Create error response with classification."""
            error_msg = str(err)
            log_with_context(
                LOGGER,
                logging.WARNING,
                f"Tool error in {name}",
                conversation_id=conversation_id,
                run_id=run_id,
                node="action",
                tool_name=name,
                tool_id=tid,
                error_type=error_type.value,
                error=error_msg[:200],
            )
            message = TOOL_CALL_ERROR_TEMPLATE.format(error=error_msg)
            return ToolMessage(
                content=message,
                name=name,
                tool_call_id=tid,
                status="error",
            )

        # LangChain tool
        if tool_name in langchain_tools:
            lc_tool = langchain_tools[tool_name.lower()]
            tool_call_copy = copy.deepcopy(tool_call)
            tool_call_copy["args"].update({"store": store, "config": config})
            try:
                # Execute with timeout
                tool_response = await asyncio.wait_for(
                    lc_tool.ainvoke(tool_call_copy),
                    timeout=TOOL_CALL_TIMEOUT_SECONDS,
                )
                if metric:
                    # Calculate response size
                    response_size = len(str(tool_response)) if tool_response else 0
                    metric.finalize(
                        success=True,
                        response_size_bytes=response_size,
                    )

                # Log tool response with truncation for console, full for Loki
                response_preview = str(tool_response)[:300]
                if len(str(tool_response)) > 300:
                    response_preview += "..."
                log_with_context(
                    LOGGER,
                    logging.DEBUG,
                    f"LangChain tool {tool_name} response",
                    conversation_id=conversation_id,
                    run_id=run_id,
                    node="action",
                    tool_id=tool_id,
                    response_size=len(str(tool_response)),
                    response_preview=response_preview,
                    loki_message=f"LangChain tool {tool_name} response (size={len(str(tool_response))}): {str(tool_response)}",
                )
            except asyncio.TimeoutError:
                error_type = ToolErrorType.TIMEOUT
                if metric:
                    metric.finalize(
                        success=False,
                        error_type=error_type.value,
                        error_message="Tool execution timed out",
                    )
                tool_response = _handle_tool_error(
                    Exception(
                        f"Tool '{tool_name}' timed out after"
                        f" {TOOL_CALL_TIMEOUT_SECONDS}s"
                    ),
                    tool_name,
                    tool_id,
                    error_type,
                )
            except (HomeAssistantError, ValidationError, ValueError, TypeError) as err:
                error_type = ToolErrorType.classify(err)
                if metric:
                    metric.finalize(
                        success=False,
                        error_type=error_type.value,
                        error_message=str(err),
                    )
                tool_response = _handle_tool_error(
                    err,
                    tool_name,
                    tool_id,
                    error_type,
                )
            except Exception as err:
                error_type = ToolErrorType.EXECUTION
                if metric:
                    metric.finalize(
                        success=False,
                        error_type=error_type.value,
                        error_message=str(err),
                    )
                log_with_context(
                    LOGGER,
                    logging.ERROR,
                    f"Unexpected error in LangChain tool {tool_name}",
                    conversation_id=conversation_id,
                    run_id=run_id,
                    node="action",
                    tool_id=tool_id,
                    error=str(err)[:200],
                    exc_info=True,
                )
                tool_response = _handle_tool_error(
                    err,
                    tool_name,
                    tool_id,
                    error_type,
                )
        # Home Assistant tool
        else:
            tool_input = llm.ToolInput(tool_name=tool_name, tool_args=tool_args)
            try:
                # Execute with timeout
                response = await asyncio.wait_for(
                    ha_llm_api.async_call_tool(tool_input),
                    timeout=TOOL_CALL_TIMEOUT_SECONDS,
                )
                tool_response = ToolMessage(
                    content=json.dumps(response),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                if metric:
                    response_size = len(json.dumps(response)) if response else 0
                    metric.finalize(
                        success=True,
                        response_size_bytes=response_size,
                    )

                # Log HA tool response with truncation for console, full for Loki
                # Parse JSON content for better readability
                try:
                    parsed_content = json.loads(tool_response.content)
                    content_preview = json.dumps(parsed_content, ensure_ascii=False)[
                        :300
                    ]
                    full_content = json.dumps(parsed_content, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    content_preview = str(tool_response.content)[:300]
                    full_content = str(tool_response.content)
                if len(str(tool_response.content)) > 300:
                    content_preview += "..."

                log_with_context(
                    LOGGER,
                    logging.DEBUG,
                    f"HA tool {tool_name} response",
                    conversation_id=conversation_id,
                    run_id=run_id,
                    node="action",
                    tool_id=tool_id,
                    response_size=len(str(tool_response.content)),
                    response_preview=content_preview,
                    loki_message=f"HA tool {tool_name} response (size={len(str(tool_response.content))}): {full_content}",
                )
            except asyncio.TimeoutError:
                error_type = ToolErrorType.TIMEOUT
                if metric:
                    metric.finalize(
                        success=False,
                        error_type=error_type.value,
                        error_message="Tool execution timed out",
                    )
                tool_response = _handle_tool_error(
                    Exception(
                        f"Tool '{tool_name}' timed out after"
                        f" {TOOL_CALL_TIMEOUT_SECONDS}s"
                    ),
                    tool_name,
                    tool_id,
                    error_type,
                )
            except (HomeAssistantError, vol.Invalid, ValueError, AttributeError) as err:
                error_type = ToolErrorType.classify(err)
                if metric:
                    metric.finalize(
                        success=False,
                        error_type=error_type.value,
                        error_message=str(err),
                    )
                tool_response = _handle_tool_error(
                    err,
                    tool_name,
                    tool_id,
                    error_type,
                )
            except Exception as err:
                error_type = ToolErrorType.EXECUTION
                if metric:
                    metric.finalize(
                        success=False,
                        error_type=error_type.value,
                        error_message=str(err),
                    )
                log_with_context(
                    LOGGER,
                    logging.ERROR,
                    f"Unexpected error in HA tool {tool_name}",
                    conversation_id=conversation_id,
                    run_id=run_id,
                    node="action",
                    tool_id=tool_id,
                    error=str(err)[:200],
                    exc_info=True,
                )
                tool_response = _handle_tool_error(
                    err,
                    tool_name,
                    tool_id,
                    error_type,
                )

        # Append to responses
        tool_responses.append(tool_response)

    # Log metrics summary if available
    if metrics_collector:
        summary = metrics_collector.get_summary()
        if summary["total_calls"] > 0:
            log_with_context(
                LOGGER,
                logging.DEBUG,
                "Tool execution summary",
                conversation_id=conversation_id,
                run_id=run_id,
                node="action",
                total_calls=summary["total_calls"],
                successful=summary["successful_calls"],
                failed=summary["failed_calls"],
                success_rate=f"{summary['success_rate'] * 100:.1f}%",
            )

    return {"messages": tool_responses}


async def _confirm_automation(
    state: State, config: RunnableConfig
) -> Command[Literal["action", "agent"]]:
    """Ask user to confirm automation creation before proceeding.

    This implements human-in-the-loop for safety-critical operations.
    Uses LangGraph's interrupt() to pause execution and wait for user response.
    """
    if "configurable" not in config:
        msg = "Configuration is missing."
        raise HomeAssistantError(msg)

    # Extract context for logging
    conversation_id = config.get("configurable", {}).get("thread_id", "unknown")
    run_id = config.get("configurable", {}).get("run_id", "unknown")

    messages = state["messages"]
    if not isinstance(messages[-1], AIMessage):
        # No AI message, skip confirmation
        return Command(goto="action")

    tool_calls = messages[-1].tool_calls or []

    # Check if any tool call is for add_automation
    automation_calls = [tc for tc in tool_calls if tc["name"] == "add_automation"]

    if not automation_calls:
        # No automation being created, proceed normally
        return Command(goto="action")

    # Check if these automation calls have already been executed by looking for
    # their ToolMessage results in the message history. This prevents asking for
    # confirmation twice for the same automation.
    automation_call_ids = {tc.get("id") for tc in automation_calls if tc.get("id")}
    executed_tool_ids = {
        m.tool_call_id
        for m in messages
        if isinstance(m, ToolMessage) and m.tool_call_id
    }

    # If all automation calls have already been executed, skip confirmation
    if automation_call_ids and automation_call_ids.issubset(executed_tool_ids):
        log_with_context(
            LOGGER,
            logging.DEBUG,
            "Automation tool calls already executed, skipping confirmation",
            conversation_id=conversation_id,
            run_id=run_id,
            node="confirm_automation",
            executed_ids=list(automation_call_ids),
        )
        return Command(goto="action")

    # Extract automation details for display
    automation_call = automation_calls[0]
    args = automation_call.get("args", {})

    # Format confirmation message
    if "yaml_config" in args:
        automation_preview = args["yaml_config"][:500]  # Limit preview length
        question = (
            f"Ich möchte eine Automation erstellen:\n\n"
            f"```yaml\n{automation_preview}\n```\n\n"
            f"Soll ich diese Automation erstellen? (ja/nein)"
        )
    elif "blueprint_name" in args:
        blueprint_name = args.get("blueprint_name", "")
        blueprint_inputs = args.get("blueprint_inputs", {})
        question = (
            f"Ich möchte eine Automation mit Blueprint '{blueprint_name}' erstellen.\n"
            f"Inputs: {blueprint_inputs}\n\n"
            f"Soll ich diese Automation erstellen? (ja/nein)"
        )
    else:
        question = "Soll ich eine Automation erstellen? (ja/nein)"

    # Pause execution and wait for user response
    log_with_context(
        LOGGER,
        logging.INFO,
        "Requesting user confirmation for automation creation",
        conversation_id=conversation_id,
        run_id=run_id,
        node="confirm_automation",
    )
    user_response = interrupt(question)

    # Check user response
    if user_response and str(user_response).lower().strip() in ["ja", "yes", "j", "y"]:
        log_with_context(
            LOGGER,
            logging.INFO,
            "User confirmed automation creation",
            conversation_id=conversation_id,
            run_id=run_id,
            node="confirm_automation",
        )
        return Command(goto="action")
    else:
        log_with_context(
            LOGGER,
            logging.INFO,
            "User cancelled automation creation",
            conversation_id=conversation_id,
            run_id=run_id,
            node="confirm_automation",
        )
        # Add cancellation message to state
        cancellation_msg = AIMessage(
            content="Die Automation-Erstellung wurde abgebrochen."
        )
        return Command(update={"messages": [cancellation_msg]}, goto="agent")


def _should_continue(
    state: State,
) -> Literal["confirm_automation", "summarize_and_remove_messages"]:
    """Return the next node in graph to execute."""
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "confirm_automation"
    return "summarize_and_remove_messages"


# Define a new graph
workflow = StateGraph(State)

# Define nodes.
workflow.add_node("agent", _call_model)
workflow.add_node("confirm_automation", _confirm_automation)
workflow.add_node("action", _call_tools)
workflow.add_node("summarize_and_remove_messages", _summarize_and_remove_messages)

# Define edges.
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_edge("action", "agent")
workflow.add_edge("summarize_and_remove_messages", END)
