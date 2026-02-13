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
    from collections.abc import Awaitable

    from homeassistant.core import HomeAssistant
    from langchain_core.runnables import RunnableConfig
    from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)

# Maps provider names to their model configuration keys.
_PROVIDER_MODEL_KEYS: dict[str, str] = {
    "openai": CONF_OPENAI_CHAT_MODEL,
    "gemini": CONF_GEMINI_CHAT_MODEL,
}


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
    ) -> ToolErrorType:
        """Classify error type based on exception."""
        if timeout_exceeded:
            return cls.TIMEOUT
        if isinstance(error, (ValidationError, ValueError, TypeError, KeyError)):
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


def _require_config(config: RunnableConfig) -> dict[str, Any]:
    """Extract and validate the configurable dict, raising on absence."""
    if "configurable" not in config:
        msg = "Configuration is missing."
        raise HomeAssistantError(msg)
    return config["configurable"]


def _log_ctx(config: RunnableConfig) -> tuple[str, str]:
    """Extract (conversation_id, run_id) for logging."""
    cfg = config.get("configurable", {})
    return cfg.get("thread_id", "unknown"), cfg.get("run_id", "unknown")


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate text with ellipsis if over max_len."""
    s = str(text)
    return s[:max_len] + "..." if len(s) > max_len else s


def _format_messages_full(messages: list[AnyMessage]) -> str:
    """Format a message list for full (Loki) logging — no truncation."""
    return "\n".join(
        f"[{i}] {type(m).__name__}: {str(getattr(m, 'content', ''))}"
        for i, m in enumerate(messages)
    )


def _format_ai_message_full(msg: AIMessage) -> str:
    """Format an AI message for full (Loki) logging."""
    content = str(getattr(msg, "content", "") or "")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        content += f" [tool_calls: {json.dumps(msg.tool_calls, ensure_ascii=False)}]"
    return content


def _determine_model_name(provider: str, opts: dict[str, Any]) -> str:
    """Determine model name based on provider and options."""
    key = _PROVIDER_MODEL_KEYS.get(provider, CONF_OLLAMA_CHAT_MODEL)
    return opts.get(key, "")


def _build_system_message(
    prompt: str,
    memories: list,
    camera_activity: list[dict[str, dict[str, str]]],
    summary: str,
) -> str:
    """Assemble the system message from prompt, memories, camera activity, and summary."""
    parts = [prompt]
    if memories:
        formatted = "\n".join(f"[{mem.key}]: {mem.value}" for mem in memories)
        parts.append(f"<memories>\n{formatted}\n</memories>")
    if camera_activity:
        ca = "\n".join(str(a) for a in camera_activity)
        parts.append(f"<recent_camera_activity>\n{ca}\n</recent_camera_activity>")
    if summary:
        parts.append(
            f"<past_conversation_summary>\n{summary}\n</past_conversation_summary>"
        )
    return "\n".join(parts)


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


# ----- Tool execution helpers -----


def _create_tool_error_message(
    err: Exception,
    tool_name: str,
    tool_id: str,
    error_type: ToolErrorType,
    conversation_id: str,
    run_id: str,
) -> ToolMessage:
    """Create an error ToolMessage with classification and logging."""
    log_with_context(
        LOGGER,
        logging.WARNING,
        f"Tool error in {tool_name}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="action",
        tool_name=tool_name,
        tool_id=tool_id,
        error_type=error_type.value,
        error=_truncate(str(err), 200),
    )
    return ToolMessage(
        content=TOOL_CALL_ERROR_TEMPLATE.format(error=str(err)),
        name=tool_name,
        tool_call_id=tool_id,
        status="error",
    )


async def _execute_tool_call(
    coro: Awaitable,
    *,
    tool_name: str,
    tool_id: str,
    metric: ToolCallMetrics | None,
    conversation_id: str,
    run_id: str,
    known_exceptions: tuple[type[Exception], ...] = (),
) -> Any:
    """Execute a tool call with timeout and unified error handling.

    Returns the raw result on success, or a ToolMessage (status="error") on failure.
    """
    try:
        return await asyncio.wait_for(coro, timeout=TOOL_CALL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        err: Exception = Exception(
            f"Tool '{tool_name}' timed out after {TOOL_CALL_TIMEOUT_SECONDS}s"
        )
        error_type = ToolErrorType.TIMEOUT
    except Exception as exc:
        err = exc
        if isinstance(exc, known_exceptions):
            error_type = ToolErrorType.classify(exc)
        else:
            error_type = ToolErrorType.EXECUTION
            log_with_context(
                LOGGER,
                logging.ERROR,
                f"Unexpected error in tool {tool_name}",
                conversation_id=conversation_id,
                run_id=run_id,
                node="action",
                tool_id=tool_id,
                error=_truncate(str(exc), 200),
                exc_info=True,
            )
    if metric:
        metric.finalize(
            success=False,
            error_type=error_type.value,
            error_message=str(err),
        )
    return _create_tool_error_message(
        err, tool_name, tool_id, error_type, conversation_id, run_id
    )


def _log_tool_response(
    tool_name: str,
    tool_id: str,
    tool_type: str,
    content: str,
    conversation_id: str,
    run_id: str,
) -> None:
    """Log a successful tool response (truncated for console, full for Loki)."""
    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"{tool_type} tool {tool_name} response",
        conversation_id=conversation_id,
        run_id=run_id,
        node="action",
        tool_id=tool_id,
        response_size=len(content),
        response_preview=_truncate(content),
        loki_message=(
            f"{tool_type} tool {tool_name} response (size={len(content)}): {content}"
        ),
    )


def _is_tool_error(result: Any) -> bool:
    """Check if a tool execution result is an error ToolMessage."""
    return isinstance(result, ToolMessage) and result.status == "error"


# ----- Graph nodes and edges -----


async def _call_model(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> dict[str, Any]:
    """Coroutine to call the chat model."""
    cfg = _require_config(config)
    conversation_id, run_id = _log_ctx(config)

    model = cfg["chat_model"]
    hass = cfg["hass"]
    opts = cfg["options"]
    chat_model_options = cfg.get("chat_model_options", {})
    user_id = cfg["user_id"]

    # Retrieve memories (semantic if last message is from user).
    last_message = state["messages"][-1]
    query_prompt = (
        EMBEDDING_MODEL_PROMPT_TEMPLATE.format(query=last_message.content)
        if isinstance(last_message, HumanMessage)
        else None
    )
    memories = await store.asearch((user_id, "memories"), query=query_prompt, limit=10)

    # Build system message with all context.
    system_message = _build_system_message(
        prompt=cfg["prompt"],
        memories=memories,
        camera_activity=await _retrieve_camera_activity(hass, store),
        summary=state.get("summary", ""),
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
    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"Calling model with messages:\n{format_messages_summary(trimmed_messages, title='Input messages')}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="agent",
        loki_message=f"Calling model with messages:\n{_format_messages_full(trimmed_messages)}",
    )

    raw_response = await model.ainvoke(trimmed_messages)

    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"Raw model response: {format_message_for_log(raw_response, max_content_length=300)}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="agent",
        loki_message=f"Raw model response: {_format_ai_message_full(raw_response)}",
    )

    response = extract_final(getattr(raw_response, "content", "") or "")

    # Create AI message, preserving tool calls if present.
    ai_response = (
        AIMessage(content=response, tool_calls=raw_response.tool_calls)
        if hasattr(raw_response, "tool_calls")
        else AIMessage(content=response)
    )

    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"AI response created: {format_message_for_log(ai_response, max_content_length=300)}",
        conversation_id=conversation_id,
        run_id=run_id,
        node="agent",
        loki_message=f"AI response created: {_format_ai_message_full(ai_response)}",
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
    cfg = _require_config(config)
    conversation_id, run_id = _log_ctx(config)

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

    # Get configurable prompts with fallbacks to defaults
    summarization_system_prompt = cfg.get(
        "summarization_system_prompt", RECOMMENDED_SUMMARIZATION_SYSTEM_PROMPT
    )
    summarization_initial_prompt = cfg.get(
        "summarization_initial_prompt", RECOMMENDED_SUMMARIZATION_INITIAL_PROMPT
    )
    summarization_prompt_template = cfg.get(
        "summarization_prompt_template", RECOMMENDED_SUMMARIZATION_PROMPT_TEMPLATE
    )

    summary_message = (
        summarization_prompt_template.format(summary=summary)
        if summary
        else summarization_initial_prompt
    )

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
        loki_message=f"Summarizing messages:\n{_format_messages_full(filtered_messages)}",
    )

    model = cfg["summarization_model"]
    raw_response = await model.ainvoke(messages)

    response = extract_final(getattr(raw_response, "content", "") or "")

    # Log summary result
    log_with_context(
        LOGGER,
        logging.DEBUG,
        f"Summary created ({len(response)} chars): {_truncate(response, 200)}",
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

    Includes unified error handling, timeouts, and metrics collection.
    """
    cfg = _require_config(config)
    conversation_id, run_id = _log_ctx(config)

    langchain_tools = cfg["langchain_tools"]
    ha_llm_api = cfg["ha_llm_api"]
    metrics_collector = cfg.get("metrics_collector")

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
        args_summary = _truncate(json.dumps(tool_args, ensure_ascii=False), 200)
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

        # Prepare invocation based on tool type
        if tool_name in langchain_tools:
            lc_tool = langchain_tools[tool_name.lower()]
            tool_call_copy = copy.deepcopy(tool_call)
            tool_call_copy["args"].update({"store": store, "config": config})
            coro = lc_tool.ainvoke(tool_call_copy)
            known_exceptions = (
                HomeAssistantError,
                ValidationError,
                ValueError,
                TypeError,
            )
            tool_type = "LangChain"
        else:
            tool_input = llm.ToolInput(tool_name=tool_name, tool_args=tool_args)
            coro = ha_llm_api.async_call_tool(tool_input)
            known_exceptions = (
                HomeAssistantError,
                vol.Invalid,
                ValueError,
                AttributeError,
            )
            tool_type = "HA"

        # Execute with unified timeout and error handling
        result = await _execute_tool_call(
            coro,
            tool_name=tool_name,
            tool_id=tool_id,
            metric=metric,
            conversation_id=conversation_id,
            run_id=run_id,
            known_exceptions=known_exceptions,
        )

        if _is_tool_error(result):
            tool_response = result
        else:
            # Build ToolMessage and log on success
            if tool_type == "HA":
                tool_response = ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                log_content = json.dumps(result, ensure_ascii=False)
            else:
                tool_response = result
                log_content = str(tool_response)

            if metric:
                metric.finalize(success=True, response_size_bytes=len(log_content))

            _log_tool_response(
                tool_name,
                tool_id,
                tool_type,
                log_content,
                conversation_id,
                run_id,
            )

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
    _require_config(config)
    conversation_id, run_id = _log_ctx(config)

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
