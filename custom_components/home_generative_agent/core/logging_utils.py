"""Logging utilities for better traceability and readability."""

import asyncio
import json
import logging
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from homeassistant.util import ulid
from langchain_core.messages import AIMessage, ToolMessage

from .loki_handler import LokiHttpHandler, InfluxMetricsHandler

# Global handler instances
_loki_http_handler: Optional[LokiHttpHandler] = None
_influx_metrics_handler: Optional[InfluxMetricsHandler] = None


def create_run_id() -> str:
    """
    Generate a short run ID for tracking a single conversation invocation.

    Returns:
        str: Last 8 characters of a ULID (e.g., "xyz789ab")
    """
    return str(ulid.ulid_now())[-8:]


def format_message_for_log(message: Any, max_content_length: int = 500) -> str:
    """
    Format a single LangChain message for readable logging.

    Args:
        message: A LangChain message object (HumanMessage, AIMessage, etc.)
        max_content_length: Maximum length of content to display

    Returns:
        str: Formatted message string
    """
    try:
        msg_type = type(message).__name__

        # Extract content
        content = getattr(message, "content", "")

        # For ToolMessage, try to parse JSON content
        if isinstance(message, ToolMessage):
            tool_call_id = getattr(message, "tool_call_id", "unknown")
            tool_name = getattr(message, "name", "unknown")

            # Try to parse JSON content for better readability
            try:
                if isinstance(content, str):
                    parsed_content = json.loads(content)
                    content_preview = json.dumps(parsed_content, ensure_ascii=False)[
                        :max_content_length
                    ]
                else:
                    content_preview = str(content)[:max_content_length]
            except (json.JSONDecodeError, TypeError):
                content_preview = str(content)[:max_content_length]

            if len(str(content)) > max_content_length:
                content_preview += "..."

            return (
                f"{msg_type}[{tool_name}](id:{tool_call_id}): "
                f"{content_preview} ({len(str(content))} chars)"
            )

        # For AIMessage with tool calls
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", [])
            if tool_calls:
                tool_info = f" + {len(tool_calls)} tool_call(s)"
                if len(tool_calls) <= 3:
                    tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                    tool_info = f" + calls: {', '.join(tool_names)}"
            else:
                tool_info = ""

            content_str = str(content)
            content_preview = content_str[:max_content_length]
            if len(content_str) > max_content_length:
                content_preview += "..."

            return (
                f'{msg_type}: "{content_preview}"{tool_info} ({len(content_str)} chars)'
            )

        # For other messages
        content_str = str(content)
        content_preview = content_str[:max_content_length]
        if len(content_str) > max_content_length:
            content_preview += "..."

        return f'{msg_type}: "{content_preview}" ({len(content_str)} chars)'

    except Exception as e:
        return f"<Error formatting message: {e}>"


def format_messages_summary(
    messages: list[Any], title: str = "Messages", show_details: bool = True
) -> str:
    """
    Create a summary of a list of messages for logging.

    Args:
        messages: List of LangChain message objects
        title: Title for the summary
        show_details: Whether to show first/last message details

    Returns:
        str: Formatted summary string
    """
    if not messages:
        return f"{title}: (empty)"

    # Count message types
    type_counts = Counter(type(m).__name__ for m in messages)
    type_summary = ", ".join(
        f"{count} {msg_type.replace('Message', '').lower()}"
        for msg_type, count in sorted(type_counts.items())
    )

    # Calculate total content length (approximate tokens)
    total_chars = sum(len(str(getattr(m, "content", ""))) for m in messages)

    summary = (
        f"{title} ({len(messages)} messages: {type_summary}, ~{total_chars} chars)"
    )

    if show_details and len(messages) > 0:
        details = []

        # Show first 2 messages
        for i in range(min(2, len(messages))):
            details.append(
                f"  [{i}] {format_message_for_log(messages[i], max_content_length=200)}"
            )

        # Show ellipsis if there are more than 4 messages
        if len(messages) > 4:
            details.append(f"  ... ({len(messages) - 4} more messages)")

        # Show last 2 messages
        if len(messages) > 2:
            start_idx = max(2, len(messages) - 2)
            for i in range(start_idx, len(messages)):
                details.append(
                    f"  [{i}] {format_message_for_log(messages[i], max_content_length=200)}"
                )

        if details:
            summary += "\n" + "\n".join(details)

    return summary


def initialize_loki_logging(config: dict) -> None:
    """Initialize Loki and InfluxDB handlers based on configuration.

    Args:
        config: Configuration dictionary with Loki/InfluxDB settings
    """
    global _loki_http_handler, _influx_metrics_handler

    _LOGGER = logging.getLogger(__name__)

    # Initialize Loki handler if enabled
    if config.get("loki_enabled", False) and config.get("loki_url"):
        try:
            _loki_http_handler = LokiHttpHandler(
                loki_url=config["loki_url"],
                buffer_path=config.get(
                    "loki_buffer_path", "/config/logs/generative_agent"
                ),
                timeout=config.get("loki_timeout", 5),
                batch_size=config.get("loki_batch_size", 50),
                batch_interval=config.get("loki_batch_interval", 10),
            )

            # Start the handler
            loop = asyncio.get_event_loop()
            loop.create_task(_loki_http_handler.start())

            _LOGGER.info(
                f"Loki logging initialized: {config['loki_url']} "
                f"(batch={config.get('loki_batch_size', 50)}, "
                f"interval={config.get('loki_batch_interval', 10)}s)"
            )

        except Exception as e:
            _LOGGER.error(f"Failed to initialize Loki handler: {e}")
            _loki_http_handler = None

    # Initialize InfluxDB metrics handler if enabled
    if config.get("influx_metrics_enabled", False) and config.get("influx_url"):
        try:
            _influx_metrics_handler = InfluxMetricsHandler(
                url=config["influx_url"],
                token=config["influx_token"],
                org=config.get("influx_org", "smarthome"),
                bucket=config.get("influx_bucket", "home_assistant"),
                batch_size=config.get("influx_batch_size", 50),
                batch_interval=config.get("influx_batch_interval", 10),
            )

            # Start the handler
            loop = asyncio.get_event_loop()
            loop.create_task(_influx_metrics_handler.start())

            _LOGGER.info(
                f"InfluxDB metrics initialized: {config['influx_url']} "
                f"-> {config.get('influx_org')}/{config.get('influx_bucket')}"
            )

        except Exception as e:
            _LOGGER.error(f"Failed to initialize InfluxDB handler: {e}")
            _influx_metrics_handler = None


async def shutdown_loki_logging() -> None:
    """Shutdown Loki and InfluxDB handlers gracefully."""
    global _loki_http_handler, _influx_metrics_handler

    if _loki_http_handler:
        await _loki_http_handler.stop()
        _loki_http_handler = None

    if _influx_metrics_handler:
        await _influx_metrics_handler.stop()
        _influx_metrics_handler = None


def get_influx_handler() -> Optional[InfluxMetricsHandler]:
    """Get the global InfluxDB metrics handler.

    Returns:
        InfluxMetricsHandler instance or None if not initialized
    """
    return _influx_metrics_handler


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    conversation_id: str | None = None,
    run_id: str | None = None,
    node: str | None = None,
    loki_message: str | None = None,
    **extra_context: Any,
) -> None:
    """
    Log a message with context information for better traceability.

    Args:
        logger: Logger instance to use
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        message: Log message (may be truncated for console readability)
        conversation_id: Conversation/thread ID
        run_id: Run ID for this specific invocation
        node: Graph node name
        loki_message: Optional full untruncated message for Loki (if None, uses message)
        **extra_context: Additional context to append to message
    """
    # Build context prefix
    context_parts = []
    if conversation_id:
        # Shorten conversation_id to last 8 chars for readability
        short_conv_id = (
            conversation_id[-8:] if len(conversation_id) > 8 else conversation_id
        )
        context_parts.append(f"conv:{short_conv_id}")
    if run_id:
        context_parts.append(f"run:{run_id}")
    if node:
        context_parts.append(f"node:{node}")

    prefix = f"[{']['.join(context_parts)}] " if context_parts else ""

    # Add extra context to message if provided
    if extra_context:
        # Format extra context nicely
        context_str = ", ".join(f"{k}={v}" for k, v in extra_context.items())
        message = f"{message} ({context_str})"

    # Log with context
    logger.log(level, f"{prefix}{message}")

    # Queue log to Loki via HTTP (batched)
    # Use full untruncated message for Loki if provided
    if _loki_http_handler:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": logging.getLevelName(level).lower(),
            "message": loki_message if loki_message is not None else message,
            "conversation_id": conversation_id or "",
            "run_id": run_id or "",
            "node": node or "",
            **extra_context,
        }

        # Create labels for Loki
        labels = {
            "job": "home_generative_agent",
            "level": logging.getLevelName(level).lower(),
            "node": node or "unknown",
            "conversation_id": conversation_id or "none",
        }

        # Add provider label if available
        if "provider" in extra_context:
            labels["provider"] = str(extra_context["provider"])

        # Add tool_name label if available
        if "tool_name" in extra_context:
            labels["tool_name"] = str(extra_context["tool_name"])

        # Non-blocking queue (batched push happens in background)
        try:
            # Use create_task to avoid blocking
            loop = asyncio.get_event_loop()
            loop.create_task(_loki_http_handler.queue_log(log_entry, labels))
        except Exception:
            # Silently fail - don't break normal logging
            pass
