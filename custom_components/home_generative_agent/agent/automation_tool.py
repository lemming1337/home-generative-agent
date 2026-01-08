"""Automation creation tool for Home Generative Agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import aiofiles
import yaml
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.automation.const import DOMAIN as AUTOMATION_DOMAIN
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import SERVICE_RELOAD
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import ulid
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore  # noqa: TC002
from voluptuous import MultipleInvalid

from ..const import (  # noqa: TID252
    AUTOMATION_TOOL_BLUEPRINT_NAME,
    AUTOMATION_TOOL_EVENT_REGISTERED,
    CONF_NOTIFY_SERVICE,
)
from .entity_validator import (
    format_available_entities_hint,
    validate_entities_in_yaml,
)
from .ha_docs_manager import (
    format_docs_for_context,
    retrieve_relevant_docs,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)


@tool(parse_docstring=True)
async def add_automation(  # noqa: D417
    automation_yaml: str = "",
    time_pattern: str = "",
    message: str = "",
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    Add an automation to Home Assistant.

    You are provided a Home Assistant blueprint as part of this tool if you need it.
    You MUST ONLY use the blueprint to create automations that involve camera image
    analysis. You MUST generate Home Assistant automation YAML for everything else.
    If using the blueprint you MUST provide the arguments "time_pattern" and "message"
    and DO NOT provide the argument "automation_yaml".

    IMPORTANT - Use CURRENT Home Assistant YAML syntax (2024.8+):
    - Use 'triggers:' (plural) NOT 'trigger:'
    - Use 'conditions:' (plural) NOT 'condition:'
    - Use 'actions:' (plural) NOT 'action:'
    - Inside triggers, use 'trigger: state' NOT 'platform: state'
    - For service calls, use 'action: light.turn_on' NOT 'service: light.turn_on'
    - Use 'target:' with 'entity_id:' for specifying devices

    CORRECT YAML example (turn on light at sunset):
    ```yaml
    alias: "Turn on lights at sunset"
    description: "Turns on living room lights when the sun sets"
    triggers:
      - trigger: sun
        event: sunset
        offset: "-00:30:00"
    conditions:
      - condition: state
        entity_id: binary_sensor.someone_home
        state: "on"
    actions:
      - action: light.turn_on
        target:
          entity_id: light.living_room
        data:
          brightness_pct: 80
    ```

    CORRECT YAML example (motion-activated light):
    ```yaml
    alias: "Motion light bathroom"
    triggers:
      - trigger: state
        entity_id: binary_sensor.bathroom_motion
        to: "on"
    actions:
      - action: light.turn_on
        target:
          entity_id: light.bathroom
      - delay:
          minutes: 5
      - action: light.turn_off
        target:
          entity_id: light.bathroom
    ```

    CORRECT YAML example (send notification):
    ```yaml
    alias: "Door open notification"
    triggers:
      - trigger: state
        entity_id: binary_sensor.front_door
        to: "on"
        for:
          seconds: 30
    actions:
      - action: notify.mobile_app
        data:
          message: "Front door has been open for 30 seconds"
          title: "Door Alert"
    ```

    Args:
        automation_yaml: A Home Assistant automation in valid YAML format.
            ONLY provide if NOT using the camera image analysis blueprint.
            MUST use current syntax with triggers/conditions/actions (plural).
        time_pattern: Cron-like time pattern (e.g., /30 for "every 30 mins").
            ONLY provide if using the camera image analysis blueprint.
        message: Image analysis prompt (e.g.,"check the front porch camera for boxes")
            ONLY provide if using the camera image analysis blueprint.

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass: HomeAssistant = config["configurable"]["hass"]
    mobile_push_service = config["configurable"]["options"].get(CONF_NOTIFY_SERVICE)

    if time_pattern and message:
        automation_data = {
            "alias": message,
            "description": f"Created with blueprint {AUTOMATION_TOOL_BLUEPRINT_NAME}.",
            "use_blueprint": {
                "path": AUTOMATION_TOOL_BLUEPRINT_NAME,
                "input": {
                    "time_pattern": time_pattern,
                    "message": message,
                    "mobile_push_service": mobile_push_service or "",
                },
            },
        }
        automation_yaml = yaml.dump(automation_data)

    # Parse YAML first to check for basic syntax errors
    try:
        automation_parsed = yaml.safe_load(automation_yaml)
    except yaml.YAMLError as err:
        # Retrieve relevant docs to help fix YAML syntax
        docs = await retrieve_relevant_docs(store, "automation yaml syntax", limit=2)
        docs_context = format_docs_for_context(docs, max_length=3000)
        return (
            f"Invalid YAML syntax: {err}\n\n"
            f"Please fix the YAML syntax and try again.\n\n"
            f"{docs_context}"
        )

    ha_automation_config: dict[str, Any] = {"id": ulid.ulid_now()}
    if isinstance(automation_parsed, list):
        ha_automation_config.update(automation_parsed[0])
    if isinstance(automation_parsed, dict):
        ha_automation_config.update(automation_parsed)

    # Validate entity IDs exist in Home Assistant (skip for blueprints)
    if not (time_pattern and message):
        entity_validation = validate_entities_in_yaml(hass, automation_yaml)
        if not entity_validation.is_valid:
            # Get relevant docs based on the automation content
            query = f"automation {automation_yaml[:200]}"
            docs = await retrieve_relevant_docs(store, query, limit=2)
            docs_context = format_docs_for_context(docs, max_length=2000)

            # Get list of available entities for common domains
            entities_hint = format_available_entities_hint(
                hass,
                domains=["light", "switch", "binary_sensor", "sensor", "cover"],
                limit_per_domain=5,
            )

            error_parts = [
                "Entity validation failed:\n",
                entity_validation.error_message,
                "\n\n",
                entities_hint[:2000],  # Limit entity list size
            ]

            if docs_context:
                error_parts.extend(["\n\n", docs_context])

            return "".join(error_parts)

    # Validate with Home Assistant's automation config validator
    try:
        await _async_validate_config_item(
            hass=hass,
            config=ha_automation_config,
            raise_on_errors=True,
            warn_on_errors=False,
        )
    except (HomeAssistantError, MultipleInvalid) as err:
        # Retrieve relevant docs to help fix the error
        error_str = str(err).lower()
        if "trigger" in error_str:
            query = "trigger state time sun event"
        elif "action" in error_str or "service" in error_str:
            query = "action service call turn on target"
        elif "condition" in error_str:
            query = "condition state time numeric template"
        else:
            query = "automation yaml syntax overview"

        docs = await retrieve_relevant_docs(store, query, limit=2)
        docs_context = format_docs_for_context(docs, max_length=3000)

        error_msg = f"Invalid automation configuration: {err}\n\n"
        error_msg += "Common issues:\n"
        error_msg += "- Use 'triggers:' not 'trigger:' (plural)\n"
        error_msg += "- Use 'actions:' not 'action:' (plural)\n"
        error_msg += "- Use 'trigger: state' not 'platform: state'\n"
        error_msg += "- Use 'action: light.turn_on' not 'service: light.turn_on'\n"

        if docs_context:
            error_msg += f"\n\nRelevant documentation:\n{docs_context}"

        return error_msg

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH, encoding="utf-8"
    ) as f:
        ha_exsiting_automation_configs = await f.read()
        ha_exsiting_automations_yaml = yaml.safe_load(ha_exsiting_automation_configs)

    async with aiofiles.open(
        Path(hass.config.config_dir) / AUTOMATION_CONFIG_PATH,
        "a" if ha_exsiting_automations_yaml else "w",
        encoding="utf-8",
    ) as f:
        ha_automation_config_raw = yaml.dump(
            [ha_automation_config], allow_unicode=True, sort_keys=False
        )
        await f.write("\n" + ha_automation_config_raw)

    await hass.services.async_call(AUTOMATION_DOMAIN, SERVICE_RELOAD)
    hass.bus.async_fire(
        AUTOMATION_TOOL_EVENT_REGISTERED,
        {
            "automation_config": ha_automation_config,
            "raw_config": ha_automation_config_raw,
        },
    )

    return f"Added automation {ha_automation_config['id']}"
