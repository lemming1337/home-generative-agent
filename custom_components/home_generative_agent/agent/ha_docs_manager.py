"""Home Assistant documentation manager for RAG.

This module manages Home Assistant automation documentation in the vector store
for retrieval-augmented generation when creating automations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

LOGGER = logging.getLogger(__name__)

# Namespace for HA documentation in the vector store
HA_DOCS_NAMESPACE = ("ha_docs", "automations")

# Documentation chunks covering different automation topics
# Each chunk is designed to be self-contained and retrievable
HA_AUTOMATION_DOCS: list[dict[str, str]] = [
    {
        "id": "syntax_overview",
        "topic": "automation yaml syntax overview modern format",
        "content": """# Home Assistant Automation YAML Syntax (2024.8+)

IMPORTANT: Use the CURRENT syntax with plural keys:
- triggers: (NOT trigger:)
- conditions: (NOT condition:)
- actions: (NOT action:)

Basic structure:
```yaml
alias: "Automation Name"
description: "What this automation does"
mode: single  # single, restart, queued, or parallel
triggers:
  - trigger: <type>
    # trigger config
conditions:  # optional
  - condition: <type>
    # condition config
actions:
  - action: <service>
    # action config
```

The mode determines behavior when automation triggers while already running:
- single: Only one instance runs at a time (default)
- restart: Restart current run with new trigger
- queued: Queue additional runs
- parallel: Run multiple instances simultaneously""",
    },
    {
        "id": "trigger_state",
        "topic": "state trigger entity state change from to",
        "content": """# State Trigger

Fires when an entity's state changes.

```yaml
triggers:
  - trigger: state
    entity_id: binary_sensor.motion_detector
    to: "on"  # optional: only trigger when changing TO this state
    from: "off"  # optional: only trigger when changing FROM this state
    for:  # optional: state must be held for duration
      seconds: 30
```

Multiple entities:
```yaml
triggers:
  - trigger: state
    entity_id:
      - binary_sensor.motion_kitchen
      - binary_sensor.motion_living_room
    to: "on"
```

Attribute changes:
```yaml
triggers:
  - trigger: state
    entity_id: media_player.living_room
    attribute: media_title
```

NOT conditions (trigger when state changes to anything EXCEPT):
```yaml
triggers:
  - trigger: state
    entity_id: vacuum.robot
    not_to: "docked"
```""",
    },
    {
        "id": "trigger_time",
        "topic": "time trigger schedule cron pattern daily weekly",
        "content": """# Time Triggers

## Fixed time trigger:
```yaml
triggers:
  - trigger: time
    at: "07:30:00"
```

## Time pattern (cron-like):
```yaml
triggers:
  - trigger: time_pattern
    minutes: "/5"  # every 5 minutes
```

```yaml
triggers:
  - trigger: time_pattern
    hours: "8,12,18"  # at 8:00, 12:00, 18:00
    minutes: "0"
```

## Input datetime entity:
```yaml
triggers:
  - trigger: time
    at: input_datetime.alarm_time
```

## Multiple times:
```yaml
triggers:
  - trigger: time
    at:
      - "06:00:00"
      - "18:00:00"
```""",
    },
    {
        "id": "trigger_sun",
        "topic": "sun trigger sunrise sunset offset",
        "content": """# Sun Trigger

Fires at sunrise or sunset, with optional offset.

```yaml
triggers:
  - trigger: sun
    event: sunset
    offset: "-00:30:00"  # 30 minutes before sunset
```

```yaml
triggers:
  - trigger: sun
    event: sunrise
    offset: "01:00:00"  # 1 hour after sunrise
```

Offset format: "HH:MM:SS" or "-HH:MM:SS" for before the event.""",
    },
    {
        "id": "trigger_numeric_state",
        "topic": "numeric state trigger above below threshold temperature humidity",
        "content": """# Numeric State Trigger

Fires when a numeric state crosses a threshold.

```yaml
triggers:
  - trigger: numeric_state
    entity_id: sensor.temperature
    above: 25  # triggers when going above 25
```

```yaml
triggers:
  - trigger: numeric_state
    entity_id: sensor.humidity
    below: 30
    for:
      minutes: 10  # must stay below for 10 minutes
```

Range trigger:
```yaml
triggers:
  - trigger: numeric_state
    entity_id: sensor.battery_level
    above: 20
    below: 80
```

With attribute:
```yaml
triggers:
  - trigger: numeric_state
    entity_id: climate.living_room
    attribute: current_temperature
    above: 24
```""",
    },
    {
        "id": "trigger_template",
        "topic": "template trigger jinja2 complex condition",
        "content": """# Template Trigger

Fires when a template evaluates to true.

```yaml
triggers:
  - trigger: template
    value_template: "{{ states('sensor.temperature') | float > 25 }}"
```

Complex example:
```yaml
triggers:
  - trigger: template
    value_template: >
      {{ states('binary_sensor.motion') == 'on' and
         states('sun.sun') == 'below_horizon' and
         states('input_boolean.guest_mode') == 'off' }}
```

With for duration:
```yaml
triggers:
  - trigger: template
    value_template: "{{ states('sensor.power') | float > 1000 }}"
    for:
      minutes: 5
```""",
    },
    {
        "id": "trigger_event",
        "topic": "event trigger home assistant event custom",
        "content": """# Event Trigger

Fires when a specific event occurs.

```yaml
triggers:
  - trigger: event
    event_type: call_service
    event_data:
      domain: light
      service: turn_on
```

Button press example (Zigbee/Z-Wave):
```yaml
triggers:
  - trigger: event
    event_type: zha_event
    event_data:
      device_id: abc123
      command: "on"
```

Mobile app notification action:
```yaml
triggers:
  - trigger: event
    event_type: mobile_app_notification_action
    event_data:
      action: "CONFIRM_ACTION"
```""",
    },
    {
        "id": "trigger_device",
        "topic": "device trigger button press motion device automation",
        "content": """# Device Trigger

Triggers based on device-specific events (UI-generated, but can be written manually).

```yaml
triggers:
  - trigger: device
    device_id: "abc123def456"
    domain: zha
    type: remote_button_short_press
    subtype: button_1
```

Note: Device triggers are typically created through the UI as they require
specific device_id values. Use state triggers for most entity-based automations.""",
    },
    {
        "id": "conditions_overview",
        "topic": "condition state time numeric template zone",
        "content": """# Conditions Overview

Conditions are optional filters that must be true for actions to run.
By default, ALL conditions must be true (AND logic).

## State condition:
```yaml
conditions:
  - condition: state
    entity_id: binary_sensor.someone_home
    state: "on"
```

## Time condition:
```yaml
conditions:
  - condition: time
    after: "22:00:00"
    before: "06:00:00"
    weekday:
      - mon
      - tue
      - wed
      - thu
      - fri
```

## Numeric state condition:
```yaml
conditions:
  - condition: numeric_state
    entity_id: sensor.temperature
    above: 18
    below: 26
```

## Template condition:
```yaml
conditions:
  - condition: template
    value_template: "{{ states('input_boolean.vacation_mode') == 'off' }}"
```

## OR logic (any condition true):
```yaml
conditions:
  - condition: or
    conditions:
      - condition: state
        entity_id: person.john
        state: "home"
      - condition: state
        entity_id: person.jane
        state: "home"
```""",
    },
    {
        "id": "actions_service",
        "topic": "action service call turn on off toggle target entity",
        "content": """# Service Call Actions

IMPORTANT: Use 'action:' NOT 'service:' in modern syntax.

## Basic service call:
```yaml
actions:
  - action: light.turn_on
    target:
      entity_id: light.living_room
```

## With data parameters:
```yaml
actions:
  - action: light.turn_on
    target:
      entity_id: light.living_room
    data:
      brightness_pct: 80
      color_temp_kelvin: 3000
```

## Multiple targets:
```yaml
actions:
  - action: light.turn_off
    target:
      entity_id:
        - light.bedroom
        - light.bathroom
        - light.kitchen
```

## Target by area or label:
```yaml
actions:
  - action: light.turn_on
    target:
      area_id: living_room
```

## Common services:
- homeassistant.turn_on / turn_off / toggle
- light.turn_on / turn_off (with brightness, color options)
- switch.turn_on / turn_off
- cover.open_cover / close_cover / set_cover_position
- climate.set_temperature / set_hvac_mode
- media_player.play_media / volume_set
- notify.<service_name> (for notifications)
- script.turn_on / script.<script_name>""",
    },
    {
        "id": "actions_notify",
        "topic": "notification notify mobile app push message alert",
        "content": """# Notification Actions

## Mobile app notification:
```yaml
actions:
  - action: notify.mobile_app_<phone_name>
    data:
      message: "Motion detected at front door"
      title: "Security Alert"
```

## With image (e.g., camera snapshot):
```yaml
actions:
  - action: notify.mobile_app_phone
    data:
      message: "Someone at the door"
      data:
        image: /api/camera_proxy/camera.front_door
```

## Actionable notification:
```yaml
actions:
  - action: notify.mobile_app_phone
    data:
      message: "Garage door is open"
      data:
        actions:
          - action: "CLOSE_GARAGE"
            title: "Close Garage"
          - action: "IGNORE"
            title: "Ignore"
```

## Persistent notification (shows in HA UI):
```yaml
actions:
  - action: persistent_notification.create
    data:
      message: "Dishwasher cycle complete"
      title: "Appliance"
```""",
    },
    {
        "id": "actions_delay_wait",
        "topic": "delay wait template pause between actions",
        "content": """# Delay and Wait Actions

## Fixed delay:
```yaml
actions:
  - action: light.turn_on
    target:
      entity_id: light.porch
  - delay:
      minutes: 10
  - action: light.turn_off
    target:
      entity_id: light.porch
```

## Delay formats:
```yaml
- delay: "00:05:00"  # 5 minutes
- delay:
    hours: 1
    minutes: 30
    seconds: 0
- delay: "{{ states('input_number.delay_minutes') | int * 60 }}"  # dynamic
```

## Wait for trigger:
```yaml
actions:
  - action: light.turn_on
    target:
      entity_id: light.bathroom
  - wait_for_trigger:
      - trigger: state
        entity_id: binary_sensor.bathroom_motion
        to: "off"
        for:
          minutes: 5
    timeout:
      minutes: 30
  - action: light.turn_off
    target:
      entity_id: light.bathroom
```

## Wait for template:
```yaml
actions:
  - wait_template: "{{ states('sensor.washing_machine_power') | float < 5 }}"
    timeout: "02:00:00"
```""",
    },
    {
        "id": "actions_conditional",
        "topic": "choose if else conditional action branch",
        "content": """# Conditional Actions (Choose/If)

## Choose (if/elif/else):
```yaml
actions:
  - choose:
      - conditions:
          - condition: state
            entity_id: binary_sensor.night_mode
            state: "on"
        sequence:
          - action: light.turn_on
            target:
              entity_id: light.bedroom
            data:
              brightness_pct: 20
      - conditions:
          - condition: numeric_state
            entity_id: sensor.illuminance
            below: 100
        sequence:
          - action: light.turn_on
            target:
              entity_id: light.bedroom
            data:
              brightness_pct: 80
    default:
      - action: light.turn_off
        target:
          entity_id: light.bedroom
```

## If/Then:
```yaml
actions:
  - if:
      - condition: state
        entity_id: input_boolean.guest_mode
        state: "on"
    then:
      - action: notify.mobile_app
        data:
          message: "Guest mode is active"
    else:
      - action: script.normal_routine
```""",
    },
    {
        "id": "actions_repeat",
        "topic": "repeat loop count while until sequence",
        "content": """# Repeat Actions (Loops)

## Repeat count:
```yaml
actions:
  - repeat:
      count: 3
      sequence:
        - action: light.toggle
          target:
            entity_id: light.alert
        - delay:
            seconds: 1
```

## Repeat while:
```yaml
actions:
  - repeat:
      while:
        - condition: state
          entity_id: input_boolean.alarm_active
          state: "on"
      sequence:
        - action: media_player.play_media
          target:
            entity_id: media_player.speaker
          data:
            media_content_id: /local/alarm.mp3
            media_content_type: music
        - delay:
            seconds: 30
```

## Repeat until:
```yaml
actions:
  - repeat:
      until:
        - condition: state
          entity_id: cover.garage
          state: "closed"
      sequence:
        - action: cover.close_cover
          target:
            entity_id: cover.garage
        - delay:
            seconds: 5
```""",
    },
    {
        "id": "variables_templates",
        "topic": "variables templates jinja2 dynamic values",
        "content": """# Variables and Templates

## Define variables:
```yaml
variables:
  light_brightness: 80
  notification_target: notify.mobile_app_phone

triggers:
  - trigger: state
    entity_id: binary_sensor.motion
    to: "on"
actions:
  - action: light.turn_on
    target:
      entity_id: light.living_room
    data:
      brightness_pct: "{{ light_brightness }}"
```

## Template in trigger data:
```yaml
actions:
  - action: notify.mobile_app
    data:
      message: >
        Temperature is {{ states('sensor.temperature') }}°C.
        {% if states('sensor.temperature') | float > 25 %}
        It's warm today!
        {% endif %}
```

## Using trigger variables:
```yaml
triggers:
  - trigger: state
    entity_id: binary_sensor.door
    to: "on"
    id: door_opened
actions:
  - action: notify.mobile_app
    data:
      message: "{{ trigger.to_state.attributes.friendly_name }} was opened"
```""",
    },
    {
        "id": "common_patterns",
        "topic": "common automation patterns examples motion light presence",
        "content": """# Common Automation Patterns

## Motion-activated light with timeout:
```yaml
alias: "Motion Light - Kitchen"
triggers:
  - trigger: state
    entity_id: binary_sensor.kitchen_motion
    to: "on"
actions:
  - action: light.turn_on
    target:
      entity_id: light.kitchen
  - wait_for_trigger:
      - trigger: state
        entity_id: binary_sensor.kitchen_motion
        to: "off"
        for:
          minutes: 5
  - action: light.turn_off
    target:
      entity_id: light.kitchen
mode: restart  # restart if motion detected again
```

## Welcome home (presence-based):
```yaml
alias: "Welcome Home"
triggers:
  - trigger: state
    entity_id: person.john
    to: "home"
conditions:
  - condition: sun
    after: sunset
actions:
  - action: light.turn_on
    target:
      area_id: entrance
  - action: climate.set_temperature
    target:
      entity_id: climate.thermostat
    data:
      temperature: 21
```

## Good night routine:
```yaml
alias: "Good Night"
triggers:
  - trigger: state
    entity_id: input_boolean.bedtime
    to: "on"
actions:
  - action: light.turn_off
    target:
      area_id:
        - living_room
        - kitchen
  - action: cover.close_cover
    target:
      entity_id: cover.all_blinds
  - action: lock.lock
    target:
      entity_id: lock.front_door
  - action: input_boolean.turn_off
    target:
      entity_id: input_boolean.bedtime
```""",
    },
]


async def populate_ha_docs(store: BaseStore) -> int:
    """Populate the vector store with Home Assistant documentation.

    Args:
        store: The LangGraph BaseStore instance

    Returns:
        Number of documents stored
    """
    stored_count = 0

    for doc in HA_AUTOMATION_DOCS:
        try:
            await store.aput(
                namespace=HA_DOCS_NAMESPACE,
                key=doc["id"],
                value={
                    "topic": doc["topic"],
                    "content": doc["content"],
                },
            )
            stored_count += 1
            LOGGER.debug("Stored HA doc: %s", doc["id"])
        except Exception as err:
            LOGGER.warning("Failed to store HA doc %s: %s", doc["id"], err)

    LOGGER.info("Populated %d HA automation documentation chunks", stored_count)
    return stored_count


async def retrieve_relevant_docs(
    store: BaseStore,
    query: str,
    limit: int = 3,
) -> list[str]:
    """Retrieve relevant HA documentation based on query.

    Args:
        store: The LangGraph BaseStore instance
        query: The search query (e.g., user's automation request)
        limit: Maximum number of documents to retrieve

    Returns:
        List of relevant documentation content strings
    """
    try:
        results = await store.asearch(
            namespace=HA_DOCS_NAMESPACE,
            query=query,
            limit=limit,
        )

        if not results:
            LOGGER.debug("No relevant HA docs found for query: %s", query[:100])
            return []

        docs = []
        for result in results:
            content = result.value.get("content", "")
            if content:
                docs.append(content)

        LOGGER.debug(
            "Retrieved %d relevant HA docs for query: %s",
            len(docs),
            query[:100],
        )
        return docs

    except Exception as err:
        LOGGER.warning("Failed to retrieve HA docs: %s", err)
        return []


async def check_docs_populated(store: BaseStore) -> bool:
    """Check if HA docs have been populated in the store.

    Args:
        store: The LangGraph BaseStore instance

    Returns:
        True if docs exist, False otherwise
    """
    try:
        results = await store.asearch(
            namespace=HA_DOCS_NAMESPACE,
            query="automation",
            limit=1,
        )
        return len(results) > 0
    except Exception:
        return False


def format_docs_for_context(docs: list[str], max_length: int = 8000) -> str:
    """Format retrieved docs for inclusion in LLM context.

    Args:
        docs: List of documentation content strings
        max_length: Maximum total length of formatted output

    Returns:
        Formatted documentation string
    """
    if not docs:
        return ""

    formatted = "# Relevant Home Assistant Documentation\n\n"
    current_length = len(formatted)

    for doc in docs:
        if current_length + len(doc) + 10 > max_length:
            # Truncate if we'd exceed max length
            remaining = max_length - current_length - 20
            if remaining > 100:
                formatted += doc[:remaining] + "\n...(truncated)"
            break
        formatted += doc + "\n\n---\n\n"
        current_length = len(formatted)

    return formatted.strip()
