"""Set up one ImageEntity per discovered camera."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED

from .core.image_entity import LastEventImage
from .core.utils import setup_camera_platform

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


async def async_setup_entry(
    hass: HomeAssistant,
    entry: Any,  # noqa: ARG001
    async_add_entities: Any,
) -> None:
    """Set up one ImageEntity per discovered camera."""
    on_started = setup_camera_platform(hass, async_add_entities, LastEventImage)
    if on_started:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, on_started)
