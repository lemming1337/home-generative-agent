"""Entity validation utilities for Home Generative Agent.

This module provides functions to validate that entity IDs referenced in
automation YAML actually exist in Home Assistant, and suggests alternatives
for invalid entities using fuzzy matching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

# Patterns to find entity_id references in YAML
ENTITY_ID_PATTERNS = [
    r"entity_id:\s*['\"]?([a-z_]+\.[a-z0-9_]+)['\"]?",
    r"entity_id:\s*\n\s*-\s*['\"]?([a-z_]+\.[a-z0-9_]+)['\"]?",
    r"target:\s*\n\s*entity_id:\s*['\"]?([a-z_]+\.[a-z0-9_]+)['\"]?",
]


@dataclass
class EntityValidationResult:
    """Result of entity validation."""

    is_valid: bool
    invalid_entities: list[str]
    suggestions: dict[str, list[str]]  # invalid_entity -> list of suggestions
    error_message: str


def _extract_entity_ids_from_yaml(yaml_content: str) -> set[str]:
    """Extract all entity_id references from YAML content.

    Args:
        yaml_content: The YAML string to parse

    Returns:
        Set of entity_id strings found in the YAML
    """
    entity_ids: set[str] = set()

    # Try parsing YAML first for structured extraction
    try:
        parsed = yaml.safe_load(yaml_content)
        if parsed:
            _extract_from_dict(parsed, entity_ids)
    except yaml.YAMLError:
        LOGGER.debug("Could not parse YAML, falling back to regex extraction")

    # Also use regex patterns to catch any we might have missed
    for pattern in ENTITY_ID_PATTERNS:
        matches = re.findall(pattern, yaml_content, re.MULTILINE | re.IGNORECASE)
        entity_ids.update(matches)

    # Additional regex for multi-line entity_id lists
    list_pattern = r"entity_id:\s*\n((?:\s*-\s*['\"]?[a-z_]+\.[a-z0-9_]+['\"]?\s*\n?)+)"
    list_matches = re.findall(list_pattern, yaml_content, re.MULTILINE | re.IGNORECASE)
    for match in list_matches:
        items = re.findall(r"['\"]?([a-z_]+\.[a-z0-9_]+)['\"]?", match)
        entity_ids.update(items)

    return entity_ids


def _extract_from_dict(data: Any, entity_ids: set[str]) -> None:
    """Recursively extract entity_ids from a parsed YAML structure.

    Args:
        data: The parsed YAML data (dict, list, or primitive)
        entity_ids: Set to add found entity_ids to
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "entity_id":
                if isinstance(value, str):
                    entity_ids.add(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            entity_ids.add(item)
            else:
                _extract_from_dict(value, entity_ids)
    elif isinstance(data, list):
        for item in data:
            _extract_from_dict(item, entity_ids)


def _calculate_similarity(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio between 0 and 1
    """
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _find_similar_entities(
    invalid_entity: str,
    valid_entities: list[str],
    max_suggestions: int = 3,
    min_similarity: float = 0.4,
) -> list[str]:
    """Find similar valid entities for an invalid entity ID.

    Args:
        invalid_entity: The invalid entity ID
        valid_entities: List of valid entity IDs in Home Assistant
        max_suggestions: Maximum number of suggestions to return
        min_similarity: Minimum similarity score to include

    Returns:
        List of similar valid entity IDs
    """
    # Extract domain from invalid entity
    invalid_domain = invalid_entity.split(".")[0] if "." in invalid_entity else ""

    suggestions: list[tuple[str, float]] = []

    for valid_entity in valid_entities:
        valid_domain = valid_entity.split(".")[0] if "." in valid_entity else ""

        # Prioritize same-domain matches
        if invalid_domain and valid_domain == invalid_domain:
            # Compare just the entity name part for same-domain entities
            invalid_name = invalid_entity.split(".")[-1]
            valid_name = valid_entity.split(".")[-1]
            similarity = _calculate_similarity(invalid_name, valid_name)
            # Boost same-domain matches
            similarity = min(1.0, similarity + 0.2)
        else:
            similarity = _calculate_similarity(invalid_entity, valid_entity)

        if similarity >= min_similarity:
            suggestions.append((valid_entity, similarity))

    # Sort by similarity descending and return top matches
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return [entity for entity, _ in suggestions[:max_suggestions]]


def validate_entities_in_yaml(
    hass: HomeAssistant,
    yaml_content: str,
) -> EntityValidationResult:
    """Validate that all entity_ids in YAML exist in Home Assistant.

    Args:
        hass: Home Assistant instance
        yaml_content: The automation YAML to validate

    Returns:
        EntityValidationResult with validation status and suggestions
    """
    # Extract entity IDs from YAML
    referenced_entities = _extract_entity_ids_from_yaml(yaml_content)

    if not referenced_entities:
        return EntityValidationResult(
            is_valid=True,
            invalid_entities=[],
            suggestions={},
            error_message="",
        )

    # Get all valid entity IDs from Home Assistant
    valid_entities = list(hass.states.async_entity_ids())

    # Find invalid entities
    invalid_entities: list[str] = []
    suggestions: dict[str, list[str]] = {}

    for entity_id in referenced_entities:
        if entity_id not in valid_entities:
            invalid_entities.append(entity_id)
            # Find similar valid entities
            similar = _find_similar_entities(entity_id, valid_entities)
            if similar:
                suggestions[entity_id] = similar

    if not invalid_entities:
        return EntityValidationResult(
            is_valid=True,
            invalid_entities=[],
            suggestions={},
            error_message="",
        )

    # Build error message
    error_parts = ["The following entity IDs do not exist in Home Assistant:\n"]
    for entity in invalid_entities:
        error_parts.append(f"  - {entity}")
        if entity in suggestions:
            error_parts.append(f"    Did you mean: {', '.join(suggestions[entity])}?")

    error_parts.append(
        "\nPlease use valid entity IDs. You can use the GetLiveContext tool "
        "to see available entities."
    )

    return EntityValidationResult(
        is_valid=False,
        invalid_entities=invalid_entities,
        suggestions=suggestions,
        error_message="\n".join(error_parts),
    )


def get_entities_by_domain(
    hass: HomeAssistant,
    domain: str,
    limit: int = 20,
) -> list[str]:
    """Get entity IDs for a specific domain.

    Args:
        hass: Home Assistant instance
        domain: The domain to filter by (e.g., "light", "switch", "sensor")
        limit: Maximum number of entities to return

    Returns:
        List of entity IDs in the specified domain
    """
    all_entities = hass.states.async_entity_ids()
    domain_entities = [e for e in all_entities if e.startswith(f"{domain}.")]
    return domain_entities[:limit]


def format_available_entities_hint(
    hass: HomeAssistant,
    domains: list[str] | None = None,
    limit_per_domain: int = 10,
) -> str:
    """Format a hint showing available entities for common domains.

    Args:
        hass: Home Assistant instance
        domains: Specific domains to include (default: common automation domains)
        limit_per_domain: Max entities per domain

    Returns:
        Formatted string listing available entities
    """
    if domains is None:
        domains = [
            "light",
            "switch",
            "binary_sensor",
            "sensor",
            "cover",
            "climate",
            "media_player",
            "automation",
            "script",
            "scene",
            "input_boolean",
            "input_number",
            "person",
        ]

    parts = ["Available entities by domain:\n"]

    for domain in domains:
        entities = get_entities_by_domain(hass, domain, limit_per_domain)
        if entities:
            parts.append(f"\n{domain}:")
            for entity in entities:
                parts.append(f"  - {entity}")
            if len(entities) == limit_per_domain:
                all_count = len(
                    [e for e in hass.states.async_entity_ids() if e.startswith(domain)]
                )
                if all_count > limit_per_domain:
                    parts.append(f"  ... and {all_count - limit_per_domain} more")

    return "\n".join(parts)
