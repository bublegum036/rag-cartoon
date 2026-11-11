"""Versioned archive representation; SHA-256 still binds exact JSON values."""

from typing import Any


def archive_data(value: Any, schema_version: int) -> Any:
    """Recreate v1/v2's filters, which predate the optional year bounds.

    Never discard actual constraints. Schema 3 serializes both year fields;
    legacy records with a non-null year bound cannot be interpreted as v1/v2.
    """
    if schema_version >= 3:
        return value
    if isinstance(value, tuple):
        return tuple(archive_data(item, schema_version) for item in value)
    if isinstance(value, list):
        return [archive_data(item, schema_version) for item in value]
    if isinstance(value, dict):
        result = {name: archive_data(item, schema_version) for name, item in value.items()}
        if isinstance(result.get("filters"), dict):
            for name in ("year_gte", "year_lte"):
                if result["filters"].get(name) is not None:
                    raise ValueError("Year filters require artifact schema 3")
                result["filters"].pop(name, None)
        return result
    return value
