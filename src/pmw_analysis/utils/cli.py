"""
Helpers for command-line and config-file arguments.
"""


def parse_bool(value: str) -> bool:
    """
    Parse "true" or "false" (any case); `type=bool` would turn any non-empty string, including "false", into True.
    """
    match value.lower():
        case "true":
            return True
        case "false":
            return False
    raise ValueError(f"Expected true or false, got {value!r}")
