import ast
import json
import re
from typing import Any, Dict, List, Tuple


_ACTION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.DOTALL)


def _split_top_level_commas(s: str) -> List[str]:
    """Split `a=1, b='x,y', c={...}` by top-level commas.

    Avoids splitting inside quotes/brackets/braces.
    """
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_single = False
    in_double = False
    escape = False
    for ch in s:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            continue

        if not in_single and not in_double:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return ""

    # Quoted strings / python literals
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw[1:-1]

    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None

    # JSON-ish collections
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(raw)
            except Exception:
                pass

    # Numbers
    try:
        if re.fullmatch(r"[-+]?\d+", raw):
            return int(raw)
        if re.fullmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", raw) or re.fullmatch(
            r"[-+]?\d+([eE][-+]?\d+)", raw
        ):
            return float(raw)
    except Exception:
        pass

    # Fallback: treat as string (critical for unquoted datetimes like 2025-06-07 07:05)
    return raw


def _parse_args(arg_str: str) -> Tuple[List[Any], Dict[str, Any]]:
    arg_str = (arg_str or "").strip()
    if not arg_str:
        return [], {}

    # Dict payload: {"k": "v"}
    if arg_str.startswith("{") and arg_str.endswith("}"):
        payload = None
        for parser in (json.loads, ast.literal_eval):
            try:
                payload = parser(arg_str)
                break
            except Exception:
                pass
        if isinstance(payload, dict):
            return [], payload

    parts = _split_top_level_commas(arg_str)

    # Kwargs payload: a=..., b=...
    if any("=" in p for p in parts):
        kwargs: Dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                # If mixed, treat as positional
                return [_parse_value(p) for p in parts], {}
            key, value = part.split("=", 1)
            kwargs[key.strip()] = _parse_value(value)
        return [], kwargs

    # Positional-only
    return [_parse_value(p) for p in parts], {}


def act_eval(action: str, tool_env: dict):
    """Safely execute tool calls without using eval().

    The original implementation used eval(action, tool_env), which is unsafe and
    also brittle: e.g. `minute=2025-06-07 07:05` triggers a SyntaxError due to
    leading zeros. Here we parse `tool_name(...)` and call the function directly.
    """
    try:
        m = _ACTION_RE.match(action or "")
        if not m:
            raise ValueError(f"Invalid action format: {action!r}")
        func_name = m.group(1)
        arg_str = m.group(2)

        func = tool_env.get(func_name)
        if not callable(func):
            raise NameError(f"Tool not found or not callable: {func_name}")

        args, kwargs = _parse_args(arg_str)
        return func(*args, **kwargs)
    except Exception as e:
        return str(e)