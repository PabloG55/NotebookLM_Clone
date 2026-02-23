# This runs before app.py to patch the gradio_client bug
import gradio_client.utils as u
from gradio_client.utils import APIInfoParseError

original = u._json_schema_to_python_type

def _safe_json_schema_to_python_type(schema, defs=None):
    try:
        if not isinstance(schema, dict):
            return "any"
        # handle anyOf with null (Optional types)
        if "anyOf" in schema:
            types = [s.get("type") for s in schema["anyOf"] if s.get("type") != "null"]
            return types[0] if types else "any"
        return original(schema, defs)
    except Exception:
        return "any"

u._json_schema_to_python_type = _safe_json_schema_to_python_type

def _safe_json_schema_to_python_type_public(schema):
    try:
        if not isinstance(schema, dict):
            return "any"
        return _safe_json_schema_to_python_type(schema, schema.get("$defs"))
    except Exception:
        return "any"

u.json_schema_to_python_type = _safe_json_schema_to_python_type_public