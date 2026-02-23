# This runs before app.py to patch the gradio_client bug
import gradio_client.utils as u

def _patched_get_type(schema):
    if not isinstance(schema, dict):
        return "any"
    if "const" in schema:
        return "literal"
    if "enum" in schema:
        return "enum"
    types = schema.get("type", [])
    if isinstance(types, str):
        types = [types]
    if "null" in types:
        types = [t for t in types if t != "null"]
    if len(types) == 0:
        return "any"
    if len(types) == 1:
        return types[0]
    return "union"

u.get_type = _patched_get_type