# This runs before app.py to patch the gradio_client bug
import gradio_client.utils as u
import inspect, textwrap, types

src = inspect.getsource(u.get_type)
new_src = src.replace(
    'if "const" in schema:',
    'if isinstance(schema, dict) and "const" in schema:'
)
new_src = textwrap.dedent(new_src)
exec(compile(new_src, "<patch>", "exec"), u.__dict__)